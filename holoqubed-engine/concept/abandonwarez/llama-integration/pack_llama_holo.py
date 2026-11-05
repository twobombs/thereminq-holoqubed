import sys
import os
import re

# Priority path injection to load local llama.cpp libraries
sys.path.insert(0, os.path.abspath("llama.cpp/gguf-py"))
sys.path.insert(0, os.path.abspath("llama.cpp"))

import zipfile
import gguf
import io
import json
import numpy as np
import zstandard
import torch
import pickle
from huggingface_hub import snapshot_download

def pack_final():
    print("Engaging Ultimate Fusion Packer (PT Dense + Holo Sparse Edition)...")
    
    # 1. LOAD THE SPARSE HOLO ARCHIVE
    layers = {}
    with zipfile.ZipFile("model_169150.holo", 'r') as archive:
        dict_bytes = archive.read('_zstd_dictionary.dict')
        zstd_dict = zstandard.ZstdCompressionDict(dict_bytes)
        decompressor = zstandard.ZstdDecompressor(dict_data=zstd_dict)

        for filename in archive.namelist():
            if filename.endswith('.coords.npy.zst'):
                base = filename.replace('.coords.npy.zst', '')
                if base not in layers: layers[base] = {}
                layers[base]['coords'] = decompressor.decompress(archive.read(filename))
            elif filename.endswith('.weights.npy.zst'):
                base = filename.replace('.weights.npy.zst', '')
                if base not in layers: layers[base] = {}
                layers[base]['weights'] = decompressor.decompress(archive.read(filename))

    # 2. LOCATE AND LOAD THE EXACT PYTORCH DENSE WEIGHTS
    pt_search_paths = [
        "model_169150.pt",
        "../../../../nanochat-d34/model_169150.pt",
        "../../../nanochat-d34/model_169150.pt",
        "../../nanochat-d34/model_169150.pt",
        "../nanochat-d34/model_169150.pt",
        "/notebooks/thereminq/nanochat-d34/model_169150.pt",
        "/notebooks/nanochat-d34/model_169150.pt"
    ]
    
    pt_path = next((p for p in pt_search_paths if os.path.exists(p)), None)
    
    if not pt_path:
        raise RuntimeError("FATAL: model_169150.pt not found! Please copy it into this folder or update the paths.")
    
    print(f" -> Extracting mathematically precise Norms & Biases from {pt_path}...")
    pt_state_raw = torch.load(pt_path, map_location="cpu", weights_only=False)
    pt_state = pt_state_raw.get("model", pt_state_raw)
    if "_orig_mod." in list(pt_state.keys())[0]:
        pt_state = {k.replace("_orig_mod.", ""): v for k, v in pt_state.items()}

    def get_pt(name, shape=None, fallback_val=0.0):
        if name in pt_state:
            arr = pt_state[name].detach().float().numpy()
            return arr.reshape(shape) if shape else arr
        print(f"    [Warning] Missing {name}, faking...")
        return np.full(shape, fallback_val, dtype=np.float32) if shape else None

    wte_arr = get_pt("transformer.wte.weight")
    vocab_size, hidden_size = wte_arr.shape
    
    n_layers = max(int(re.search(r'\.(\d+)\.', k).group(1)) + 1 for k in pt_state.keys() if re.search(r'\.h\.(\d+)\.', k))
    ff_dim = hidden_size * 4 
    n_heads = hidden_size // 64

    print(f" -> GPT-2 Architecture: {n_layers} Layers | {hidden_size} Hidden | {ff_dim} FFN | {vocab_size} Vocab")

    writer = gguf.GGUFWriter("nanochat_holo_fused.gguf", "gpt2")
    writer.add_architecture()
    writer.add_context_length(2048)
    writer.add_vocab_size(vocab_size)
    writer.add_block_count(n_layers)
    writer.add_embedding_length(hidden_size)
    writer.add_feed_forward_length(ff_dim)
    writer.add_head_count(n_heads)
    writer.add_head_count_kv(n_heads)
    writer.add_layer_norm_eps(1e-5)
    writer.add_file_type(1) 

    print("\nDownloading Native Nanochat Tokenizer from HuggingFace...")
    tokens = [b""] * vocab_size
    loaded = False
    
    try:
        nanochat_dir = snapshot_download(repo_id="karpathy/nanochat-d34", allow_patterns=["token_bytes.pt", "tokenizer.pkl"])
        tok_bytes_path = os.path.join(nanochat_dir, "token_bytes.pt")
        
        if os.path.exists(tok_bytes_path):
            token_data = torch.load(tok_bytes_path, map_location="cpu", weights_only=False)
            for i, b in enumerate(token_data):
                if i < vocab_size: tokens[i] = b if isinstance(b, bytes) else str(b).encode('utf-8')
            loaded = True
    except Exception as e:
        print(f" -> WARNING: Failed to download from HuggingFace: {e}")

    seen_tokens = set()
    for i in range(vocab_size):
        if not tokens[i]: tokens[i] = f"[PAD_{i}]".encode('utf-8')
        elif isinstance(tokens[i], str): tokens[i] = tokens[i].encode('utf-8')
            
        orig_token = tokens[i]
        suffix = 0
        while tokens[i] in seen_tokens:
            suffix += 1
            tokens[i] = orig_token + f"_dup{suffix}".encode('utf-8')
        seen_tokens.add(tokens[i])

    writer.add_tokenizer_model("llama") 
    writer.add_token_list(tokens)
    writer.add_token_scores([0.0] * vocab_size) 
    writer.add_token_types([1] * vocab_size) 
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    writer.add_chat_template("{{ prompt }}")

    print("\nFusing PT Dense with Holoqubed Sparse Matrices...")
    
    # EXACT EMBEDDINGS (Fix applied here!)
    writer.add_tensor("token_embd.weight", wte_arr)
    writer.add_tensor("position_embd.weight", get_pt("transformer.wpe.weight", (2048, hidden_size)))
    
    # Output Head
    if "lm_head.weight" in pt_state:
        writer.add_tensor("output.weight", get_pt("lm_head.weight", (vocab_size, hidden_size)))
    else:
        writer.add_tensor("output.weight", wte_arr)

    writer.add_tensor("output_norm.weight", get_pt("transformer.ln_f.weight", (hidden_size,), fallback_val=1.0))
    writer.add_tensor("output_norm.bias", get_pt("transformer.ln_f.bias", (hidden_size,), fallback_val=0.0))

    MAGIC_HOLO = np.array([0x484F4C4F51554244], dtype=np.uint64).tobytes()

    for i in range(n_layers):
        # Dense QKV & Norms from Native PyTorch model
        writer.add_tensor(f"blk.{i}.attn_qkv.weight", get_pt(f"transformer.h.{i}.attn.c_attn.weight", (3*hidden_size, hidden_size)))
        writer.add_tensor(f"blk.{i}.attn_qkv.bias", get_pt(f"transformer.h.{i}.attn.c_attn.bias", (3*hidden_size,)))
        
        writer.add_tensor(f"blk.{i}.attn_norm.weight", get_pt(f"transformer.h.{i}.ln_1.weight", (hidden_size,), fallback_val=1.0))
        writer.add_tensor(f"blk.{i}.attn_norm.bias", get_pt(f"transformer.h.{i}.ln_1.bias", (hidden_size,), fallback_val=0.0))
        
        writer.add_tensor(f"blk.{i}.ffn_norm.weight", get_pt(f"transformer.h.{i}.ln_2.weight", (hidden_size,), fallback_val=1.0))
        writer.add_tensor(f"blk.{i}.ffn_norm.bias", get_pt(f"transformer.h.{i}.ln_2.bias", (hidden_size,), fallback_val=0.0))

        # SPARSE HOLOQUBED MAGIC BLOCKS (Executed on GPU)
        targets = [
            (f"transformer.h.{i}.attn.c_proj.weight", f"blk.{i}.attn_output", (hidden_size, hidden_size)),
            (f"transformer.h.{i}.mlp.c_fc.weight", f"blk.{i}.ffn_up", (ff_dim, hidden_size)),
            (f"transformer.h.{i}.mlp.c_proj.weight", f"blk.{i}.ffn_down", (hidden_size, ff_dim))
        ]

        for pt_key, gguf_name, pt_shape in targets:
            matched_holo_key = next((k for k in layers.keys() if pt_key in k), None)
            
            if matched_holo_key and 'coords' in layers[matched_holo_key]:
                coords_raw = np.load(io.BytesIO(layers[matched_holo_key]['coords']), allow_pickle=False)
                weights = np.load(io.BytesIO(layers[matched_holo_key]['weights']), allow_pickle=False)
                
                coords = coords_raw.astype(np.uint32)
                num_elements = np.array([len(coords)], dtype=np.uint64).tobytes()
                
                fused_payload = MAGIC_HOLO + num_elements + coords.tobytes() + weights.tobytes()
                expected_bytes = pt_shape[0] * pt_shape[1] * 4
                padded_payload = fused_payload + b'\x00' * (expected_bytes - len(fused_payload))
                
                fused_array = np.zeros(pt_shape, dtype=np.float32)
                byte_view = fused_array.view(np.uint8).reshape(-1)
                byte_view[:len(padded_payload)] = np.frombuffer(padded_payload, dtype=np.uint8)
                
                writer.add_tensor(f"{gguf_name}.weight", fused_array)
                print(f" -> Embedded HOLO-GPU: {gguf_name}.weight")
            else:
                writer.add_tensor(f"{gguf_name}.weight", get_pt(pt_key, pt_shape))
            
            writer.add_tensor(f"{gguf_name}.bias", get_pt(pt_key.replace('.weight', '.bias'), pt_shape[0]))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print("="*50)
    print("nanochat_holo_fused.gguf is ready! Brain restored. GPU Math mapped.")

if __name__ == "__main__":
    pack_final()
