import sys
import os

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
from transformers import AutoModelForCausalLM
from huggingface_hub import snapshot_download

NAME_MAP = {
    "transformer.wte.weight": "token_embd.weight",
    "lm_head.weight": "output.weight",
    "transformer.ln_f.weight": "output_norm.weight",
    "transformer.ln_f.bias": "output_norm.bias",
}

def map_name(name):
    suffix = "weight" if name.endswith(".weight") else "bias" if name.endswith(".bias") else ""
    if name in NAME_MAP: return NAME_MAP[name]
    if name.startswith("transformer.h."):
        parts = name.split('.')
        idx = parts[2]
        if "attn.c_q" in name: return f"blk.{idx}.attn_q.{suffix}"
        if "attn.c_k" in name: return f"blk.{idx}.attn_k.{suffix}"
        if "attn.c_v" in name: return f"blk.{idx}.attn_v.{suffix}"
        if "attn.c_proj" in name: return f"blk.{idx}.attn_output.{suffix}"
        if "mlp.c_fc" in name: return f"blk.{idx}.ffn_up.{suffix}"
        if "mlp.c_proj" in name: return f"blk.{idx}.ffn_down.{suffix}"
        if "ln_1" in name: return f"blk.{idx}.attn_norm.{suffix}"
        if "ln_2" in name: return f"blk.{idx}.ffn_norm.{suffix}"
    return name

def pack_final():
    print("Loading cached base model from HuggingFace to extract LayerNorms and exact dimensions...")
    model_id = "twobombs/nanochat-d34-sft-hf"
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=True)
    state_dict = model.state_dict()

    writer = gguf.GGUFWriter("nanochat_holo_fused.gguf", "llama")
    
    # Core Architecture Hyperparameters
    hidden_size = getattr(model.config, "hidden_size", getattr(model.config, "n_embd", 2048))
    ff_dim = getattr(model.config, "intermediate_size", getattr(model.config, "n_inner", 8192))
    n_heads = getattr(model.config, "num_attention_heads", getattr(model.config, "n_head", 32))
    n_kv_heads = getattr(model.config, "num_key_value_heads", n_heads)
    eps = getattr(model.config, "rms_norm_eps", getattr(model.config, "layer_norm_eps", 1e-5))
    ctx_len = getattr(model.config, "max_position_embeddings", getattr(model.config, "n_positions", 2048))
    vocab_size = getattr(model.config, "vocab_size", 32000)
    n_layers = getattr(model.config, "num_hidden_layers", getattr(model.config, "n_layer", 34))

    writer.add_architecture()
    writer.add_context_length(ctx_len)
    writer.add_vocab_size(vocab_size)
    writer.add_block_count(n_layers)
    writer.add_embedding_length(hidden_size)
    writer.add_feed_forward_length(ff_dim)
    writer.add_head_count(n_heads)
    writer.add_head_count_kv(n_kv_heads)
    writer.add_layer_norm_rms_eps(eps)
    writer.add_file_type(1) # GGUF Code 1 = F16 Model

    print("\nDownloading Tokenizer configuration...")
    local_dir = snapshot_download(model_id, allow_patterns=["*token*", "config.json"])
    
    print("Injecting Vocabulary with strict Array Sanitization...")
    tok_path = os.path.join(local_dir, "tokenizer.json")
    with open(tok_path, "r", encoding="utf-8") as f:
        tok_json = json.load(f)
        
    vocab_dict = tok_json.get("model", {}).get("vocab", {})
    merges = tok_json.get("model", {}).get("merges", [])
    
    max_id = max(vocab_dict.values()) if vocab_dict else 0
    tokens = [""] * (max_id + 1)
    for tok, tid in vocab_dict.items():
        tokens[tid] = str(tok)
        
    for added in tok_json.get("added_tokens", []):
        tid = added["id"]
        tok = str(added["content"])
        if tid >= len(tokens):
            tokens.extend([""] * (tid - len(tokens) + 1))
        tokens[tid] = tok
        
    writer.add_tokenizer_model("gpt2") 
    writer.add_token_list(tokens)
    
    if merges:
        # THE FIX: Absolute sanitization of the merges array
        clean_merges = []
        for m in merges:
            if isinstance(m, list):
                clean_merges.append(" ".join([str(x) for x in m]))
            else:
                clean_merges.append(str(m))
        
        if clean_merges:
            writer.add_token_merges(clean_merges)

    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)

    print("\nOpening Holo Archive...")
    with zipfile.ZipFile("model_169150.holo", 'r') as archive:
        dict_bytes = archive.read('_zstd_dictionary.dict')
        zstd_dict = zstandard.ZstdCompressionDict(dict_bytes)
        decompressor = zstandard.ZstdDecompressor(dict_data=zstd_dict)

        layers = {}
        for filename in archive.namelist():
            if filename.endswith('.coords.npy.zst'):
                base = filename.replace('.coords.npy.zst', '')
                if base not in layers: layers[base] = {}
                layers[base]['coords'] = archive.read(filename)
            elif filename.endswith('.weights.npy.zst'):
                base = filename.replace('.weights.npy.zst', '')
                if base not in layers: layers[base] = {}
                layers[base]['weights'] = archive.read(filename)

    print("Writing mathematically precise GGUF Tensors...")
    for hf_name, tensor in state_dict.items():
        gguf_name = map_name(hf_name)
        shape = tuple(reversed(tensor.shape))
        
        base_name = hf_name.replace(".weight", "")
        if base_name in layers and 'coords' in layers[base_name] and hf_name.endswith(".weight"):
            raw_c = decompressor.decompress(layers[base_name]['coords'])
            coords = np.load(io.BytesIO(raw_c), allow_pickle=False)

            raw_w = decompressor.decompress(layers[base_name]['weights'])
            weights = np.load(io.BytesIO(raw_w), allow_pickle=False)

            num_elements = np.array([len(coords)], dtype=np.uint64).tobytes()
            fused_payload = num_elements + coords.tobytes() + weights.tobytes()

            dense_elements = 1
            for dim in shape: dense_elements *= dim
            expected_bytes = dense_elements * 8
            
            padded_payload = fused_payload + b'\x00' * (expected_bytes - len(fused_payload))
            fused_array = np.frombuffer(padded_payload, dtype=np.int8)
            
            writer.add_tensor(gguf_name, fused_array, raw_shape=shape, raw_dtype=41)
        else:
            t_data = tensor.cpu().float().numpy() if tensor.dtype == torch.bfloat16 else tensor.cpu().numpy()
            writer.add_tensor(gguf_name, t_data)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print("="*50)
    print("nanochat_holo_fused.gguf is architecturally perfect AND natively tokenized!")

if __name__ == "__main__":
    pack_final()
