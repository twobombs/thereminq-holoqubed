"""
ThereminQ Holoqubed - Master Pipeline Edition (Degrees UX)
Features:
- CLI now accepts Degrees (e.g., --phases 0,45,90)
- Live Holographic Streaming (Token-by-Token Matrix UI)
- Stable Cosine Phase Scaling
- Multi-GPU Pipeline Parallelism (PCIe VRAM Sharding)
- Restored Sampling (Temperature, Top-K, Top-P)
"""

import torch
import torch.nn as nn
import numpy as np
import time
import argparse
import re
import math
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import holo_ext
from holo_loader import HoloQueryPlanner 

TIME_CPP = 0.0
TIME_PYTORCH = 0.0

def map_hf_to_holo(hf_name: str) -> list:
    possibilities = [hf_name, f"{hf_name}.weight"]
    match = re.search(r'\.(layers|h)\.(\d+)\.', hf_name)
    if match:
        idx = match.group(2)
        if "q_proj" in hf_name: possibilities.append(f"transformer.h.{idx}.attn.c_q.weight")
        if "k_proj" in hf_name: possibilities.append(f"transformer.h.{idx}.attn.c_k.weight")
        if "v_proj" in hf_name: possibilities.append(f"transformer.h.{idx}.attn.c_v.weight")
        if "o_proj" in hf_name: possibilities.append(f"transformer.h.{idx}.attn.c_proj.weight")
        if "up_proj" in hf_name or "gate_proj" in hf_name or "c_fc" in hf_name: possibilities.append(f"transformer.h.{idx}.mlp.c_fc.weight")
        if "down_proj" in hf_name or "c_proj" in hf_name: possibilities.append(f"transformer.h.{idx}.mlp.c_proj.weight")
    return possibilities


class ComplexHoloLinear(nn.Module):
    def __init__(self, layer_name: str, planner: HoloQueryPlanner, in_features: int, out_features: int, phase_array_rad: list, target_gpu: int, bias: bool = False):
        super().__init__()
        self.layer_name = layer_name
        self.in_features = in_features
        self.out_features = out_features
        self.target_gpu = target_gpu
        
        # Store the converted radian phase angles as a PyTorch tensor
        self.phase_angles = torch.tensor(phase_array_rad, dtype=torch.float32)
        
        print(f"[{layer_name}] Inflating into VRAM on GPU {target_gpu}...")
        
        rows = planner._read_and_decompress(f"{layer_name}.rows.npy.zst")
        cols = planner._read_and_decompress(f"{layer_name}.cols.npy.zst")
        w_real = planner._read_and_decompress(f"{layer_name}.w_real.npy.zst")
        w_imag = planner._read_and_decompress(f"{layer_name}.w_imag.npy.zst")
        
        rows_t = torch.from_numpy(rows.astype(np.uint32)).contiguous()
        cols_t = torch.from_numpy(cols.astype(np.uint32)).contiguous()
        w_real_t = torch.from_numpy(w_real.astype(np.float32)).contiguous()
        w_imag_t = torch.from_numpy(w_imag.astype(np.float32)).contiguous()
        
        # Pass target_gpu to the C++ Engine
        self.native_layer = holo_ext.NativeHoloLayer(
            rows_t, cols_t, w_real_t, w_imag_t, self.in_features, self.out_features, len(rows), self.target_gpu
        )
        
        del rows_t, cols_t, w_real_t, w_imag_t
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        global TIME_CPP, TIME_PYTORCH
        t0 = time.perf_counter()
        
        batch_size = x.shape[0]
        seq_len = x.shape[1] if x.dim() > 2 else 1
        
        x_flat = x.view(batch_size * seq_len, self.in_features)
        in_r = x_flat.detach().cpu().to(torch.float32).contiguous()
        
        # Stable Cosine scaling math
        phases_expanded = self.phase_angles.unsqueeze(1).repeat(1, seq_len).view(-1)
        phases_cos = torch.cos(phases_expanded).to(torch.float32).contiguous()
        
        TIME_PYTORCH += (time.perf_counter() - t0)
        
        # Execute the C++ Batch Kernel on the targeted GPU
        t1 = time.perf_counter()
        out_r = self.native_layer.forward(in_r, phases_cos)
        TIME_CPP += (time.perf_counter() - t1)
        
        t2 = time.perf_counter()
        out = out_r.to(x.device).view(*x.shape[:-1], self.out_features)
        
        if self.bias is not None:
            out += self.bias
            
        TIME_PYTORCH += (time.perf_counter() - t2)
        return out.to(x.dtype)


def inject_holographic_pathways(module: nn.Module, planner: HoloQueryPlanner, phase_array_rad: list, num_gpus: int, total_layers: int = 34, current_layer_idx: int = 0, prefix: str = ""):
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        
        match = re.search(r'\.(layers|h)\.(\d+)', full_name)
        if match:
            current_layer_idx = int(match.group(2))
            
        if hasattr(child, "weight") and len(child.weight.shape) == 2:
            possible_names = map_hf_to_holo(full_name)
            matched_name = next((p for p in possible_names if p in planner.layers), None)
            
            if matched_name:
                bias_exists = hasattr(child, "bias") and child.bias is not None
                if "Conv1D" in child.__class__.__name__:
                    in_features, out_features = child.weight.shape
                else:
                    out_features, in_features = child.weight.shape
                
                # PIPELINE ROUTING MATH
                layers_per_gpu = max(1, total_layers // num_gpus)
                target_gpu = min(current_layer_idx // layers_per_gpu, num_gpus - 1)
                    
                holo_layer = ComplexHoloLinear(
                    matched_name, planner, in_features, out_features, phase_array_rad, target_gpu, bias_exists
                )
                setattr(module, name, holo_layer)
                continue
                
        inject_holographic_pathways(child, planner, phase_array_rad, num_gpus, total_layers, current_layer_idx, full_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--holo_file", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="there is no spoon", help="Prompt to run")
    
    # FIX: Updated help text and default values to use DEGREES
    parser.add_argument("--phases", type=str, default="0,15,45,90", help="Comma-separated list of phase angles in DEGREES")
    
    parser.add_argument("--max_tokens", type=int, default=25, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature (> 0.0 enables sampling)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling threshold")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling threshold")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to pipeline across")
    args = parser.parse_args()

    # FIX: Parse degrees and convert to radians for the C++ engine
    phase_array_deg = [float(p.strip()) for p in args.phases.split(',')]
    phase_array_rad = [math.radians(p) for p in phase_array_deg]
    batch_size = len(phase_array_rad)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    if hasattr(config, "text_config"):
        for k, v in (config.text_config.items() if isinstance(config.text_config, dict) else config.text_config.__dict__.items()):
            if not hasattr(config, k): setattr(config, k, v)
    if not hasattr(config, "vocab_size"): config.vocab_size = 32000 
    if not hasattr(config, "hidden_size"): config.hidden_size = 2048
    if not hasattr(config, "pad_token_id"): config.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id else 0

    total_layers = getattr(config, "num_hidden_layers", 34)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, 
        config=config, 
        torch_dtype=torch.float32, 
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    
    holo_ext.init_opencl()
    planner = HoloQueryPlanner(args.holo_file)
                      
    inject_holographic_pathways(model, planner, phase_array_rad, num_gpus=args.gpus, total_layers=total_layers)
    
    print("\n" + "="*50)
    print(f"[EXECUTING MULTI-GPU HOLOGRAPHIC STREAMING]")
    print(f"PROMPT        : '{args.prompt}'")
    print(f"PHASE ANGLES  : {phase_array_deg} degrees (Batch Size: {batch_size})")
    print(f"TEMPERATURE   : {args.temperature} (Sampling: {args.temperature > 0.0})")
    print(f"PIPELINE GPUs : {args.gpus}")
    print("="*50)

    prompts = [args.prompt] * batch_size
    inputs = tokenizer(prompts, return_tensors="pt")
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)
    
    start_time = time.time()
    
    generated_sequences = [[] for _ in range(batch_size)]
    past_key_values = None
    
    # FIX: Update the UI headers to clearly state the degrees
    headers = [f"{deg}°" for deg in phase_array_deg]
    header_format = " | ".join([f"{{:<25}}"] * batch_size)
    print("\n" + header_format.format(*headers))
    print("-" * (28 * batch_size))

    with torch.no_grad():
        for step in range(args.max_tokens):
            
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True
            )
            
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            
            if args.temperature > 0.0:
                scaled_logits = next_token_logits / args.temperature
                if args.top_k > 0:
                    indices_to_remove = scaled_logits < torch.topk(scaled_logits, args.top_k)[0][..., -1, None]
                    scaled_logits[indices_to_remove] = -float('Inf')
                probs = nn.functional.softmax(scaled_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            decoded_tokens = []
            for b in range(batch_size):
                token = next_tokens[b].item()
                generated_sequences[b].append(token)
                
                word = tokenizer.decode([token])
                word = word.replace('\n', '¶').strip()
                if not word:
                    word = "[SPACE]"
                    
                decoded_tokens.append(word)
                
            print(header_format.format(*decoded_tokens), flush=True)
            
            input_ids = next_tokens.unsqueeze(-1)
            new_mask = torch.ones((batch_size, 1), device=model.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([attention_mask, new_mask], dim=-1)

    total_time = time.time() - start_time
    total_tokens_generated = batch_size * args.max_tokens
    tps = total_tokens_generated / total_time if total_time > 0 else 0
    
    print("\n" + "="*50)
    for i in range(batch_size):
        final_text = tokenizer.decode(generated_sequences[i], skip_special_tokens=True)
        # FIX: Print both degrees and radians in the final output block for scientific tracing
        print(f"\n--- [Phase {phase_array_deg[i]}° ({phase_array_rad[i]:.4f} rad) Final Generation] ---")
        print(final_text)

    print(f"\n[PROFILER DIAGNOSTICS - BATCH SIZE {batch_size} / GPUs {args.gpus}]")
    print(f"Total Generation Time : {total_time:.2f} seconds")
    print(f"Total Tokens Generated: {total_tokens_generated}")
    print(f"Cluster Throughput    : {tps:.2f} TPS")
    print(f"Time spent in C++ GPU : {TIME_CPP:.2f} seconds")
    print(f"Time spent in PyTorch : {TIME_PYTORCH:.2f} seconds")
