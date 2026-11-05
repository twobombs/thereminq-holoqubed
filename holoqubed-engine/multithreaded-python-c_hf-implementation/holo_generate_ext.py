"""
ThereminQ Holoqubed - Pure OpenCL Edition (Thread-Safe Concurrency Fix)
"""

import torch
import torch.nn as nn
import numpy as np
import time
import argparse
import re
import threading
import concurrent.futures
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import holo_ext
from holo_loader import HoloQueryPlanner 

TIME_CPP = 0.0
TIME_PYTORCH = 0.0
time_lock = threading.Lock() # Lock to ensure safe profiler logging across threads

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


class HoloLinear(nn.Module):
    def __init__(self, layer_name: str, planner: HoloQueryPlanner, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.layer_name = layer_name
        self.in_features = in_features
        self.out_features = out_features
        
        print(f"[{layer_name}] Pre-inflating into OpenCL VRAM...")
        coords, weights = planner._fetch_layer_data(layer_name)
        
        coords_tensor = torch.from_numpy(coords.astype(np.int64)).contiguous()
        weights_tensor = torch.from_numpy(weights.astype(np.float16).view(np.int16)).contiguous()
        
        self.native_layer = holo_ext.NativeHoloLayer(
            coords_tensor, weights_tensor, self.in_features, self.out_features, len(coords)
        )
        
        del coords_tensor
        del weights_tensor
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        global TIME_CPP, TIME_PYTORCH
        
        t0 = time.perf_counter()
        original_shape = x.shape
        x_flat = x.view(-1, self.in_features) 
        
        outputs = []
        
        for i in range(x_flat.shape[0]):
            in_vec = x_flat[i].detach().cpu().to(torch.float32).contiguous()
            
            t1 = time.perf_counter()
            
            # Execute OpenCL GEMV (Cleanly thread-safe via C++ dynamic buffers)
            out_vec = self.native_layer.forward(in_vec)
            
            t2 = time.perf_counter()
            
            with time_lock:
                TIME_PYTORCH += (t1 - t0)
                TIME_CPP += (t2 - t1)
            
            outputs.append(out_vec.to(x.device))
            t0 = time.perf_counter()
            
        output_flat = torch.stack(outputs)
        out = output_flat.view(*original_shape[:-1], self.out_features)
        
        if self.bias is not None:
            out += self.bias
            
        with time_lock:
            TIME_PYTORCH += (time.perf_counter() - t0)
            
        return out.to(x.dtype)


def inject_holographic_pathways(module: nn.Module, planner: HoloQueryPlanner, prefix: str = ""):
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            possible_names = map_hf_to_holo(full_name)
            matched_name = next((p for p in possible_names if p in planner.layers), None)
            
            if matched_name:
                bias_exists = child.bias is not None
                holo_layer = HoloLinear(matched_name, planner, child.in_features, child.out_features, bias_exists)
                setattr(module, name, holo_layer)
                continue
        inject_holographic_pathways(child, planner, full_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--trust_remote_code", action="store_true", default=False, help="Allow remote code execution")
    parser.add_argument("--holo_file", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="there is no spoon", help="Prompt to run across all threads")
    parser.add_argument("--threads", type=int, default=5, help="Number of concurrent generation threads")
    parser.add_argument("--max_tokens", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling threshold")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling threshold")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    if hasattr(config, "text_config"):
        for k, v in (config.text_config.items() if isinstance(config.text_config, dict) else config.text_config.__dict__.items()):
            if not hasattr(config, k): setattr(config, k, v)
    if not hasattr(config, "vocab_size"): config.vocab_size = 32000 
    if not hasattr(config, "hidden_size"): config.hidden_size = 2048
    if not hasattr(config, "pad_token_id"): config.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id else 0

    # CPU / FP32 shell strictly protects from PyTorch CUDA native asserts
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, 
        config=config, 
        torch_dtype=torch.float32, 
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code
    )
    
    holo_ext.init_opencl()
    planner = HoloQueryPlanner(args.holo_file)
    inject_holographic_pathways(model, planner)
    
    print("\n" + "="*50)
    print(f"[SPAWNING {args.threads} CONCURRENT GENERATION THREADS]")
    print(f"PROMPT: '{args.prompt}'")
    print("="*50)

    def generate_task(thread_id: int, prompt_text: str):
        print(f"[Thread {thread_id}] Preparing prompt...")
        inputs = tokenizer(prompt_text, return_tensors="pt")
        
        do_sample = args.temperature > 0.0
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=do_sample,
                temperature=args.temperature if do_sample else 1.0,
                top_p=args.top_p,
                top_k=args.top_k,
                pad_token_id=config.pad_token_id
            )
            
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        output_lines = [f"--- [Thread {thread_id} Output] ---", decoded, "-"*40 + "\n"]
        print("\n".join(output_lines))
        
        prompt_length = len(inputs['input_ids'][0])
        output_length = len(outputs[0])
        return output_length - prompt_length

    start_time = time.time()
    
    total_tokens_generated = 0
    # Execute identical prompt across specified number of threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = []
        for i in range(args.threads):
            futures.append(executor.submit(generate_task, i, args.prompt))
            
        for future in concurrent.futures.as_completed(futures):
            total_tokens_generated += future.result()
        
    total_time = time.time() - start_time
    tps = total_tokens_generated / total_time if total_time > 0 else 0
    
    print(f"\n[PROFILER DIAGNOSTICS - {args.threads} CONCURRENT THREADS]")
    print(f"Total Generation Time : {total_time:.2f} seconds")
    print(f"Total Tokens Generated: {total_tokens_generated}")
    print(f"Cluster Throughput    : {tps:.2f} TPS")
    print(f"Time spent in C++ GPU : {TIME_CPP:.2f} seconds")
    print(f"Time spent in PyTorch : {TIME_PYTORCH:.2f} seconds")
