"""
ThereminQ Holoqubed - Pure OpenCL Edition
Scatters sparse Morton coordinates into dense OpenCL VRAM buffers during init, 
and executes purely via OpenCL GEMV kernels to bypass CPU limitations.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import argparse
import re
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
        
        # C++ handles the one-time scatter and permanently pins the dense matrix in OpenCL
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
            TIME_PYTORCH += (t1 - t0)
            
            # Execute OpenCL GEMV
            out_vec = self.native_layer.forward(in_vec)
            
            t2 = time.perf_counter()
            TIME_CPP += (t2 - t1)
            
            outputs.append(out_vec.to(x.device))
            t0 = time.perf_counter()
            
        output_flat = torch.stack(outputs)
        out = output_flat.view(*original_shape[:-1], self.out_features)
        
        if self.bias is not None:
            out += self.bias
            
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
    parser.add_argument("--holo_file", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="there is no spoon")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    if hasattr(config, "text_config"):
        for k, v in (config.text_config.items() if isinstance(config.text_config, dict) else config.text_config.__dict__.items()):
            if not hasattr(config, k): setattr(config, k, v)
    if not hasattr(config, "vocab_size"): config.vocab_size = 32000 
    if not hasattr(config, "hidden_size"): config.hidden_size = 2048
    if not hasattr(config, "pad_token_id"): config.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id else 0

    model = AutoModelForCausalLM.from_pretrained(args.model_id, config=config, dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True)
    
    holo_ext.init_opencl()
    planner = HoloQueryPlanner(args.holo_file)
    inject_holographic_pathways(model, planner)
    
    print("\n" + "="*50)
    print(f"PROMPT: {args.prompt}")
    print("="*50)
    
    inputs = tokenizer(args.prompt, return_tensors="pt")
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=True,
            temperature=0.7,
            pad_token_id=config.pad_token_id
        )
        
    gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("\n" + "="*50)
    print(gen_text)
    print("="*50)
    
    total_time = time.time() - start_time
    prompt_length = len(inputs['input_ids'][0])
    output_length = len(outputs[0])
    tokens_generated = output_length - prompt_length
    tps = tokens_generated / total_time if total_time > 0 else 0
    
    print(f"\n[PROFILER DIAGNOSTICS]")
    print(f"Total Generation Time : {total_time:.2f} seconds")
    print(f"Tokens Generated      : {tokens_generated}")
    print(f"Tokens Per Second     : {tps:.2f} TPS")
    print(f"Time spent in C++ GPU : {TIME_CPP:.2f} seconds ({TIME_CPP/total_time*100:.1f}%)")
    print(f"Time spent in PyTorch : {TIME_PYTORCH:.2f} seconds ({TIME_PYTORCH/total_time*100:.1f}%)")
