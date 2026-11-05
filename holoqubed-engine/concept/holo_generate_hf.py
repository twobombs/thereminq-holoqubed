"""
ThereminQ Holoqubed - Text Generation Engine
Monkey-patches a Hugging Face PreTrainedModel, replacing dense nn.Linear layers 
with custom HoloLinear PyTorch modules backed by the OpenCL SpMV engine.
"""

import torch
import torch.nn as nn
import numpy as np
import pyopencl as cl
import time
import argparse
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Import the planner and kernel string from your previous script
from holo_loader import HoloQueryPlanner, SPMV_KERNEL_CODE

class HoloLinear(nn.Module):
    def __init__(self, layer_name: str, planner: HoloQueryPlanner, ctx: cl.Context, queue: cl.CommandQueue, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.layer_name = layer_name
        self.ctx = ctx
        self.queue = queue
        self.in_features = in_features
        self.out_features = out_features
        
        # 1. Fetch the surviving weights and coords for this specific layer
        print(f"[{layer_name}] Forging HoloLinear Module...")
        coords, weights = planner._fetch_layer_data(layer_name)
        
        self.num_elements = np.int32(len(coords))
        
        # 2. Pre-allocate the static OpenCL Zero-Copy buffers
        mf = cl.mem_flags
        cl_coords = coords.astype(np.uint64)
        cl_weights_raw = weights.astype(np.float16).view(np.uint16)
        
        self.coords_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cl_coords)
        self.weights_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cl_weights_raw)
        
        # 3. Compile the OpenCL Program once for this layer
        self.prg = cl.Program(self.ctx, SPMV_KERNEL_CODE).build()
        
        # 4. Handle standard PyTorch bias if the original layer had it
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The Intercepted Forward Pass.
        Translates PyTorch tensors to OpenCL, executes SpMV, and translates back.
        """
        original_shape = x.shape
        x_flat = x.view(-1, self.in_features) 
        
        output_flat = torch.zeros((x_flat.shape[0], self.out_features), device=x.device, dtype=torch.float32)
        
        mf = cl.mem_flags
        
        for i in range(x_flat.shape[0]):
            input_np = x_flat[i].detach().cpu().numpy().astype(np.float32)
            
            in_vec_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=input_np)
            out_vec_bytes = self.out_features * 4
            out_vec_buf = cl.Buffer(self.ctx, mf.READ_WRITE, size=out_vec_bytes)
            
            cl.enqueue_fill_buffer(self.queue, out_vec_buf, np.zeros(1, dtype=np.float32), 0, out_vec_bytes)
            
            self.prg.spmv_holo_weights(
                self.queue, 
                (int(self.num_elements),), 
                None, 
                self.coords_buf, 
                self.weights_buf, 
                in_vec_buf,
                out_vec_buf,
                self.num_elements
            )
            
            result_np = np.empty((self.out_features,), dtype=np.float32)
            cl.enqueue_copy(self.queue, result_np, out_vec_buf).wait()
            
            output_flat[i] = torch.from_numpy(result_np).to(x.device)
            
        out = output_flat.view(*original_shape[:-1], self.out_features)
        
        if self.bias is not None:
            out += self.bias
            
        return out.to(x.dtype)


def inject_holographic_pathways(module: nn.Module, planner: HoloQueryPlanner, ctx: cl.Context, queue: cl.CommandQueue, prefix: str = ""):
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        
        if isinstance(child, nn.Linear) and full_name in planner.layers:
            bias_exists = child.bias is not None
            
            holo_layer = HoloLinear(
                layer_name=full_name,
                planner=planner,
                ctx=ctx,
                queue=queue,
                in_features=child.in_features,
                out_features=child.out_features,
                bias=bias_exists
            )
            
            setattr(module, name, holo_layer)
        else:
            inject_holographic_pathways(child, planner, ctx, queue, full_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ThereminQ Holoqubed - Text Generation")
    parser.add_argument("--model_id", type=str, required=True, help="Hugging Face Model ID (e.g., Qwen/Qwen3.5-9B)")
    parser.add_argument("--trust_remote_code", action="store_true", default=False, help="Allow remote code execution")
    parser.add_argument("--holo_file", type=str, required=True, help="Path to your .holo file")
    parser.add_argument("--prompt", type=str, default="The future of artificial intelligence is", help="Text to generate from")
    args = parser.parse_args()

    print("1. Loading PyTorch/Hugging Face ecosystem...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    
    # ---------------------------------------------------------
    # THE FIX: Dynamic Config Bubble-Up
    # ---------------------------------------------------------
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    
    if hasattr(config, "text_config"):
        print("   -> Multimodal Config detected. Bubbling up text_config attributes...")
        text_cfg = config.text_config
        # Handle both dictionary and object attribute structures safely
        iterable_cfg = text_cfg.items() if isinstance(text_cfg, dict) else text_cfg.__dict__.items()
        
        for key, value in iterable_cfg:
            if not hasattr(config, key):
                setattr(config, key, value)
                
    # Fallbacks for the absolute essentials just in case the config is completely shattered
    if not hasattr(config, "vocab_size"): config.vocab_size = 248320
    if not hasattr(config, "hidden_size"): config.hidden_size = 4096
    if not hasattr(config, "pad_token_id"): config.pad_token_id = 151643
    # ---------------------------------------------------------

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, 
        config=config, 
        dtype=torch.float32, 
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code
    )
    
    print("\n2. Booting Holoqubed Engine & OpenCL...")
    planner = HoloQueryPlanner(args.holo_file)
    platforms = cl.get_platforms()
    device = platforms[0].get_devices()[0]
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)
    
    print("\n3. Performing Surgery: Injecting OpenCL SpMV layers into PyTorch...")
    inject_holographic_pathways(model, planner, ctx, queue)
    
    print("\nModel successfully lobotomized and re-wired with holographic pathways.")
    print("Moving model to standard CPU/GPU execution mode...\n")

    print("="*50)
    print(f"PROMPT: {args.prompt}")
    print("="*50)
    
    inputs = tokenizer(args.prompt, return_tensors="pt")
    
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        
    gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(gen_text)
    print(f"\nGeneration completed in {time.time() - start_time:.2f} seconds.")
