"""
ThereminQ Holoqubed - The Geometry Forge
Transforms Dense Flatland AI models into Complex Hilbert Phase Space.
Pre-computes Hilbert curve topologies and Cartesian Phase (Real/Imaginary) 
for pure FP32 zero-branching OpenCL execution.
Supports .gguf, .safetensors, and PyTorch (.pt, .bin) input formats.
"""

import numpy as np
import gguf
import os
import argparse
import zipfile
import io
import gc
import concurrent.futures
import zstandard
from numba import njit, prange

# =============================================================================
# 1. THE HILBERT TOPOLOGY ENGINE (JIT Compiled to C for extreme speed)
# =============================================================================

@njit
def rot(n, x, y, rx, ry):
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y
        return y, x
    return x, y

@njit(parallel=True)
def map_to_hilbert(rows, cols, grid_size):
    num_elements = len(rows)
    distances = np.zeros(num_elements, dtype=np.uint64)
    
    for i in prange(num_elements):
        x = rows[i]
        y = cols[i]
        d = 0
        s = grid_size // 2
        
        while s > 0:
            rx = (x & s) > 0
            ry = (y & s) > 0
            d += s * s * ((3 * rx) ^ ry)
            x, y = rot(s, x, y, rx, ry)
            s //= 2
            
        distances[i] = d
        
    return distances

def get_next_power_of_2(n):
    return 1 if n == 0 else 2**(n - 1).bit_length()

# =============================================================================
# 2. THE PHASE INJECTOR
# =============================================================================

def apply_complex_phase(magnitudes: np.ndarray, rows: np.ndarray, cols: np.ndarray, mode: str):
    if mode == "flatland":
        return magnitudes.astype(np.float32), np.zeros_like(magnitudes, dtype=np.float32)
        
    elif mode == "quantum":
        theta = np.mod(rows + cols, 2 * np.pi).astype(np.float32)
        w_real = magnitudes * np.cos(theta)
        w_imag = magnitudes * np.sin(theta)
        return w_real.astype(np.float32), w_imag.astype(np.float32)

# =============================================================================
# 3. MULTIPROCESSING FORGE WORKER
# =============================================================================

def forge_layer_worker(name: str, data: np.ndarray, is_bypassed: bool, std_factor: float, max_sparsity: float, zstd_level: int, phase_mode: str) -> dict:
    result = {
        "name": name,
        "files": {}, 
        "orig_params": data.size,
        "surv_params": 0,
        "uncompressed_bytes": 0,
        "compressed_bytes": 0,
        "log": ""
    }
    
    compressor = zstandard.ZstdCompressor(level=zstd_level)
    
    if is_bypassed:
        dense_data = data.astype(np.float32)
        result["surv_params"] = dense_data.size
        
        buf = io.BytesIO()
        np.save(buf, dense_data, allow_pickle=False)
        c_data = compressor.compress(buf.getvalue())
        
        result["uncompressed_bytes"] = buf.tell()
        result["compressed_bytes"] = len(c_data)
        result["files"][f"{name}.npy.zst"] = c_data
        result["log"] = f"  [BYPASSED] {name} | Saved as Zstd Dense (Flatland FP32)"
        return result

    abs_data = np.abs(data)
    layer_std = np.std(data) + 1e-8 
    
    std_threshold = std_factor * layer_std
    percentile_threshold = np.percentile(abs_data, max_sparsity)
    dynamic_threshold = min(std_threshold, percentile_threshold)
    
    mask = abs_data > dynamic_threshold
    surviving_weights = data[mask]
    result["surv_params"] = surviving_weights.size
    
    if result["surv_params"] == 0:
        result["log"] = f"  [DELETED] {name} (100% Sparsity)"
        return result

    dense_indices = np.argwhere(mask)
    if dense_indices.shape[1] != 2:
        rows = np.zeros(len(dense_indices), dtype=np.uint32)
        cols = dense_indices[:, 0].astype(np.uint32)
    else:
        rows = dense_indices[:, 0].astype(np.uint32)
        cols = dense_indices[:, 1].astype(np.uint32)

    max_dim = max(data.shape[0] if len(data.shape) > 0 else 1, data.shape[1] if len(data.shape) > 1 else 1)
    grid_size = get_next_power_of_2(max_dim)
    
    hilbert_distances = map_to_hilbert(rows, cols, grid_size)
    
    sort_order = np.argsort(hilbert_distances)
    sorted_rows = rows[sort_order]
    sorted_cols = cols[sort_order]
    sorted_weights = surviving_weights[sort_order]
    
    w_real, w_imag = apply_complex_phase(sorted_weights, sorted_rows, sorted_cols, phase_mode)
    
    arrays_to_pack = {
        "rows": sorted_rows,
        "cols": sorted_cols,
        "w_real": w_real,
        "w_imag": w_imag
    }
    
    for suffix, arr in arrays_to_pack.items():
        buf = io.BytesIO()
        np.save(buf, arr, allow_pickle=False)
        raw_bytes = buf.getvalue()
        c_bytes = compressor.compress(raw_bytes)
        
        result["uncompressed_bytes"] += len(raw_bytes)
        result["compressed_bytes"] += len(c_bytes)
        result["files"][f"{name}.{suffix}.npy.zst"] = c_bytes

    sparsity = 100.0 * (1.0 - (result["surv_params"] / result["orig_params"]))
    result["log"] = f"  [FORGED] {name} | Phase: {phase_mode.upper()} | Sparsity: {sparsity:.2f}% | Survivors: {result['surv_params']:,}"
    
    del mask, dense_indices, hilbert_distances, sort_order, w_real, w_imag, abs_data
    gc.collect()
    
    return result

# =============================================================================
# 4. ORCHESTRATOR
# =============================================================================

def run_geometry_forge(input_path: str, output_path: str, args):
    print(f"Igniting Geometry Forge: Loading {input_path}...")
    tensor_iterator = None

    if input_path.lower().endswith('.gguf'):
        try:
            reader = gguf.GGUFReader(input_path)
            tensor_iterator = reader.tensors
        except Exception as e:
            raise RuntimeError(f"Failed to parse GGUF file: {e}")
            
    elif input_path.lower().endswith('.safetensors'):
        try:
            from safetensors import safe_open
            print("Safetensors format detected. Forging with zero-copy mmap...")
            
            class SafetensorsAdapter:
                def __init__(self, name, data):
                    self.name = name
                    self.data = data

            def st_tensor_generator():
                with safe_open(input_path, framework="np", device="cpu") as f:
                    for key in f.keys():
                        yield SafetensorsAdapter(key, f.get_tensor(key))
            
            tensor_iterator = st_tensor_generator()
        except ImportError:
            raise ImportError("Please install safetensors to use this format: pip install safetensors")
        except Exception as e:
            raise RuntimeError(f"Failed to parse Safetensors file: {e}")
            
    elif input_path.lower().endswith(('.pt', '.pth', '.bin')):
        try:
            import torch
            print("PyTorch format detected. Attempting to load state dictionary with mmap...")
            
            try:
                state_dict = torch.load(input_path, map_location="cpu", mmap=True, weights_only=True)
            except RuntimeError as e:
                if "mmap can only be used" in str(e) or "_use_new_zipfile_serialization" in str(e):
                    print("Legacy PyTorch format detected. Falling back to standard RAM load (mmap=False)...")
                    state_dict = torch.load(input_path, map_location="cpu", mmap=False, weights_only=True)
                else:
                    raise e
            
            class PyTorchTensorAdapter:
                def __init__(self, name, data):
                    self.name = name
                    self.data = data

            def pt_tensor_generator():
                for name, tensor in state_dict.items():
                    if tensor.dtype == torch.bfloat16 or tensor.dtype == torch.float16:
                        yield PyTorchTensorAdapter(name, tensor.float().numpy())
                    else:
                        yield PyTorchTensorAdapter(name, tensor.numpy())
                    
            tensor_iterator = pt_tensor_generator()
            
        except ImportError:
            raise ImportError("PyTorch is required to load .pt files. Please run: pip install torch")
        except Exception as e:
            raise RuntimeError(f"Failed to parse PyTorch file: {e}")
    else:
        raise ValueError(f"Unsupported file format: {input_path}. Please provide a .gguf, .safetensors, or .pt/.bin file.")

    total_orig = 0
    total_surv = 0
    
    with zipfile.ZipFile(output_path, 'w', compression=zipfile.ZIP_STORED) as holo_zip:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = []
            
            for tensor in tensor_iterator:
                name = tensor.name
                bypass_keywords = ["embd", "embed", "wte", "output", "lm_head"]
                is_bypassed = len(tensor.data.shape) < 2 or any(k in name.lower() for k in bypass_keywords)
                
                data = tensor.data
                if data.dtype == np.uint8 or data.dtype == np.uint16:
                    data = data.view(np.float16).astype(np.float32)
                else:
                    data = data.astype(np.float32)
                
                # --- PRE-FORGE TRANSPOSE FIX ---
                # GPT-2 uses Conv1D. We mathematically transpose the dense matrix here 
                # so the Hilbert mapping naturally aligns with SpMV execution expectations.
                is_conv1d_layer = "c_attn" in name or "c_proj" in name or "c_fc" in name
                if is_conv1d_layer and not is_bypassed and len(data.shape) == 2:
                    data = data.T 
                
                future = executor.submit(forge_layer_worker, name, data, is_bypassed, args.std_factor, args.max_sparsity, args.zstd_level, args.phase_mode)
                futures.append(future)
                
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                for fname, b_data in result["files"].items():
                    holo_zip.writestr(fname, b_data)
                
                print(result["log"])
                total_orig += result["orig_params"]
                total_surv += result["surv_params"]

    sparsity = 100.0 * (1.0 - (total_surv / total_orig)) if total_orig > 0 else 0
    print("\n" + "="*50)
    print("GEOMETRY FORGE COMPLETE")
    print("="*50)
    print(f"Phase Geometry   : {args.phase_mode.upper()}")
    print(f"Total Parameters : {total_orig:,}")
    print(f"Qubed Survivors  : {total_surv:,} (Sparsity: {sparsity:.2f}%)")
    print(f"Output File      : {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str, help="Path to dense model (.gguf, .safetensors, .pt, .bin)")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--phase_mode", type=str, choices=["flatland", "quantum"], default="flatland", help="Phase injection strategy")
    parser.add_argument("--std_factor", type=float, default=0.5)
    parser.add_argument("--max_sparsity", type=float, default=60.0)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--zstd_level", type=int, default=3)
    
    args = parser.parse_args()
    out_path = args.output if args.output else args.input_path.rsplit('.', 1)[0] + '.holo'
    
    run_geometry_forge(args.input_path, out_path, args)
