"""
ThereminQ Holoqubed - Offline Model Ingress & Conversion Pipeline
Converts dense .gguf and .pt models (FP32/FP16/BF16) into the sparse, 
spatially encoded .holo dictionary. 
*PARALLEL STREAMING VERSION WITH BF16 BIT-SHIFTING, OOM PROTECTION, & ZSTD COMPRESSION*
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

MORTON_DIM = 32

def decode_bf16_to_fp32(tensor_data: np.ndarray) -> np.ndarray:
    """
    NumPy lacks native BF16 support and often loads them as raw integers/bytes.
    This function safely converts the data into a valid FP32 decimal array 
    so the standard deviation and pruning math are accurate.
    """
    if tensor_data.dtype == np.uint8:
        tensor_data = tensor_data.view(np.uint16)
        
    if tensor_data.dtype == np.uint16:
        # Shift the 16 bits of BF16 to the top of a 32-bit container to restore FP32
        return np.left_shift(tensor_data.astype(np.uint32), 16).view(np.float32)
        
    return tensor_data.astype(np.float32)

def encode_morton_vectorized(dense_indices: np.ndarray, chunk_size: int = 5_000_000) -> np.ndarray:
    """
    Batched Morton (Z-order) bit-interleaving. Processes coordinates in chunks 
    to prevent catastrophic RAM explosions on massive layers.
    """
    num_dims = dense_indices.shape[1]
    num_elements = dense_indices.shape[0]
    
    spatial_coords = np.zeros(num_elements, dtype=np.uint64)
    
    for start in range(0, num_elements, chunk_size):
        end = min(start + chunk_size, num_elements)
        chunk = dense_indices[start:end].astype(np.uint64)
        
        chunk_coords = np.zeros(chunk.shape[0], dtype=np.uint64)
        
        for dim in range(num_dims):
            val = chunk[:, dim]
            spread_val = np.zeros_like(val)
            
            for bit in range(16):
                spread_val |= ((val >> bit) & 1) << (bit * num_dims)
                
            chunk_coords |= (spread_val << dim)
            
        spatial_coords[start:end] = chunk_coords % (1 << MORTON_DIM)
        
    return spatial_coords

def forge_layer_worker(name: str, data: np.ndarray, is_bypassed: bool, std_factor: float, zstd_level: int) -> dict:
    """
    Isolated background process. Does the heavy math, compresses the raw bytes 
    using Zstandard, and returns them for O(1) random-access ZIP storage.
    """
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
        dense_data = data.astype(np.float16)
        result["surv_params"] = dense_data.size
        
        buf = io.BytesIO()
        np.save(buf, dense_data, allow_pickle=False)
        
        raw_bytes = buf.getvalue()
        compressed_data = compressor.compress(raw_bytes)
        
        result["uncompressed_bytes"] = len(raw_bytes)
        result["compressed_bytes"] = len(compressed_data)
        result["files"][f"{name}.npy.zst"] = compressed_data
        
        ratio = result["uncompressed_bytes"] / result["compressed_bytes"] if result["compressed_bytes"] > 0 else 1.0
        result["log"] = f"  [BYPASSED] {name} | Saved as Zstd Dense (Sparsity: 0.00%) | Zstd Ratio: {ratio:.2f}x"
        
        del dense_data, buf
        return result

    # --- THE DYNAMIC SCALPEL ---
    layer_std = np.std(data)
    dynamic_threshold = std_factor * layer_std
    mask = np.abs(data) > dynamic_threshold
    
    surviving_weights = data[mask].astype(np.float16)
    surviving_count = surviving_weights.size
    result["surv_params"] = surviving_count
    
    if surviving_count == 0:
        result["log"] = f"  [DELETED] {name} (100% Sparsity)"
        del mask, surviving_weights
        return result

    # Extract indices and encode
    dense_indices = np.argwhere(mask)
    spatial_coords = encode_morton_vectorized(dense_indices)
    
    # Sort for O(log N) Query Planner
    sort_order = np.argsort(spatial_coords)
    spatial_coords = spatial_coords[sort_order]
    surviving_weights = surviving_weights[sort_order]
    
    # Pack Coords into memory buffer & Compress
    coord_buf = io.BytesIO()
    np.save(coord_buf, spatial_coords, allow_pickle=False)
    raw_coords = coord_buf.getvalue()
    c_coords = compressor.compress(raw_coords)
    
    # Pack Weights into memory buffer & Compress
    weight_buf = io.BytesIO()
    np.save(weight_buf, surviving_weights, allow_pickle=False)
    raw_weights = weight_buf.getvalue()
    c_weights = compressor.compress(raw_weights)
    
    # Track byte sizes for compression ratio math
    result["uncompressed_bytes"] = len(raw_coords) + len(raw_weights)
    result["compressed_bytes"] = len(c_coords) + len(c_weights)
    
    result["files"][f"{name}.coords.npy.zst"] = c_coords
    result["files"][f"{name}.weights.npy.zst"] = c_weights
    
    sparsity = 100.0 * (1.0 - (surviving_count / result["orig_params"]))
    ratio = result["uncompressed_bytes"] / result["compressed_bytes"] if result["compressed_bytes"] > 0 else 1.0
    
    result["log"] = f"  [FORGED] {name} | Cutoff: {dynamic_threshold:.4f} | Sparsity: {sparsity:.2f}% | Survivors: {surviving_count:,} | Zstd Ratio: {ratio:.2f}x"
    
    # Aggressively wipe worker RAM
    del mask, dense_indices, spatial_coords, surviving_weights, sort_order
    del coord_buf, weight_buf
    gc.collect()
    
    return result

def forge_holo_dictionary(input_path: str, output_holo_path: str, std_factor: float = 0.5, max_workers: int = 12, zstd_level: int = 3) -> dict:
    print(f"Igniting the Forge: Loading {input_path}...")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Could not find file at {input_path}")
    
    input_file_size = os.path.getsize(input_path)
    tensor_iterator = None

    # --- DUAL LOADER LOGIC ---
    if input_path.lower().endswith('.gguf'):
        try:
            reader = gguf.GGUFReader(input_path)
            tensor_iterator = reader.tensors
        except Exception as e:
            raise RuntimeError(f"Failed to parse GGUF file: {e}")
            
    elif input_path.lower().endswith(('.pt', '.pth', '.bin')):
        try:
            import torch
            print("PyTorch format detected. Loading state dictionary with mmap...")
            state_dict = torch.load(input_path, map_location="cpu", mmap=True, weights_only=True)
            
            # Simple mock class to mimic GGUF tensor attributes
            class PyTorchTensorAdapter:
                def __init__(self, name, data):
                    self.name = name
                    self.data = data

            # Generator yields tensors one-by-one to preserve OOM protection
            def pt_tensor_generator():
                for name, tensor in state_dict.items():
                    # PyTorch natively handles BF16 to FP32 casting here
                    yield PyTorchTensorAdapter(name, tensor.float().numpy())
                    
            tensor_iterator = pt_tensor_generator()
            
        except ImportError:
            raise ImportError("PyTorch is required to load .pt files. Please run: pip install torch")
        except Exception as e:
            raise RuntimeError(f"Failed to parse PyTorch file: {e}")
    else:
        raise ValueError(f"Unsupported file format: {input_path}. Please provide a .gguf or .pt file.")

    total_original_params = 0
    total_surviving_params = 0
    total_uncompressed_bytes = 0
    total_compressed_bytes = 0

    print(f"Applying Dynamic Holoqubed Collapse (StdDev: {std_factor} | Workers: {max_workers} | Zstd Level: {zstd_level})...")
    print(f"Streaming directly to disk: {output_holo_path}")
    
    with zipfile.ZipFile(output_holo_path, 'w', compression=zipfile.ZIP_STORED) as holo_zip:
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            active_futures = set()
            
            for tensor in tensor_iterator:
                name = tensor.name
                
                # Broadened bypass keywords for both GGUF and PyTorch standard architectures
                bypass_keywords = ["embd", "embed", "wte", "output", "lm_head"]
                is_bypassed = len(tensor.data.shape) < 2 or any(k in name.lower() for k in bypass_keywords)
                
                # Pass through our normalizer (handles raw GGUF bf16 bytes, does nothing to PyTorch fp32)
                data = decode_bf16_to_fp32(tensor.data)
                
                future = executor.submit(forge_layer_worker, name, data, is_bypassed, std_factor, zstd_level)
                active_futures.add(future)
                
                while len(active_futures) >= max_workers:
                    done, _ = concurrent.futures.wait(active_futures, return_when=concurrent.futures.FIRST_COMPLETED)
                    
                    for completed_future in done:
                        result = completed_future.result()
                        
                        for fname, b_data in result["files"].items():
                            holo_zip.writestr(fname, b_data)
                            
                        print(result["log"])
                        total_original_params += result["orig_params"]
                        total_surviving_params += result["surv_params"]
                        total_uncompressed_bytes += result["uncompressed_bytes"]
                        total_compressed_bytes += result["compressed_bytes"]
                        
                        active_futures.remove(completed_future)
                    
                    gc.collect()
            
            # Drain any remaining tasks
            for completed_future in concurrent.futures.as_completed(active_futures):
                result = completed_future.result()
                for fname, b_data in result["files"].items():
                    holo_zip.writestr(fname, b_data)
                    
                print(result["log"])
                total_original_params += result["orig_params"]
                total_surviving_params += result["surv_params"]
                total_uncompressed_bytes += result["uncompressed_bytes"]
                total_compressed_bytes += result["compressed_bytes"]

    print("\n" + "="*50)
    print("FORGE COMPLETE: DICTIONARY STATISTICS")
    print("="*50)
    
    total_sparsity = 100.0 * (1.0 - (total_surviving_params / total_original_params)) if total_original_params > 0 else 0
    final_zstd_ratio = total_uncompressed_bytes / total_compressed_bytes if total_compressed_bytes > 0 else 1.0
    final_disk_reduction = input_file_size / total_compressed_bytes if total_compressed_bytes > 0 else 1.0
    
    print(f"Original Parameters    : {total_original_params:,}")
    print(f"Holographic Survivors  : {total_surviving_params:,}")
    print(f"Total Sparsity Cut     : {total_sparsity:.2f}%")
    print("-" * 50)
    print(f"Input Archive Size     : {input_file_size / (1024**2):.2f} MB")
    print(f"Uncompressed Payload   : {total_uncompressed_bytes / (1024**2):.2f} MB")
    print(f"Final Zstd Archive     : {total_compressed_bytes / (1024**2):.2f} MB")
    print("-" * 50)
    print(f"Total Zstd Compression : {final_zstd_ratio:.2f}x")
    print(f"Total Storage Reduction: {final_disk_reduction:.2f}x")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ThereminQ Holoqubed - Dense to Sparse Converter")
    parser.add_argument("input_path", type=str, help="Path to input .gguf or .pt/.pth file (Must be unquantized, e.g., BF16/FP16)")
    parser.add_argument("--output", type=str, default=None, help="Optional output path for .holo file")
    parser.add_argument("--std_factor", type=float, default=0.5, help="Standard Deviation multiplier for pruning (default: 0.5)")
    parser.add_argument("--workers", type=int, default=12, help="Number of CPU cores to use (Warning: 12 workers requires massive RAM!)")
    parser.add_argument("--zstd_level", type=int, default=3, help="Zstandard compression level (1-22). Default: 3")
    
    args = parser.parse_args()
    
    out_path = args.output if args.output else args.input_path.rsplit('.', 1)[0] + '.holo'
    
    forge_holo_dictionary(
        args.input_path, 
        out_path, 
        std_factor=args.std_factor, 
        max_workers=args.workers,
        zstd_level=args.zstd_level
    )
