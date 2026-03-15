"""
ThereminQ Holoqubed - Offline GGUF Ingress & Conversion Pipeline
Converts standard dense .gguf models (FP32/FP16/BF16) into the sparse, 
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


MORTON_DIM = 24


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
        "files": {}, # Dictionary of {filename: compressed_bytes}
        "orig_params": data.size,
        "surv_params": 0,
        "log": ""
    }
    
    compressor = zstandard.ZstdCompressor(level=zstd_level)
    
    if is_bypassed:
        dense_data = data.astype(np.float16)
        result["surv_params"] = dense_data.size
        
        buf = io.BytesIO()
        np.save(buf, dense_data, allow_pickle=False)
        result["files"][f"{name}.npy.zst"] = compressor.compress(buf.getvalue())
        result["log"] = f"  [BYPASSED] {name} | Saved as Zstd Dense (Sparsity: 0.00%)"
        
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
    result["files"][f"{name}.coords.npy.zst"] = compressor.compress(coord_buf.getvalue())
    
    # Pack Weights into memory buffer & Compress
    weight_buf = io.BytesIO()
    np.save(weight_buf, surviving_weights, allow_pickle=False)
    result["files"][f"{name}.weights.npy.zst"] = compressor.compress(weight_buf.getvalue())
    
    sparsity = 100.0 * (1.0 - (surviving_count / result["orig_params"]))
    result["log"] = f"  [FORGED] {name} | Cutoff: {dynamic_threshold:.4f} | Sparsity: {sparsity:.2f}% | Survivors: {surviving_count:,}"
    
    # Aggressively wipe worker RAM
    del mask, dense_indices, spatial_coords, surviving_weights, sort_order
    del coord_buf, weight_buf
    gc.collect()
    
    return result


def forge_holo_dictionary(gguf_path: str, output_holo_path: str, std_factor: float = 0.5, max_workers: int = 4, zstd_level: int = 3) -> dict:
    print(f"Igniting the Forge: Loading {gguf_path}...")
    
    if not os.path.exists(gguf_path):
        raise FileNotFoundError(f"Could not find GGUF file at {gguf_path}")
    
    try:
        reader = gguf.GGUFReader(gguf_path)
    except Exception as e:
        raise RuntimeError(f"Failed to parse GGUF file: {e}")

    total_original_params = 0
    total_surviving_params = 0

    print(f"Applying Dynamic Holoqubed Collapse (StdDev: {std_factor} | Workers: {max_workers} | Zstd Level: {zstd_level})...")
    print(f"Streaming directly to disk: {output_holo_path}")
    
    # Open the file as an uncompressed ZIP stream (Compression is handled by workers)
    with zipfile.ZipFile(output_holo_path, 'w', compression=zipfile.ZIP_STORED) as holo_zip:
        
        # Fire up the parallel worker pool
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            active_futures = set()
            
            for tensor in reader.tensors:
                name = tensor.name
                
                # Check if layer is 1D or a massive vocabulary/head layer to bypass
                is_bypassed = len(tensor.data.shape) < 2 or "embd" in name or "output" in name
                
                # Main thread decodes the raw bytes
                data = decode_bf16_to_fp32(tensor.data)
                
                # Toss the decoded tensor to a background worker
                future = executor.submit(forge_layer_worker, name, data, is_bypassed, std_factor, zstd_level)
                active_futures.add(future)
                
                # ROLLING WINDOW: explicitly protects RAM from overflowing with queued tensors
                while len(active_futures) >= max_workers:
                    done, _ = concurrent.futures.wait(active_futures, return_when=concurrent.futures.FIRST_COMPLETED)
                    
                    for completed_future in done:
                        result = completed_future.result()
                        
                        # Main thread handles writing to prevent file corruption
                        for fname, b_data in result["files"].items():
                            holo_zip.writestr(fname, b_data)
                            
                        print(result["log"])
                        total_original_params += result["orig_params"]
                        total_surviving_params += result["surv_params"]
                        
                        active_futures.remove(completed_future)
                    
                    gc.collect() # Flush main thread RAM
            
            # Drain any remaining tasks in the queue after the loop finishes
            for completed_future in concurrent.futures.as_completed(active_futures):
                result = completed_future.result()
                for fname, b_data in result["files"].items():
                    holo_zip.writestr(fname, b_data)
                    
                print(result["log"])
                total_original_params += result["orig_params"]
                total_surviving_params += result["surv_params"]

    print("\nForge Complete. All layers streamed to disk successfully.")
    
    total_sparsity = 100.0 * (1.0 - (total_surviving_params / total_original_params))
    print(f"Total Model Sparsity Achieved: {total_sparsity:.2f}%")
    print(f"Original Params: {total_original_params:,} -> Holographic Pathways: {total_surviving_params:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ThereminQ Holoqubed - Dense to Sparse Converter")
    parser.add_argument("gguf_path", type=str, help="Path to input .gguf file (Must be unquantized, e.g., BF16/FP16)")
    parser.add_argument("--output", type=str, default=None, help="Optional output path for .holo file")
    parser.add_argument("--std_factor", type=float, default=0.5, help="Standard Deviation multiplier for pruning (default: 0.5)")
    parser.add_argument("--workers", type=int, default=4, help="Number of CPU cores to use (Warning: Higher = More RAM used!)")
    parser.add_argument("--zstd_level", type=int, default=3, help="Zstandard compression level (1-22). Default: 3")
    
    args = parser.parse_args()
    
    out_path = args.output if args.output else args.gguf_path.rsplit('.', 1)[0] + '.holo'
    
    forge_holo_dictionary(
        args.gguf_path, 
        out_path, 
        std_factor=args.std_factor, 
        max_workers=args.workers,
        zstd_level=args.zstd_level
    )

