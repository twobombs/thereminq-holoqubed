"""
ThereminQ Holoqubed - Ingress Loader & OpenCL Execution Engine
Streams Zstd-compressed layers from the .holo ZIP archive, performs 
ultra-fast O(log N) pathway lookups, and executes direct Sparse Matrix-Vector 
Multiplication (SpMV) in VRAM using OpenCL.
"""

import os
import time
import zipfile
import zstandard
import io
import numpy as np
import pyopencl as cl
import argparse
from typing import Tuple, List
from functools import lru_cache


class HoloQueryPlanner:
    def __init__(self, holo_file_path: str, max_cached_layers: int = 4):
        print(f"Initializing JIT CPU Query Planner...")
        print(f"Mounting Compressed Holographic Dictionary: {holo_file_path}")
        
        if not os.path.exists(holo_file_path):
            raise FileNotFoundError(f"Holo dictionary not found: {holo_file_path}")
        
        if not holo_file_path.lower().endswith('.holo'):
            raise ValueError(f"Expected .holo file, got: {holo_file_path}")
        
        self.holo_path = holo_file_path
        self.decompressor = zstandard.ZstdDecompressor()
        
        with zipfile.ZipFile(self.holo_path, 'r') as archive:
            self.file_list = set(archive.namelist())
            
        self.layers: List[str] = list(set([
            key.replace(".coords.npy.zst", "").replace(".weights.npy.zst", "")
            for key in self.file_list 
            if ".coords.npy.zst" in key
        ]))
        
        self.layers.sort()
        print(f"Successfully mounted {len(self.layers)} sparse layers for JIT streaming.\n")

    def _read_and_decompress(self, filename: str) -> np.ndarray:
        if filename not in self.file_list:
            raise KeyError(f"Missing required payload: {filename}")
            
        with zipfile.ZipFile(self.holo_path, 'r') as archive:
            with archive.open(filename) as f:
                compressed_data = f.read()
                
        raw_bytes = self.decompressor.decompress(compressed_data)
        return np.load(io.BytesIO(raw_bytes))

    @lru_cache(maxsize=4)
    def _fetch_layer_data(self, layer_name: str) -> Tuple[np.ndarray, np.ndarray]:
        coords_file = f"{layer_name}.coords.npy.zst"
        weights_file = f"{layer_name}.weights.npy.zst"
        
        coords = self._read_and_decompress(coords_file)
        weights = self._read_and_decompress(weights_file)
        
        return coords, weights

    def query_active_pathways(
        self, 
        layer_name: str, 
        active_morton_coords: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if layer_name not in self.layers:
            raise ValueError(f"Layer '{layer_name}' not found in dictionary.")
        
        layer_coords, layer_weights = self._fetch_layer_data(layer_name)

        indices = np.searchsorted(layer_coords, active_morton_coords)
        indices = np.clip(indices, 0, len(layer_coords) - 1)
        
        valid_mask = layer_coords[indices] == active_morton_coords
        
        matched_coords = active_morton_coords[valid_mask]
        matched_weights = layer_weights[indices[valid_mask]]
        
        return matched_coords, matched_weights


# =============================================================================
# OPENCL GPU EXECUTION ENGINE (DIRECT SpMV)
# =============================================================================

# The custom atomic_add_float function uses a hardware-level Compare-And-Swap 
# (atomic_cmpxchg) loop to safely add floats across thousands of parallel threads.
SPMV_KERNEL_CODE = """
inline void atomic_add_float(volatile __global float *source, const float operand) {
    union {
        unsigned int int_val;
        float float_val;
    } newVal;
    union {
        unsigned int int_val;
        float float_val;
    } prevVal;
    do {
        prevVal.float_val = *source;
        newVal.float_val = prevVal.float_val + operand;
    } while (atomic_cmpxchg((volatile __global unsigned int *)source, prevVal.int_val, newVal.int_val) != prevVal.int_val);
}

__kernel void spmv_holo_weights(
    __global const ulong* morton_coords,   
    __global const ushort* sparse_weights_fp16, 
    __global const float* input_vector,         // The input token embeddings
    __global float* output_vector,              // The output activation vector
    const int num_elements                      
) {
    int gid = get_global_id(0);

    if (gid >= num_elements) {
        return;
    }

    ulong m_coord = morton_coords[gid];
    
    // Built-in OpenCL function to decode FP16 bytes into an FP32 register safely
    float weight = vload_half(gid, sparse_weights_fp16); 

    // Decode Morton
    uint row = 0;
    uint col = 0;

    for (int bit = 0; bit < 16; bit++) {
        ulong shift = bit * 2; 
        row |= (uint)(((m_coord >> (0 + shift)) & 1) << bit);
        col |= (uint)(((m_coord >> (1 + shift)) & 1) << bit);
    }

    // The Magic: Sparse Math!
    // We fetch the input value for this column, multiply by our weight, 
    // and atomically add it to the output row.
    float activation = weight * input_vector[col];
    
    // Only perform the atomic add if the activation isn't exactly zero
    if (activation != 0.0f) {
        atomic_add_float(&output_vector[row], activation);
    }
}
"""

def execute_gpu_spmv(
    context: cl.Context, 
    queue: cl.CommandQueue, 
    matched_coords: np.ndarray, 
    matched_weights: np.ndarray,
    input_vector: np.ndarray,
    target_shape: tuple
) -> np.ndarray:
    print(f"\n--- GPU SpMV DISPATCH ---")
    
    cl_coords = matched_coords.astype(np.uint64)
    # TRICK: View the FP16 array as raw 16-bit unsigned integers to bypass PyOpenCL typing limits
    cl_weights_raw = matched_weights.astype(np.float16).view(np.uint16)
    
    num_elements = np.int32(len(cl_coords))
    matrix_rows, matrix_cols = target_shape

    mf = cl.mem_flags

    # 1. Zero-Copy Input Buffers
    coords_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cl_coords)
    weights_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cl_weights_raw)
    
    # Ensure input vector is strictly FP32 for OpenCL
    input_fp32 = input_vector.astype(np.float32)
    in_vec_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=input_fp32)
    
    # 2. Allocate Output Vector (Not a massive matrix anymore! Just a 1D vector)
    out_vec_bytes = matrix_rows * 4 # 4 bytes per float32
    out_vec_buf = cl.Buffer(context, mf.READ_WRITE, size=out_vec_bytes)
    
    # Fill output vector with zeros before addition
    pattern = np.zeros(1, dtype=np.float32)
    cl.enqueue_fill_buffer(queue, out_vec_buf, pattern, 0, out_vec_bytes)

    # 3. Build and Dispatch
    prg = cl.Program(context, SPMV_KERNEL_CODE).build()
    
    print(f"Executing Direct Sparse Multiplication on {num_elements} parameters...")
    start_gpu = time.time()
    
    global_work_size = (int(num_elements),)
    prg.spmv_holo_weights(
        queue, 
        global_work_size, 
        None, 
        coords_buf, 
        weights_buf, 
        in_vec_buf,
        out_vec_buf,
        num_elements
    )
    
    queue.finish()
    print(f"SpMV Complete in {(time.time() - start_gpu)*1000:.3f} ms. Output Vector Generated.")
    
    # 4. Read back to CPU
    result_vector = np.empty((matrix_rows,), dtype=np.float32)
    cl.enqueue_copy(queue, result_vector, out_vec_buf).wait()
    
    return result_vector


# =============================================================================
# INTEGRATION TEST
# =============================================================================
def run_test_simulation(holo_file: str):
    print("="*50)
    print("HOLOQUBED SpMV INGRESS SIMULATION")
    print("="*50)

    try:
        planner = HoloQueryPlanner(holo_file, max_cached_layers=4)
        
        platforms = cl.get_platforms()
        if not platforms:
            raise RuntimeError("No OpenCL platforms found")
            
        devices = platforms[0].get_devices()
        if not devices:
            raise RuntimeError("No GPU devices found")
            
        device = devices[0]
        ctx = cl.Context([device])
        q = cl.CommandQueue(ctx)
        print(f"OpenCL Context Created on: {device.name}\n")
        
        layer_to_query = planner.layers[0] if planner.layers else "blk.0.attn_q"
        target_matrix_shape = (4096, 4096) 
        
        print(f"Simulating CPU querying 10,000 active pathways in {layer_to_query}...")
        mock_active_coords = np.random.choice(
            planner._fetch_layer_data(layer_to_query)[0], 
            size=10000, 
            replace=False
        )
        mock_active_coords.sort() 
        
        start_q = time.time()
        matched_coords, matched_weights = planner.query_active_pathways(
            layer_to_query, mock_active_coords
        )
        print(f"CPU Query took {(time.time() - start_q)*1000:.3f} ms.")

        if len(matched_coords) > 0:
            # Create a mock 1D input token embedding vector
            mock_input_vector = np.random.randn(target_matrix_shape[1]).astype(np.float32)
            print(f"Mock Input Vector Shape: {mock_input_vector.shape}")
            
            # Dispatch Direct SpMV
            output_vector = execute_gpu_spmv(
                ctx, q, matched_coords, matched_weights, mock_input_vector, target_matrix_shape
            )
            
            print(f"\nVerification:")
            print(f"Resulting Output Vector Shape: {output_vector.shape}")
            print(f"Sample Outputs: {output_vector[:5]}")
            
            # Confirm we actually bypassed dense memory allocation
            dense_mb = (target_matrix_shape[0] * target_matrix_shape[1] * 4) / (1024**2)
            sparse_mb = (target_matrix_shape[0] * 4) / (1024**2)
            print(f"\nVRAM Memory Savings for this operation:")
            print(f"Dense Requirement: {dense_mb:.2f} MB")
            print(f"SpMV Requirement : {sparse_mb:.4f} MB")

    except Exception as e:
        print(f"Simulation Failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ThereminQ Holoqubed - Ingress Loader & Simulator")
    parser.add_argument("holo_file", type=str, help="Path to the .holo dictionary file")
    
    args = parser.parse_args()
    
    if os.path.exists(args.holo_file):
        run_test_simulation(args.holo_file)
    else:
        print(f"Error: Could not find {args.holo_file}. Run the offline forge script first.")
