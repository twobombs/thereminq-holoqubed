"""
ThereminQ Holoqubed - PyOpenCL Prototype
Target: AMD Radeon Pro V340 (Vega 10) via Mesa Rusticl
"""

import os
import pyopencl as cl
import numpy as np
import time

# =============================================================================
# 1. HILBERT CURVE ENCODING (Translated from Holoqubed C++)
# =============================================================================
HILBERT_DIM = 24

def encode_boundary_index(row_data: np.ndarray) -> int:
    """
    Translates a float array of activation thresholds into a 1D spatial coordinate
    using bitwise XOR interleaving.
    """
    index = 0
    for i, val in enumerate(row_data):
        # Scale to 0-255 and apply bitwise XOR interleaving
        v = int(abs(val) * 255.0) % 255
        index ^= (v << (i * 2))
        
    return index % (1 << HILBERT_DIM)

# =============================================================================
# 2. OPENCL RUSTICL INITIALIZATION
# =============================================================================
def initialize_opencl():
    print("--- Holoqubed OpenCL Initialization ---")
    
    # Ensure the environment variables are set (usually handled by Docker)
    if os.environ.get('HSA_OVERRIDE_GFX_VERSION') != '9.0.0':
        print("WARNING: HSA_OVERRIDE_GFX_VERSION is not set to 9.0.0. Vega 10 may fail.")

    # Find the Rusticl platform
    platforms = cl.get_platforms()
    rusticl_platform = next((p for p in platforms if 'Rusticl' in p.name), None)
    
    if not rusticl_platform:
        raise RuntimeError("Rusticl OpenCL platform not found! Check your Mesa installation.")
    
    print(f"Platform: {rusticl_platform.name}")

    # Grab the first available GPU (One of your 8GB Vega 10 dies)
    devices = rusticl_platform.get_devices(device_type=cl.device_type.GPU)
    if not devices:
        raise RuntimeError("No GPU devices found on the Rusticl platform.")
    
    device = devices[0]
    print(f"Device: {device.name}")
    print(f"Max Compute Units: {device.max_compute_units}")
    print(f"Global Memory: {device.global_mem_size / (1024**3):.2f} GB\n")

    # Create Context and Command Queue
    context = cl.Context([device])
    queue = cl.CommandQueue(context)
    
    return context, queue

# =============================================================================
# 3. ZERO-COPY BUFFER & FP16 KERNEL EXECUTION
# =============================================================================
def run_sparse_kernel(context, queue):
    # The OpenCL C Kernel (Using FP16 Rapid Packed Math)
    # We enable the cl_khr_fp16 extension to allow native 'half' data types.
    kernel_code = """
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable

    __kernel void process_spatial_coordinates(
        __global half *dictionary_slice, 
        __global const int *active_indices,
        const int num_indices) 
    {
        int gid = get_global_id(0);
        
        if (gid < num_indices) {
            int target_index = active_indices[gid];
            
            // Perform dummy Rapid Packed Math on the specific coordinate
            // In the real engine, this will execute your holographic Top-K logic
            half current_val = dictionary_slice[target_index];
            dictionary_slice[target_index] = current_val * (half)2.0; 
        }
    }
    """
    
    print("Compiling FP16 OpenCL Kernel...")
    program = cl.Program(context, kernel_code).build()

    # --- SIMULATING THE 320GB SYSTEM RAM DICTIONARY ---
    # We create a NumPy array in host memory using float16.
    dictionary_size = 1_000_000 
    print(f"Allocating {dictionary_size} elements in System RAM...")
    host_dictionary = np.ones(dictionary_size, dtype=np.float16)
    
    # Generate some dummy spatial coordinates using your Hilbert function
    dummy_activations = np.array([0.5, -0.1, 0.9, 0.05])
    target_coord = encode_boundary_index(dummy_activations)
    
    # We only want to process a few sparse coordinates, not the whole dense matrix!
    active_indices = np.array([target_coord, target_coord + 1, target_coord + 2], dtype=np.int32)
    num_indices = np.int32(len(active_indices))

    # --- ZERO-COPY MEMORY ALLOCATION ---
    # ALLOC_HOST_PTR tells the driver to pin this memory in system RAM so the GPU 
    # can stream it directly over the PCIe bus via DMA.
    mf = cl.mem_flags
    dict_buffer = cl.Buffer(context, mf.READ_WRITE | mf.ALLOC_HOST_PTR | mf.COPY_HOST_PTR, hostbuf=host_dictionary)
    indices_buffer = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=active_indices)

    print(f"\nDispatching Sparse Kernel for coordinate: {target_coord}...")
    start_time = time.time()
    
    # Execute the kernel, launching only as many threads as we have active coordinates
    program.process_spatial_coordinates(queue, active_indices.shape, None, dict_buffer, indices_buffer, num_indices)
    
    # Read the modified Zero-Copy buffer back into the NumPy array
    cl.enqueue_copy(queue, host_dictionary, dict_buffer).wait()
    
    end_time = time.time()
    
    print(f"Execution Time: {(end_time - start_time) * 1000:.2f} ms")
    print(f"Modified Value at {target_coord}: {host_dictionary[target_coord]} (Expected: 2.0)")

if __name__ == "__main__":
    try:
        ctx, q = initialize_opencl()
        run_sparse_kernel(ctx, q)
        print("\nSUCCESS: Holoqubed execution completed without dropping off the PCIe bus!")
    except Exception as e:
        print(f"\nERROR: {e}")
