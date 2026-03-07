"""
ThereminQ Holoqubed - Dictionary Loader & CPU Query Planner
Memory-maps the massive .holo dictionary and performs ultra-fast O(log N) / O(1) 
pathway lookups before shuttling data to the OpenCL GPUs.
"""

import numpy as np
import pyopencl as cl
import time
from typing import Tuple, List, Optional


class HoloQueryPlanner:
    def __init__(self, holo_file_path: str):
        """
        Initialize the CPU Query Planner with a memory-mapped .holo dictionary.
        
        Args:
            holo_file_path: Path to the .holo dictionary file
        """
        print(f"Initializing CPU Query Planner...")
        print(f"Memory-Mapping Holographic Dictionary: {holo_file_path}")
        
        # Validate file exists
        if not os.path.exists(holo_file_path):
            raise FileNotFoundError(f"Holo dictionary not found: {holo_file_path}")
        
        if not holo_file_path.lower().endswith('.holo'):
            raise ValueError(f"Expected .holo file, got: {holo_file_path}")
        
        # mmap_mode='r' is the magic. It tells Linux NOT to load the massive file 
        # into active RAM, but to map the disk addresses directly to virtual memory.
        try:
            self.dictionary = np.load(holo_file_path, mmap_mode='r')
        except Exception as e:
            raise RuntimeError(f"Failed to memory-map .holo file: {e}")
        
        # Cache layer names for quick reference
        self.layers: List[str] = [
            key.replace(".coords", "") 
            for key in self.dictionary.files 
            if ".coords" in key
        ]
        print(f"Successfully mapped {len(self.layers)} sparse layers into virtual memory.\n")

    def query_active_pathways(
        self, 
        layer_name: str, 
        active_hilbert_coords: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        The O(log N) CPU Search: 
        Finds the exact FP16 weights that correspond to the active spatial coordinates.
        
        Args:
            layer_name: Name of the layer to query (e.g., "blk.0.attn_q")
            active_hilbert_coords: Array of 1D spatial coordinates to look up
            
        Returns:
            Tuple of (matched_coords, matched_weights) for valid lookups
        """
        # Validate layer exists
        if layer_name not in self.layers:
            raise ValueError(f"Layer '{layer_name}' not found in dictionary. Available: {self.layers}")
        
        # 1. Access the mapped arrays (Instantaneous, no data copied yet)
        layer_coords = self.dictionary[f"{layer_name}.coords"]
        layer_weights = self.dictionary[f"{layer_name}.weights"]

        # 2. Find intersecting pathways using binary search
        indices = np.searchsorted(layer_coords, active_hilbert_coords)
        
        # Handle out-of-bounds safety
        indices = np.clip(indices, 0, len(layer_coords) - 1)
        
        # Check which coordinates actually exist in the sparse dictionary
        valid_mask = layer_coords[indices] == active_hilbert_coords
        
        # 3. Extract ONLY the surviving, matching data
        matched_coords = active_hilbert_coords[valid_mask]
        matched_weights = layer_weights[indices[valid_mask]]
        
        return matched_coords, matched_weights

    def get_layer_info(self) -> dict:
        """
        Returns metadata about all loaded layers.
        
        Returns:
            Dictionary with layer names and their pathway counts
        """
        info = {}
        for layer in self.layers:
            coords = self.dictionary[f"{layer}.coords"]
            weights = self.dictionary[f"{layer}.weights"]
            info[layer] = {
                "pathway_count": len(weights),
                "coord_dtype": str(coords.dtype),
                "weight_dtype": str(weights.dtype)
            }
        return info


# =============================================================================
# INTEGRATION TEST
# =============================================================================
def simulate_inference_step(
    planner: HoloQueryPlanner, 
    context: cl.Context, 
    queue: cl.CommandQueue
) -> None:
    """
    Simulates a single inference step with the Query Planner.
    
    Args:
        planner: HoloQueryPlanner instance
        context: OpenCL context
        queue: OpenCL command queue
    """
    layer_to_query = planner.layers[0] if planner.layers else "blk.0.attn_q"
    
    # Simulate the CPU generating 3 active spatial coordinates
    mock_active_coords = np.array([4096, 128555, 8000000], dtype=np.int64)
    
    print(f"--- INFERENCE STEP ---")
    print(f"1. CPU querying layer '{layer_to_query}' for coordinates: {mock_active_coords}")
    
    start_q = time.time()
    
    # The CPU instantly extracts only the intersecting pathways
    try:
        matched_coords, matched_weights = planner.query_active_pathways(
            layer_to_query, 
            mock_active_coords
        )
    except ValueError as e:
        print(f"   -> Query failed: {e}")
        return
    
    print(f"   -> Query took {(time.time() - start_q)*1000:.3f} ms. Found {len(matched_coords)} valid pathways.")
    
    if len(matched_coords) == 0:
        print("   -> No resonant pathways found. Skipping GPU dispatch.")
        return

    print(f"2. Preparing Zero-Copy PCIe Shuttle...")
    
    # Create the Zero-Copy payload for the PyOpenCL Loom
    mf = cl.mem_flags
    
    # We cast coordinates to int32 because OpenCL GPUs prefer 32-bit math for indexing
    coords_32 = matched_coords.astype(np.int32) 
    
    cl_coords_buf = cl.Buffer(
        context, 
        mf.READ_ONLY | mf.ALLOC_HOST_PTR | mf.COPY_HOST_PTR, 
        hostbuf=coords_32
    )
    cl_weights_buf = cl.Buffer(
        context, 
        mf.READ_ONLY | mf.ALLOC_HOST_PTR | mf.COPY_HOST_PTR, 
        hostbuf=matched_weights
    )
    
    print(f"3. Data pinned. Ready for GPU Scatter-Gather dispatch!")


if __name__ == "__main__":
    import os
    
    # Note: To run this test, you must have first generated a .holo file using gguf_to_holo.py
    test_holo_file = "models/llama-3-8b.holo"
    
    if os.path.exists(test_holo_file):
        try:
            planner = HoloQueryPlanner(test_holo_file)
            
            # Mock OpenCL initialization for the test
            platforms = cl.get_platforms()
            if not platforms:
                raise RuntimeError("No OpenCL platforms found")
            
            devices = platforms[0].get_devices()
            if not devices:
                raise RuntimeError("No GPU devices found")
                
            device = devices[0]
            ctx = cl.Context([device])
            q = cl.CommandQueue(ctx)
            
            simulate_inference_step(planner, ctx, q)
        except Exception as e:
            print(f"Error during test: {e}")
    else:
        print(f"Error: {test_holo_file} not found. Run the offline forge script first!")
