"""
ThereminQ Holoqubed - Offline GGUF Ingress & Conversion Pipeline
Converts standard dense .gguf models into the sparse, spatially encoded .holo dictionary.
"""

import numpy as np
import gguf
import os
from typing import Optional


HILBERT_DIM = 24


def encode_hilbert_vectorized(dense_indices: np.ndarray) -> np.ndarray:
    """
    A vectorized version of the Hilbert/Morton bit-interleaving function.
    Takes an N-dimensional array of dense matrix indices and encodes 
    them into a 1D spatial coordinate array for O(1) / O(log N) lookup.
    
    Uses OR-based Morton encoding for better performance while preserving
    approximate spatial locality.
    """
    # For a 2D weight matrix, dense_indices has shape (N, 2)
    # We apply the bitwise OR logic across the columns (Morton encoding)
    spatial_coords = np.zeros(dense_indices.shape[0], dtype=np.int64)
    
    for dim in range(dense_indices.shape[1]):
        # Shift and OR based on Morton encoding
        val = dense_indices[:, dim] % 255
        spatial_coords |= (val << (dim * 2))
        
    return spatial_coords % (1 << HILBERT_DIM)


def forge_holo_dictionary(
    gguf_path: str, 
    output_holo_path: str, 
    prune_threshold: float = 0.05
) -> dict:
    """
    Converts a dense GGUF model into the sparse Holoqubed format.
    
    Args:
        gguf_path: Path to input .gguf file
        output_holo_path: Path for output .holo file
        prune_threshold: Weight magnitude threshold for pruning
        
    Returns:
        Dictionary with conversion statistics
    """
    print(f"Igniting the Forge: Loading {gguf_path}...")
    
    # Validate input file exists
    if not os.path.exists(gguf_path):
        raise FileNotFoundError(f"Could not find GGUF file at {gguf_path}")
    
    # Validate file extension
    if not gguf_path.lower().endswith('.gguf'):
        raise ValueError(f"Expected .gguf file, got: {gguf_path}")
    
    # Parse the Dense GGUF Model
    try:
        reader = gguf.GGUFReader(gguf_path)
    except Exception as e:
        raise RuntimeError(f"Failed to parse GGUF file: {e}")

    if not any(len(tensor.data.shape) >= 2 for tensor in reader.tensors):
        raise ValueError("No 2D dense matrices found in GGUF file.")
    
    holo_dictionary = {}
    total_original_params = 0
    total_surviving_params = 0

    print(f"Applying Holoqubed Collapse (Threshold: {prune_threshold})...")
    
    # Iterate through every tensor in the model
    for tensor in reader.tensors:
        name = tensor.name
        data = tensor.data  # This loads the dense NumPy array
        
        # Skip 1D tensors (like layer norms or biases)
        if len(data.shape) < 2:
            holo_dictionary[name] = data.astype(np.float16)
            continue
            
        original_count = data.size
        total_original_params += original_count
        
        # The Collapse: Identify weights that survive the threshold
        mask = np.abs(data) > prune_threshold
        
        # Extract the surviving FP16 values
        surviving_weights = data[mask].astype(np.float16)
        surviving_count = surviving_weights.size
        total_surviving_params += surviving_count
        
        if surviving_count == 0:
            print(f"  [DELETED] {name} (100% Sparsity)")
            continue

        # Get the original 2D/3D indices of the survivors
        dense_indices = np.argwhere(mask)
        
        # Translate to 1D Hilbert/Morton spatial signatures
        spatial_coords = encode_hilbert_vectorized(dense_indices)
        
        # SORTING FOR O(log N) QUERY PLANNER (CRITICAL STEP)
        sort_order = np.argsort(spatial_coords)
        spatial_coords = spatial_coords[sort_order]
        surviving_weights = surviving_weights[sort_order]
        
        # Pack into CSR-style arrays
        holo_dictionary[f"{name}.coords"] = spatial_coords
        holo_dictionary[f"{name}.weights"] = surviving_weights
        
        sparsity = 100.0 * (1.0 - (surviving_count / original_count))
        print(f"  [FORGED] {name} | Sparsity: {sparsity:.2f}% | Survivors: {surviving_count:,}")

    # Export the .holo Dictionary
    print(f"\nForge Complete. Packing spatial database to {output_holo_path}...")
    
    # Use np.savez (NOT compressed) for mmap compatibility
    try:
        np.savez(output_holo_path, **holo_dictionary)
    except Exception as e:
        raise RuntimeError(f"Failed to save .holo file: {e}")
    
    total_sparsity = 100.0 * (1.0 - (total_surviving_params / total_original_params))
    stats = {
        "sparsity": total_sparsity,
        "original_params": total_original_params,
        "holo_pathways": total_surviving_params
    }
    print(f"Total Model Sparsity Achieved: {total_sparsity:.2f}%")
    print(f"Original Params: {total_original_params:,} -> Holographic Pathways: {total_surviving_params:,}")
    
    return stats


if __name__ == "__main__":
    # Example usage:
    # input_model = "models/Meta-Llama-3-8B.gguf"
    # output_holo = "models/Meta-Llama-3-8B.holo"
    # threshold = 0.08
    # 
    # forge_holo_dictionary(input_model, output_holo, prune_threshold=threshold)
    pass
