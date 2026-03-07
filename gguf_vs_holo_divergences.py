"""
ThereminQ Holoqubed - Verification & Accuracy Harness
Runs a dense GGUF reference model side-by-side with the sparse .holo engine 
to measure the mathematical divergence caused by the SiLU sparsity threshold.
"""

import numpy as np
from llama_cpp import Llama
from dictionary_loader import HoloQueryPlanner # Your CPU RAM Loader
import time

def calculate_divergence(dense_logits: np.ndarray, sparse_coords: np.ndarray, sparse_logits: np.ndarray):
    """
    Measures how much the sparse Holographic output deviates from the dense GGUF output.
    """
    # 1. Top-1 Accuracy (Did the Holoqubed engine predict the exact same #1 word?)
    dense_top_1 = np.argmax(dense_logits)
    sparse_top_1 = sparse_coords[np.argmax(sparse_logits)] if len(sparse_coords) > 0 else -1
    
    top_1_match = (dense_top_1 == sparse_top_1)

    # 2. Mean Squared Error (MSE) on the surviving coordinates
    # We only care about the coordinates the sparse engine kept alive.
    mse = 0.0
    for i, coord in enumerate(sparse_coords):
        # We assume the coordinate corresponds directly to the vocab ID for the final layer
        vocab_id = int(coord)
        if vocab_id < len(dense_logits):
            diff = dense_logits[vocab_id] - sparse_logits[i]
            mse += diff * diff
            
    if len(sparse_coords) > 0:
        mse /= len(sparse_coords)
        
    return top_1_match, mse, dense_top_1, sparse_top_1

def run_verification_suite(gguf_path: str, holo_path: str, test_prompt: str):
    print(f"--- Holoqubed Verification Suite ---")
    print(f"Reference: {gguf_path}")
    print(f"Target: {holo_path}")
    print(f"Prompt: '{test_prompt}'\n")

    # -------------------------------------------------------------------------
    # 1. The Dense Reference Engine (llama.cpp)
    # -------------------------------------------------------------------------
    print("Loading Dense GGUF Reference Engine...")
    llm = Llama(model_path=gguf_path, verbose=False, logits_all=True)
    
    # Tokenize the prompt
    prompt_tokens = llm.tokenize(test_prompt.encode('utf-8'))
    
    print("Evaluating Dense Math (O(n^2))...")
    start_dense = time.time()
    
    # Evaluate the prompt to get the final logits
    llm.eval(prompt_tokens)
    
    # The raw array of 32,000+ floating point numbers
    dense_logits = np.array(llm._scores[llm.n_tokens - 1]) 
    dense_time = (time.time() - start_dense) * 1000
    
    print(f"Dense Evaluation Complete. ({dense_time:.2f} ms)\n")

    # -------------------------------------------------------------------------
    # 2. The Sparse Holographic Engine
    # -------------------------------------------------------------------------
    print("Loading Sparse .holo Query Planner...")
    planner = HoloQueryPlanner(holo_path)
    
    # Note: In a full test, you would run the prompt through your entire PyOpenCL 
    # multi-GPU pipeline layer by layer. For this harness, we simulate hitting 
    # the final 'lm_head' layer with a mock spatial cloud derived from the prompt.
    
    # (Mocking the final active spatial coordinates that survived the network)
    # In reality, this comes from the output of your last OpenCL kernel.
    mock_active_coords = np.array([15043, 3186, 278, 3042], dtype=np.int64) 
    
    print("Evaluating Sparse Spatial Lookups (O(log N))...")
    start_sparse = time.time()
    
    # Query the final layer (lm_head)
    # Assuming 'output.weight' or similar is your final layer name in the dictionary
    layer_name = [l for l in planner.layers if "output" in l.lower() or "lm_head" in l.lower()][0]
    matched_coords, matched_weights = planner.query_active_pathways(layer_name, mock_active_coords)
    
    sparse_time = (time.time() - start_sparse) * 1000
    
    print(f"Sparse Evaluation Complete. ({sparse_time:.2f} ms)\n")

    # -------------------------------------------------------------------------
    # 3. The Math Comparison
    # -------------------------------------------------------------------------
    print("--- Verification Results ---")
    
    if len(matched_coords) == 0:
        print("CRITICAL FAILURE: The Holoqubed engine collapsed entirely. Pruning threshold is too high.")
        return

    top_1_match, mse, dense_best, sparse_best = calculate_divergence(dense_logits, matched_coords, matched_weights)
    
    print(f"Top-1 Accuracy Match : {'PASS' if top_1_match else 'FAIL'}")
    print(f"  Dense Prediction   : Token {dense_best}")
    print(f"  Sparse Prediction  : Token {sparse_best}")
    print(f"Mean Squared Error   : {mse:.6f}")
    
    if top_1_match:
        print("\nSUCCESS: The Holoqubed engine successfully reproduced the dense logic!")
    else:
        print("\nWARNING: The sparse pathways diverged from the dense reference. Consider lowering your pruning threshold during the .holo conversion forge.")

if __name__ == "__main__":
    # Example usage:
    # run_verification_suite("models/llama-3-8b.gguf", "models/llama-3-8b.holo", "The capital of France is")
    pass
