import os
import sys
import json
import time
from openai import OpenAI

# ==============================================================================
# Configuration
# ==============================================================================

# Point this to your local orchestrator model (e.g., Nemotron, Llama-3, etc.)
LLM_API_BASE = os.getenv("LLM_API_BASE", "http://127.0.0.1:8080/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "local-sk")
LLM_MODEL = os.getenv("LLM_MODEL", "nemotron-orchestrator-8b")

client = OpenAI(base_url=LLM_API_BASE, api_key=LLM_API_KEY)

# ==============================================================================
# Hyper-Granular Decomposition Engine
# ==============================================================================

def decompose_to_atomic_pieces(large_query: str) -> list:
    """
    Forces the LLM to break a large query down into the maximum number of 
    micro-tasks or microscopic data chunks possible.
    """
    print(f"\n[1] 📥 INGRESS: Analyzing massive query...\n    Length: {len(large_query)} characters")
    print(f"[2] 🔬 DECOMPOSITION: Engaging atomic breakdown...")

    system_prompt = """You are an algorithmic micro-task decomposer.
Your sole purpose is to take a large, complex query or task and shatter it into the absolute maximum number of microscopic, atomic, independent pieces possible.
Do not group steps together. If a step can be divided into two smaller steps, you MUST divide it.
Think at the most granular level possible (e.g., instead of "setup database", output "define schema", "write create table query for users", "write create table query for posts", "configure connection string").

CRITICAL: Output ONLY a valid, flat JSON array of strings. No markdown formatting, no conversational text, no nested objects.
Example: ["micro piece 1", "micro piece 2", "micro piece 3"]"""

    try:
        start_time = time.time()
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Decompose this to the atomic level:\n\n{large_query}"}
            ],
            temperature=0.7, # Slightly higher temperature encourages more creative fragmentation
            max_tokens=4096  # Allow a massive output window for hundreds of pieces
        )
        
        raw_output = response.choices[0].message.content.strip()
        
        # Aggressive cleanup for local LLM formatting hallucinations
        if raw_output.startswith("```json"):
            raw_output = raw_output[7:-3].strip()
        elif raw_output.startswith("```"):
            raw_output = raw_output[3:-3].strip()
            
        atomic_pieces = json.loads(raw_output)
        
        if not isinstance(atomic_pieces, list):
            raise ValueError("LLM did not return a JSON list.")
            
        elapsed = round(time.time() - start_time, 2)
        print(f"    [+] Success! Shattered into {len(atomic_pieces)} distinct micro-pieces in {elapsed}s.")
        return atomic_pieces

    except json.JSONDecodeError:
        print("    [!] Error: The model failed to output strictly valid JSON.")
        print(f"    [RAW OUTPUT DUMP]:\n{raw_output}")
        return []
    except Exception as e:
        print(f"    [!] Decomposition Error: {e}")
        return []

# ==============================================================================
# Utility: File Output
# ==============================================================================

def export_to_manifest(pieces: list, filename: str = "atomic_manifest.json"):
    """Saves the fragmented pieces to a manifest for the orchestrator to consume."""
    if not pieces:
        print("    [!] No pieces to export.")
        return
        
    os.makedirs("wiki/manifests", exist_ok=True)
    filepath = os.path.join("wiki/manifests", filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump({
            "total_pieces": len(pieces),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "atomic_queue": pieces
        }, f, indent=4)
        
    print(f"\n[3] 💾 EXPORT: Saved atomic manifest to {filepath}")
    
    # Print a sample of the extreme granularity
    print("\n    --- Sample of Micro-Pieces ---")
    for i, piece in enumerate(pieces[:5]): # Show first 5
        print(f"    {i+1}. {piece}")
    if len(pieces) > 5:
        print(f"    ... and {len(pieces) - 5} more.")

# ==============================================================================
# Execution
# ==============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Read from file if the user passes a filepath, otherwise treat as raw string
        input_arg = " ".join(sys.argv[1:])
        if os.path.exists(input_arg):
            with open(input_arg, "r", encoding="utf-8") as f:
                target_query = f.read()
        else:
            target_query = input_arg
    else:
        # Default massive test query
        target_query = """
        Build a complete, secure, production-ready React and Node.js e-commerce application. 
        It needs a PostgreSQL database, user authentication via JWT, a product catalog with 
        search and filtering, a shopping cart, Stripe payment integration, order history, 
        and an admin dashboard to manage inventory. Write all the code, setup instructions, 
        and deployment scripts using Docker and AWS ECS.
        """
        
    print("=== STARTING ATOMIC DECOMPOSER ===")
    
    fragments = decompose_to_atomic_pieces(target_query)
    export_to_manifest(fragments)
    
    print("\n=== PIPELINE FINISHED ===")
