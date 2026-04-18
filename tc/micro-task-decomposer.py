import os
import sys
import json
import time
import re
import argparse
from openai import OpenAI

# ==============================================================================
# Configuration
# ==============================================================================

LLM_API_BASE = os.getenv("LLM_API_BASE", "http://192.168.2.134:8033/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "local-sk")
LLM_MODEL = os.getenv("LLM_MODEL", "nemotron-orchestrator-8b")
MAX_RETRIES = 3

client = OpenAI(base_url=LLM_API_BASE, api_key=LLM_API_KEY)

# ==============================================================================
# Hyper-Granular Decomposition Engine
# ==============================================================================

def extract_json_array(raw_text: str) -> str:
    """Uses regex to extract the first JSON array from a messy string."""
    match = re.search(r'\[.*\]', raw_text, re.DOTALL)
    if match:
        return match.group(0)
    return ""

def decompose_to_atomic_pieces(large_query: str) -> list:
    """
    Forces the LLM to break a large query down into the maximum number of 
    micro-tasks. Includes streaming output for real-time progress monitoring.
    """
    print(f"\n[1] 📥 INGRESS: Analyzing massive query...\n    Length: {len(large_query)} characters")

    system_prompt = """You are an algorithmic micro-task decomposer.
Your sole purpose is to take a large, complex query or task and shatter it into the absolute maximum number of microscopic, atomic, independent pieces possible.
Do not group steps together. If a step can be divided into two smaller steps, you MUST divide it.

CRITICAL: Output ONLY a valid, flat JSON array of strings. No markdown formatting, no conversational text.
Example: ["micro piece 1", "micro piece 2", "micro piece 3"]"""

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"[2] 🔬 DECOMPOSITION: Engaging atomic breakdown (Attempt {attempt}/{MAX_RETRIES})...")
        raw_output = ""
        
        try:
            start_time = time.time()
            
            # Enable streaming on the API call
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Decompose this to the atomic level:\n\n{large_query}"}
                ],
                temperature=0.7, 
                max_tokens=4096,
                stream=True # <--- Streaming enabled
            )
            
            print("    [~] Streaming Live Generation:\n    >> ", end="", flush=True)
            
            # Iterate through the incoming stream chunks
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    text_chunk = chunk.choices[0].delta.content
                    # Print to console immediately without a newline
                    print(text_chunk, end="", flush=True)
                    # Accumulate for our JSON parser
                    raw_output += text_chunk
            
            print("\n") # Add a newline after the stream finishes typing
            
            cleaned_output = extract_json_array(raw_output)
            
            if not cleaned_output:
                raise ValueError("Could not locate a JSON array in the LLM response.")
                
            atomic_pieces = json.loads(cleaned_output)
            
            if not isinstance(atomic_pieces, list):
                raise ValueError("LLM returned JSON, but it was not a flat array.")
                
            elapsed = round(time.time() - start_time, 2)
            print(f"    [+] Success! Shattered into {len(atomic_pieces)} distinct micro-pieces in {elapsed}s.")
            return atomic_pieces

        except json.JSONDecodeError:
            print(f"    [!] Error: Model generated invalid JSON syntax.")
            if attempt == MAX_RETRIES:
                print(f"    [RAW OUTPUT DUMP]:\n{raw_output}")
        except Exception as e:
            print(f"\n    [!] Decomposition Error: {e}")
            
        if attempt < MAX_RETRIES:
            print("    [!] Retrying...")
            time.sleep(2)

    print("    [!] Fatal: Exhausted all retries. Returning empty queue.")
    return []

# ==============================================================================
# Utility: File Output
# ==============================================================================

def export_to_manifest(pieces: list, filename: str = "atomic_manifest.json"):
    """Saves the fragmented pieces to a manifest for the orchestrator to consume."""
    if not pieces:
        print("    [!] No pieces to export. Pipeline halted.")
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
    
    print("\n    --- Sample of Micro-Pieces ---")
    for i, piece in enumerate(pieces[:5]): 
        print(f"    {i+1}. {piece}")
    if len(pieces) > 5:
        print(f"    ... and {len(pieces) - 5} more.")

# ==============================================================================
# Execution
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyper-Granular LLM Task Decomposer")
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-f", "--file", type=str, help="Path to a text file containing the prompt/query.")
    group.add_argument("-p", "--prompt", type=str, help="Direct string input of the prompt/query.")
    
    args = parser.parse_args()
    
    if args.file:
        if not os.path.exists(args.file):
            print(f"[!] Fatal Error: The file '{args.file}' does not exist.")
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8") as f:
            target_query = f.read()
        print(f"[*] Loaded query from file: {args.file}")
        
    elif args.prompt:
        target_query = args.prompt
        print("[*] Loaded query from command line argument.")
        
    else:
        print("[*] No input arguments provided. Using default test query.")
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
