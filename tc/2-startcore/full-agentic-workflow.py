import os
import sys
import json
import time
import re
import argparse
import concurrent.futures
import queue
from datetime import datetime
from pathlib import Path
from openai import OpenAI

# ==============================================================================
# Configuration & Endpoints
# ==============================================================================

# Orchestrator Node (Handles Decomposition & Synthesis)
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://192.168.2.137:8080/v1")
ORCHESTRATOR_MODEL = os.getenv("ORCHESTRATOR_MODEL", "nvidia_Orchestrator-8B-Q6_K.gguf")
ORCH_API_KEY = os.getenv("ORCH_API_KEY", "local-sk")
MAX_RETRIES = 3

# Worker Nodes (Agentic Qwen instances with 128k context, reasoning, and tools)
WORKER_ENDPOINTS = [
    "http://192.168.2.137:8034/v1",
    "http://192.168.2.137:8035/v1"
]
WORKER_MODEL = os.getenv("WORKER_MODEL", "Qwen3.5-9B-IQ4_XS.gguf")
WORKER_API_KEY = os.getenv("WORKER_API_KEY", "local-sk")

# Server Concurrency & Resilience
WORKER_PARALLEL_SLOTS = 2
WORKER_RETRIES = 3

# Setup Orchestrator Client
orch_client = OpenAI(base_url=ORCHESTRATOR_URL, api_key=ORCH_API_KEY)

# Base directory for runs
BASE_DIR = Path(__file__).parent

# ==============================================================================
# Phase 1: Hyper-Granular Decomposition
# ==============================================================================

def extract_json_array(raw_text: str) -> str:
    """Uses regex to extract the first JSON array from a messy string."""
    cleaned_text = re.sub(r'```json\s*', '', raw_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'```\s*', '', cleaned_text)
    
    match = re.search(r'\[.*?\]', cleaned_text, re.DOTALL)
    if match:
        return match.group(0)
    return ""

def decompose_to_atomic_pieces(large_query: str) -> list:
    """Forces the LLM to break a large query down into atomic micro-tasks with streaming."""
    print(f"\n[1] 📥 INGRESS: Analyzing massive query...\n    Length: {len(large_query)} characters", flush=True)

    system_prompt = """You are an algorithmic micro-task decomposer.
Your sole purpose is to take a large, complex query or task and shatter it into atomic, independent pieces for parallel processing.
Output ONLY a valid, flat JSON array of strings. No markdown formatting, no conversational text.
Example: ["micro piece 1", "micro piece 2", "micro piece 3"]"""

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"[2] 🔬 DECOMPOSITION: Engaging atomic breakdown (Attempt {attempt}/{MAX_RETRIES})...", flush=True)
        raw_output = ""
        
        try:
            start_time = time.time()
            response = orch_client.chat.completions.create(
                model=ORCHESTRATOR_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Decompose this to the atomic level:\n\n{large_query}"}
                ],
                temperature=0.7, 
                max_tokens=4096,
                stream=True,
                timeout=300.0
            )
            
            print("    [~] Streaming Live Generation:\n    >> ", end="", flush=True)
            
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    text_chunk = chunk.choices[0].delta.content
                    print(text_chunk, end="", flush=True)
                    raw_output += text_chunk
            print("\n", flush=True) 
            
            cleaned_output = extract_json_array(raw_output)
            if not cleaned_output:
                raise ValueError("Could not locate a JSON array in the LLM response.")
                
            atomic_pieces = json.loads(cleaned_output)
            if not isinstance(atomic_pieces, list):
                raise ValueError("LLM returned JSON, but it was not a flat array.")
                
            elapsed = round(time.time() - start_time, 2)
            print(f"    [+] Success! Shattered into {len(atomic_pieces)} distinct micro-pieces in {elapsed}s.", flush=True)
            return atomic_pieces

        except Exception as e:
            print(f"\n    [!] Decomposition Error: {e}", flush=True)
            if attempt < MAX_RETRIES:
                print("    [!] Retrying...", flush=True)
                time.sleep(2)

    print("    [!] Fatal: Exhausted all retries. Falling back to single-task execution.", flush=True)
    return [large_query]

# ==============================================================================
# Phase 2: Audit Trail / File Export Setup
# ==============================================================================

def export_to_split_files(pieces: list) -> Path:
    """Shatters the queue into individual markdown files. Returns the master run directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = BASE_DIR / f"runs/orchestrator_run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    if len(pieces) <= 1:
        return run_dir
        
    print(f"\n[3] 💾 QUEUE EXPORT: Saving task matrix to disk...", flush=True)
    
    tasks_dir = run_dir / "tasks"
    tasks_dir.mkdir(exist_ok=True)
    
    for idx, piece in enumerate(pieces, start=1):
        filepath = tasks_dir / f"task_{idx:03d}.md"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"{piece.strip()}\n")
            
    print(f"    [+] Saved {len(pieces)} task files to {tasks_dir.absolute()}/", flush=True)
    return run_dir

# ==============================================================================
# Phase 3: Parallel Dispatch & Artifact Harvesting
# ==============================================================================

def process_subtask(task_id: int, task_prompt: str, endpoint: str, original_query: str, run_dir: Path) -> dict:
    """Worker thread function: executes agentic sub-task and harvests generated files."""
    print(f"    -> [Thread-{task_id:02d}] Dispatched to {endpoint} | Task: '{task_prompt[:40]}...' ", flush=True)
    
    worker_client = OpenAI(base_url=endpoint, api_key=WORKER_API_KEY)
    start_time = time.time()
    
    system_instruction = (
        "You are an autonomous, highly-capable worker agent equipped with advanced reasoning. "
        "Think step-by-step to formulate a plan. You must execute your specific objective fully. "
        "CRITICAL: If your task involves writing code, creating configurations, or generating files, "
        "you MUST output the file contents wrapped exactly in these XML tags:\n"
        '<file path="filename.ext">\n[YOUR FILE CONTENT HERE]\n</file>\n'
        "Do this for every file you generate so the orchestrator can extract them."
    )
    
    user_instruction = (
        f"BACKGROUND CONTEXT (The overall project):\n{original_query}\n\n"
        f"YOUR SPECIFIC OBJECTIVE:\n{task_prompt}"
    )
    
    saved_artifacts = []
    comp_tokens = 0
    tot_tokens = 0
    status = "success"
    
    try:
        response = worker_client.chat.completions.create(
            model=WORKER_MODEL,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_instruction}
            ],
            temperature=0.4,
            max_tokens=32768, 
            timeout=1200.0,   
        )
        result_text = response.choices[0].message.content.strip()
        
        # Token Tracking
        if response.usage:
            comp_tokens = response.usage.completion_tokens
            tot_tokens = response.usage.total_tokens
        
        # --- ARTIFACT EXTRACTION LOGIC ---
        file_matches = re.finditer(r'<file\s+path="([^"]+)">([\s\S]*?)</file>', result_text, re.IGNORECASE)
        for match in file_matches:
            file_path = match.group(1).strip()
            file_content = match.group(2).strip()
            
            safe_filename = os.path.basename(file_path)
            artifact_dir = run_dir / "artifacts" / f"thread_{task_id:02d}"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            
            full_path = artifact_dir / safe_filename
            with open(full_path, "w", encoding="utf-8") as af:
                af.write(file_content)
                
            saved_artifacts.append(safe_filename)

        if len(result_text) < 20:
            status = "failed_validation (output too short)"
            
    except Exception as e:
        result_text = f"Worker Error: {str(e)}"
        status = "error"

    elapsed = round(time.time() - start_time, 2)
    task_tps = round(comp_tokens / elapsed, 2) if elapsed > 0 else 0
    
    return {
        "id": task_id,
        "prompt": task_prompt,
        "result": result_text,
        "artifacts": saved_artifacts,
        "status": status,
        "completion_tokens": comp_tokens,
        "total_tokens": tot_tokens,
        "elapsed": elapsed,
        "tps": task_tps
    }

def dispatch_and_gather(sub_tasks: list, original_query: str, run_dir: Path) -> list:
    """Distributes tasks across a dynamic pool of worker endpoints with auto-retries."""
    max_concurrent = len(WORKER_ENDPOINTS) * WORKER_PARALLEL_SLOTS
    print(f"\n[4] 🚀 DISPATCH: Firing up to {max_concurrent} simultaneous tasks using Dynamic Load Balancing...", flush=True)
    
    # 1. Initialize the dynamic endpoint pool
    endpoint_queue = queue.Queue()
    for ep in WORKER_ENDPOINTS:
        for _ in range(WORKER_PARALLEL_SLOTS):
            endpoint_queue.put(ep)

    # 2. Wrapper function to manage checkout/check-in of endpoints and retries
    def dynamic_worker(tid: int, prompt: str):
        last_result = None
        for attempt in range(1, WORKER_RETRIES + 1):
            # This will block if all endpoint slots are currently checked out
            endpoint = endpoint_queue.get()
            
            try:
                if attempt > 1:
                    print(f"    [↻] [Thread-{tid:02d}] Retrying (Attempt {attempt}/{WORKER_RETRIES}) on {endpoint}...", flush=True)
                    
                result = process_subtask(tid, prompt, endpoint, original_query, run_dir)
                
                # If successful, immediately return the good result
                if result["status"] == "success":
                    return result
                    
                # Otherwise, log the failure and prepare for next attempt
                last_result = result
                print(f"    [!] [Thread-{tid:02d}] Task failed on {endpoint} | Status: {result['status']}", flush=True)
                
            except Exception as e:
                print(f"    [!] [Thread-{tid:02d}] Hard crash on {endpoint}: {e}", flush=True)
                last_result = {
                    "id": tid, "prompt": prompt, "result": f"Hard crash: {str(e)}", 
                    "artifacts": [], "status": "error", "completion_tokens": 0, 
                    "total_tokens": 0, "elapsed": 0, "tps": 0
                }
            finally:
                # Crucial: Always return the endpoint slot to the queue
                endpoint_queue.put(endpoint)
                
            # Brief cool-down before pulling a new endpoint from the queue for the retry
            if attempt < WORKER_RETRIES:
                time.sleep(2)
                
        # If we exit the loop, all retries failed
        print(f"    [❌] [Thread-{tid:02d}] Exhausted all {WORKER_RETRIES} retries. Marking task as failed.", flush=True)
        return last_result

    results = []
    aggregate_total_tokens = 0
    aggregate_completion_tokens = 0
    aggregate_time = 0.0

    # 3. Fire tasks into the executor
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        future_to_task = {
            executor.submit(dynamic_worker, i + 1, task): i + 1 
            for i, task in enumerate(sub_tasks)
        }
        
        for future in concurrent.futures.as_completed(future_to_task):
            try:
                task_result = future.result()
                results.append(task_result)
                
                # Metric Aggregation
                aggregate_total_tokens += task_result["total_tokens"]
                aggregate_completion_tokens += task_result["completion_tokens"]
                aggregate_time += task_result["elapsed"]
                
                agg_tps = round(aggregate_completion_tokens / aggregate_time, 2) if aggregate_time > 0 else 0
                
                print(f"    <- [Thread-{task_result['id']:02d}] Finished in {task_result['elapsed']}s | "
                      f"Status: {task_result['status']} | "
                      f"Task TPS: {task_result['tps']} | "
                      f"Agg TPS: {agg_tps} | "
                      f"Total Tokens: {aggregate_total_tokens:,}", flush=True)
                      
            except Exception as exc:
                print(f"    [!] Thread generated an exception: {exc}", flush=True)

    results.sort(key=lambda x: x["id"])
    return results

# ==============================================================================
# Phase 4: Synthesis
# ==============================================================================

def synthesize_results(original_query: str, completed_tasks: list) -> str:
    """Takes all worker outputs and synthesizes the final comprehensive answer."""
    print(f"\n[5] 🧠 SYNTHESIS: Consolidating worker progress into final output...", flush=True)
    
    context_blocks = []
    for t in completed_tasks:
        artifact_note = f" (Generated Files: {', '.join(t['artifacts'])})" if t['artifacts'] else ""
        context_blocks.append(
            f"--- Sub-Task: {t['prompt']} ---\n"
            f"STATUS: {t['status']}{artifact_note}\n"
            f"RESULT:\n{t['result']}\n"
        )
    
    consolidated_context = "\n".join(context_blocks)
    
    system_prompt = """You are the Synthesis Layer of a master AI orchestrator.
Read the original user query and the compiled reports from multiple autonomous worker nodes.
Merge these separate reports into a single, cohesive, highly-detailed final response.
If the workers generated code files (artifacts), explicitly list them and explain how they connect together.
Resolve any contradictions, remove redundancies, and directly fulfill the user's original query."""

    user_prompt = f"ORIGINAL QUERY: {original_query}\n\nWORKER REPORTS:\n{consolidated_context}"

    try:
        response = orch_client.chat.completions.create(
            model=ORCHESTRATOR_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5,
            max_tokens=8192 
        )
        
        final_answer = response.choices[0].message.content.strip()
        print("    [+] Synthesis complete.", flush=True)
        return final_answer
        
    except Exception as e:
        print(f"    [!] Error during synthesis: {e}", flush=True)
        return f"Synthesis Phase Failed. Orchestrator Error: {str(e)}"

# ==============================================================================
# Main Execution Engine
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Local LLM Orchestrator Engine")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-f", "--file", type=str, help="Path to a text file containing the prompt/query.")
    group.add_argument("-p", "--prompt", type=str, help="Direct string input of the prompt/query.")
    args = parser.parse_args()
    
    if args.file:
        if not os.path.exists(args.file):
            print(f"[!] Fatal Error: The file '{args.file}' does not exist.", flush=True)
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8") as f:
            target_query = f.read()
        print(f"[*] Loaded query from file: {args.file}", flush=True)
    elif args.prompt:
        target_query = args.prompt
        print("[*] Loaded query from command line argument.", flush=True)
    else:
        print("[*] No input arguments provided. Using default complex query.", flush=True)
        target_query = """
        Build a complete, secure, production-ready React and Node.js e-commerce application. 
        It needs a PostgreSQL database, user authentication via JWT, a product catalog with 
        search and filtering, a shopping cart, Stripe payment integration, order history, 
        and an admin dashboard to manage inventory. Write all the code, setup instructions, 
        and deployment scripts using Docker and AWS ECS.
        """
        
    print("\n=== STARTING UNIFIED ORCHESTRATOR ENGINE ===", flush=True)
    
    fragments = decompose_to_atomic_pieces(target_query)
    run_directory = export_to_split_files(fragments)
    worker_results = dispatch_and_gather(fragments, target_query, run_directory)
    final_output = synthesize_results(target_query, worker_results)
    
    print(f"\n[6] 💾 MASTER EXPORT: Saving final synthesis to disk...", flush=True)
    final_file_path = run_directory / "FINAL_SYNTHESIS.md"
    try:
        with open(final_file_path, "w", encoding="utf-8") as f:
            f.write(final_output)
        print(f"    [+] Successfully saved final output to: {final_file_path.absolute()}", flush=True)
    except Exception as e:
        print(f"    [!] Failed to save final output to disk: {e}", flush=True)
    
    print("\n==============================================================================", flush=True)
    print("✨ FINAL SYNTHESIZED OUTPUT ✨\n", flush=True)
    print(final_output, flush=True)
    print("\n==============================================================================", flush=True)
    print(f"📂 Run Master Directory: {run_directory.absolute()}", flush=True)
