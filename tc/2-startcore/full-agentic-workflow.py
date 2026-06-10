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

ORCHESTRATOR_ENDPOINTS = [
    "http://192.168.2.137:8080/v1",
    "http://192.168.2.137:8080/v1"
]
ORCHESTRATOR_MODEL = os.getenv("ORCHESTRATOR_MODEL", "nvidia_Orchestrator-8B-Q6_K.gguf")
ORCH_API_KEY = os.getenv("ORCH_API_KEY", "local-sk")
MAX_RETRIES = 3

WORKER_ENDPOINTS = [
    "http://192.168.2.137:8034/v1",
    "http://192.168.2.137:8035/v1"
]
WORKER_MODEL = os.getenv("WORKER_MODEL", "Qwen3.5-9B-IQ4_XS.gguf")
WORKER_API_KEY = os.getenv("WORKER_API_KEY", "local-sk")

WORKER_PARALLEL_SLOTS = 2
WORKER_RETRIES = 3
ORCH_PARALLEL_SLOTS = 2

MAP_REDUCE_BATCH_SIZE = 3
BASE_DIR = Path(__file__).parent

# ==============================================================================
# Helper: Fallback Token Estimator
# ==============================================================================

def estimate_tokens(text: str) -> int:
    """Fallback token calculation if local endpoints drop the usage object."""
    return len(str(text)) // 4

# ==============================================================================
# Phase 1: Hyper-Granular Decomposition
# ==============================================================================

def extract_json_array(raw_text: str) -> str:
    cleaned_text = re.sub(r'```json\s*', '', raw_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'```\s*', '', cleaned_text)
    match = re.search(r'\[.*?\]', cleaned_text, re.DOTALL)
    return match.group(0) if match else ""

def decompose_to_atomic_pieces(large_query: str) -> tuple:
    print(f"\n[1] 📥 INGRESS: Analyzing massive query...\n    Length: {len(large_query)} characters", flush=True)

    system_prompt = """You are an algorithmic micro-task decomposer.
Your sole purpose is to take a large, complex query or task and shatter it into atomic, independent pieces for parallel processing.
Output ONLY a valid, flat JSON array of strings. No markdown formatting, no conversational text."""

    target_orch = ORCHESTRATOR_ENDPOINTS[0]
    client = OpenAI(base_url=target_orch, api_key=ORCH_API_KEY)

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"[2] 🔬 DECOMPOSITION: Engaging atomic breakdown via {target_orch} (Attempt {attempt}/{MAX_RETRIES})...", flush=True)
        raw_output = ""
        prompt_tokens = 0
        comp_tokens = 0
        
        try:
            start_time = time.time()
            response = client.chat.completions.create(
                model=ORCHESTRATOR_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Decompose this to the atomic level:\n\n{large_query}"}
                ],
                temperature=0.7, 
                max_tokens=40960,
                stream=True,
                stream_options={"include_usage": True},
                timeout=600.0
            )
            
            print("    [~] Streaming Live Generation:\n    >> ", end="", flush=True)
            
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    text_chunk = chunk.choices[0].delta.content
                    print(text_chunk, end="", flush=True)
                    raw_output += text_chunk
                
                if hasattr(chunk, 'usage') and chunk.usage is not None:
                    prompt_tokens = chunk.usage.prompt_tokens
                    comp_tokens = chunk.usage.completion_tokens
                    
            print("\n", flush=True) 
            
            cleaned_output = extract_json_array(raw_output)
            if not cleaned_output:
                raise ValueError("Could not locate a JSON array in the LLM response.")
                
            atomic_pieces = json.loads(cleaned_output)
            
            # Apply Fallback if stream stripped usage
            if prompt_tokens == 0 and comp_tokens == 0:
                prompt_tokens = estimate_tokens(system_prompt + large_query)
                comp_tokens = estimate_tokens(raw_output)
                
            elapsed = round(time.time() - start_time, 2)
            print(f"    [+] Success! Shattered into {len(atomic_pieces)} distinct micro-pieces in {elapsed}s.", flush=True)
            return atomic_pieces, prompt_tokens, comp_tokens

        except Exception as e:
            print(f"\n    [!] Decomposition Error: {e}", flush=True)
            if attempt < MAX_RETRIES:
                target_orch = ORCHESTRATOR_ENDPOINTS[attempt % len(ORCHESTRATOR_ENDPOINTS)]
                client = OpenAI(base_url=target_orch, api_key=ORCH_API_KEY)
                time.sleep(2)

    return [large_query], estimate_tokens(large_query), 10

# ==============================================================================
# Phase 2: Audit Trail / File Export Setup
# ==============================================================================

def export_to_split_files(pieces: list) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = BASE_DIR / f"runs/orchestrator_run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    if len(pieces) <= 1: return run_dir
        
    print("\n[3] 💾 QUEUE EXPORT: Saving task matrix to disk...", flush=True)
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
    print(f"    -> [Thread-{task_id:02d}] Dispatched to {endpoint} | Task: '{task_prompt[:40]}...' ", flush=True)
    
    worker_client = OpenAI(base_url=endpoint, api_key=WORKER_API_KEY)
    start_time = time.time()
    
    system_instruction = (
        "You are an autonomous, highly-capable worker agent equipped with advanced reasoning. "
        "Think step-by-step to formulate a plan. You must execute your specific objective fully. "
        "CRITICAL: If your task involves writing code, creating configurations, or generating files, "
        "you MUST output the file contents wrapped exactly in these XML tags:\n"
        '<file path="filename.ext">\n[YOUR FILE CONTENT HERE]\n</file>\n'
    )
    user_instruction = f"BACKGROUND CONTEXT:\n{original_query}\n\nYOUR SPECIFIC OBJECTIVE:\n{task_prompt}"
    
    saved_artifacts = []
    status = "success"
    
    try:
        response = worker_client.chat.completions.create(
            model=WORKER_MODEL,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_instruction}
            ],
            temperature=0.4, max_tokens=65536, timeout=1200.0,   
        )
        result_text = response.choices[0].message.content.strip()
        
        if response.usage and response.usage.prompt_tokens > 0:
            prompt_tokens = response.usage.prompt_tokens
            comp_tokens = response.usage.completion_tokens
        else:
            prompt_tokens = estimate_tokens(system_instruction + user_instruction)
            comp_tokens = estimate_tokens(result_text)
            
        tot_tokens = prompt_tokens + comp_tokens
        
        file_matches = re.finditer(r'<file\s+path="([^"]+)">([\s\S]*?)</file>', result_text, re.IGNORECASE)
        for match in file_matches:
            file_path, file_content = match.group(1).strip(), match.group(2).strip()
            safe_filename = os.path.basename(file_path)
            artifact_dir = run_dir / "artifacts" / f"thread_{task_id:02d}"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            
            with open(artifact_dir / safe_filename, "w", encoding="utf-8") as af:
                af.write(file_content)
            saved_artifacts.append(safe_filename)

        if len(result_text) < 20: status = "failed_validation"
            
    except Exception as e:
        result_text, status = f"Worker Error: {str(e)}", "error"
        prompt_tokens, comp_tokens, tot_tokens = 0, 0, 0

    elapsed = round(time.time() - start_time, 2)
    task_tps = round(comp_tokens / elapsed, 2) if elapsed > 0 else 0
    
    return {
        "id": task_id, "prompt": task_prompt, "result": result_text,
        "artifacts": saved_artifacts, "status": status,
        "prompt_tokens": prompt_tokens, "completion_tokens": comp_tokens,
        "total_tokens": tot_tokens, "elapsed": elapsed, "tps": task_tps
    }

def dispatch_and_gather(sub_tasks: list, original_query: str, run_dir: Path) -> list:
    max_concurrent = len(WORKER_ENDPOINTS) * WORKER_PARALLEL_SLOTS
    print(f"\n[4] 🚀 DISPATCH: Firing up to {max_concurrent} simultaneous tasks using Dynamic Load Balancing...", flush=True)
    
    endpoint_queue = queue.Queue()
    for ep in WORKER_ENDPOINTS:
        for _ in range(WORKER_PARALLEL_SLOTS): endpoint_queue.put(ep)

    def dynamic_worker(tid: int, prompt: str):
        last_result = None
        for attempt in range(1, WORKER_RETRIES + 1):
            endpoint = endpoint_queue.get()
            try:
                result = process_subtask(tid, prompt, endpoint, original_query, run_dir)
                if result["status"] == "success": return result
                last_result = result
            except Exception as e:
                last_result = {"id": tid, "status": "error", "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "elapsed": 0, "tps": 0}
            finally:
                endpoint_queue.put(endpoint)
            time.sleep(2)
        return last_result

    results = []
    aggregate_total_tokens = 0
    aggregate_completion_tokens = 0
    
    # strictly tracks true parallel duration
    dispatch_start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        future_to_task = {executor.submit(dynamic_worker, i + 1, task): i + 1 for i, task in enumerate(sub_tasks)}
        
        for future in concurrent.futures.as_completed(future_to_task):
            task_result = future.result()
            results.append(task_result)
            
            aggregate_total_tokens += task_result["total_tokens"]
            aggregate_completion_tokens += task_result["completion_tokens"]
            
            current_elapsed = time.time() - dispatch_start_time
            agg_tps = round(aggregate_completion_tokens / current_elapsed, 2) if current_elapsed > 0 else 0
            
            print(f"    <- [Thread-{task_result['id']:02d}] Finished in {task_result['elapsed']}s | "
                  f"Status: {task_result['status']} | "
                  f"Task TPS: {task_result['tps']} | "
                  f"Agg TPS: {agg_tps} | "
                  f"Total Tokens: {aggregate_total_tokens:,}", flush=True)

    results.sort(key=lambda x: x["id"])
    return results

# ==============================================================================
# Phase 4: Parallel Map-Reduce & Deduplication Cluster
# ==============================================================================

def process_map_reduce_batch(batch_idx: int, batch: list, endpoint: str, original_query: str) -> tuple:
    client = OpenAI(base_url=endpoint, api_key=ORCH_API_KEY)
    
    system_prompt = "You are the Map-Reduce compression layer of a distributed AI cluster. DEDUPLICATE, RETAIN, and CONDENSE."
    batch_context = ""
    for t in batch:
        batch_context += f"--- Sub-Task: {t['prompt']} ---\nRESULT:\n{t['result']}\n\n"
        
    user_prompt = f"ORIGINAL QUERY: {original_query}\n\nREPORTS TO CONDENSE:\n{batch_context}"
    
    try:
        start_time = time.time()
        response = client.chat.completions.create(
            model=ORCHESTRATOR_MODEL,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.3, max_tokens=16384
        )
        elapsed = round(time.time() - start_time, 2)
        print(f"    [+] Batch {batch_idx} compressed via {endpoint} in {elapsed}s.", flush=True)
        
        res_content = response.choices[0].message.content.strip()
        if response.usage and response.usage.prompt_tokens > 0:
            return res_content, response.usage.prompt_tokens, response.usage.completion_tokens
        else:
            return res_content, estimate_tokens(system_prompt + user_prompt), estimate_tokens(res_content)
            
    except Exception as e:
        print(f"    [!] Error during Map-Reduce on batch {batch_idx}: {e}", flush=True)
        return f"Batch {batch_idx} Failed.", 0, 0

def map_reduce_deduplication(completed_tasks: list, original_query: str) -> tuple:
    print(f"\n[5] 🗜️ MAP-REDUCE: Activating orchestrator pool to compress {len(completed_tasks)} tasks...", flush=True)
    batches = [completed_tasks[i:i + MAP_REDUCE_BATCH_SIZE] for i in range(0, len(completed_tasks), MAP_REDUCE_BATCH_SIZE)]
    
    orch_queue = queue.Queue()
    for ep in ORCHESTRATOR_ENDPOINTS:
        for _ in range(ORCH_PARALLEL_SLOTS): orch_queue.put(ep)
            
    max_orch_concurrency = len(ORCHESTRATOR_ENDPOINTS) * ORCH_PARALLEL_SLOTS
    condensed_reports = [None] * len(batches)
    total_prompt_tokens = 0
    total_comp_tokens = 0

    def worker_wrapper(idx, batch):
        endpoint = orch_queue.get()
        try:
            return idx, *process_map_reduce_batch(idx, batch, endpoint, original_query)
        finally: orch_queue.put(endpoint)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_orch_concurrency) as executor:
        futures = {executor.submit(worker_wrapper, i + 1, b): i for i, b in enumerate(batches)}
        for future in concurrent.futures.as_completed(futures):
            batch_id, result_text, p_tok, c_tok = future.result()
            condensed_reports[batch_id - 1] = result_text
            total_prompt_tokens += p_tok
            total_comp_tokens += c_tok

    return [r for r in condensed_reports if r is not None], total_prompt_tokens, total_comp_tokens

# ==============================================================================
# Phase 5: Synthesis
# ==============================================================================

def synthesize_results(original_query: str, condensed_reports: list) -> tuple:
    print("\n[6] 🧠 SYNTHESIS: Consolidating reduced reports into final output...", flush=True)
    consolidated_context = "\n".join([f"--- CONDENSED BATCH {i+1} ---\n{report}\n" for i, report in enumerate(condensed_reports)])
    
    system_prompt = "You are the Synthesis Layer. Merge these briefs into a cohesive final response."
    client = OpenAI(base_url=ORCHESTRATOR_ENDPOINTS[0], api_key=ORCH_API_KEY)
    user_prompt = f"ORIGINAL QUERY: {original_query}\n\nCONDENSED BRIEFS:\n{consolidated_context}"

    try:
        response = client.chat.completions.create(
            model=ORCHESTRATOR_MODEL,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.5, max_tokens=40960 
        )
        print("    [+] Synthesis complete.", flush=True)
        res_content = response.choices[0].message.content.strip()
        
        if response.usage and response.usage.prompt_tokens > 0:
            return res_content, response.usage.prompt_tokens, response.usage.completion_tokens
        else:
            return res_content, estimate_tokens(system_prompt + user_prompt), estimate_tokens(res_content)
            
    except Exception as e:
        return f"Synthesis Failed. Error: {str(e)}", 0, 0

# ==============================================================================
# Main Execution Engine
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Clustered LLM Orchestrator")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-f", "--file", type=str, help="Path to prompt file.")
    group.add_argument("-p", "--prompt", type=str, help="Direct prompt input.")
    args = parser.parse_args()
    
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f: target_query = f.read()
    elif args.prompt:
        target_query = args.prompt
    else:
        target_query = "Create a simple Python HTTP server using Flask that returns 'Hello, Orchestrator!' on the root endpoint. Also, write a standard Dockerfile to containerize it."
        
    print("\n=== STARTING UNIFIED DISTRIBUTED ORCHESTRATOR CLUSTER ===", flush=True)
    
    master_start_time = time.time()
    global_input_tokens = 0
    global_output_tokens = 0
    
    # 1. Decomposition
    fragments, p_tok, c_tok = decompose_to_atomic_pieces(target_query)
    global_input_tokens += p_tok
    global_output_tokens += c_tok
    
    # 2. File Setup
    run_directory = export_to_split_files(fragments)
    
    # 3. Threaded Worker Dispatch
    worker_results = dispatch_and_gather(fragments, target_query, run_directory)
    for res in worker_results:
        global_input_tokens += res.get("prompt_tokens", 0)
        global_output_tokens += res.get("completion_tokens", 0)
    
    # 4. Clustered/Parallel Map-Reduce Pass
    condensed_results, p_tok, c_tok = map_reduce_deduplication(worker_results, target_query)
    global_input_tokens += p_tok
    global_output_tokens += c_tok
    
    # 5. Final Synthesis
    final_output, p_tok, c_tok = synthesize_results(target_query, condensed_results)
    global_input_tokens += p_tok
    global_output_tokens += c_tok
    
    master_elapsed_time = time.time() - master_start_time
    
    print("\n[7] 💾 MASTER EXPORT: Saving final synthesis to disk...", flush=True)
    final_file_path = run_directory / "FINAL_SYNTHESIS.md"
    with open(final_file_path, "w", encoding="utf-8") as f:
        f.write(final_output)
    
    print("\n==============================================================================", flush=True)
    print("✨ EXECUTION RUN COMPLETE ✨", flush=True)
    print(f"    ⏱️  Total Wall-Clock Time:  {master_elapsed_time:.2f} seconds", flush=True)
    print(f"    📥 Total Input Tokens:     {global_input_tokens:,}", flush=True)
    print(f"    📤 Total Output Tokens:    {global_output_tokens:,}", flush=True)
    print(f"    📊 Total Cluster Tokens:   {global_input_tokens + global_output_tokens:,}", flush=True)
    print(f"    [+] Synthesis Payload Size: {len(final_output):,} characters", flush=True)
    print(f"    📂 Run Master Directory:    {run_directory.absolute()}", flush=True)
    print("==============================================================================", flush=True)
