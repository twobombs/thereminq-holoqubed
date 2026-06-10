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
# Phase 3: Rolling Synthesis Pipeline
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

def rolling_synthesis_step(step_idx: int, total_steps: int, current_synthesis: str, task: dict, endpoint: str, original_query: str) -> tuple:
    client = OpenAI(base_url=endpoint, api_key=ORCH_API_KEY)
    
    system_prompt = (
        "You are the Master Synthesizer of a distributed AI cluster. "
        "Your job is to seamlessly integrate incoming sequential worker reports into a master unified document. "
        "DEDUPLICATE overlapping information. Retain all original code and configurations. "
        "Format cleanly to create a cohesive final output."
    )
    
    if not current_synthesis:
        user_prompt = (
            f"ORIGINAL QUERY: {original_query}\n\n"
            f"--- INITIAL DATA (Step {step_idx}/{total_steps}) ---\n"
            f"Task Addressed: {task['prompt']}\n\n"
            f"Worker Output:\n{task['result']}\n\n"
            "INSTRUCTION: Establish the master document based on this initial data."
        )
    else:
        user_prompt = (
            f"ORIGINAL QUERY: {original_query}\n\n"
            f"--- CURRENT MASTER DOCUMENT (Steps 1 to {step_idx - 1}) ---\n"
            f"{current_synthesis}\n\n"
            f"--- NEW DATA TO INTEGRATE (Step {step_idx}/{total_steps}) ---\n"
            f"Task Addressed: {task['prompt']}\n\n"
            f"Worker Output:\n{task['result']}\n\n"
            "INSTRUCTION: Merge the NEW DATA into the CURRENT MASTER DOCUMENT. "
            "Do not throw away previous work. Expand, refine, and seamlessly integrate the new data."
        )
        
    try:
        start_time = time.time()
        response = client.chat.completions.create(
            model=ORCHESTRATOR_MODEL,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.3, max_tokens=40960
        )
        elapsed = round(time.time() - start_time, 2)
        print(f"    [+] Synthesis of Step {step_idx}/{total_steps} via {endpoint} completed in {elapsed}s.", flush=True)
        
        res_content = response.choices[0].message.content.strip()
        if response.usage and response.usage.prompt_tokens > 0:
            return res_content, response.usage.prompt_tokens, response.usage.completion_tokens
        else:
            return res_content, estimate_tokens(system_prompt + user_prompt), estimate_tokens(res_content)
            
    except Exception as e:
        print(f"    [!] Error during rolling synthesis on step {step_idx}: {e}", flush=True)
        return "", 0, 0

def execute_rolling_pipeline(sub_tasks: list, original_query: str, run_dir: Path) -> tuple:
    max_worker_concurrent = len(WORKER_ENDPOINTS) * WORKER_PARALLEL_SLOTS
    
    print(f"\n[4] 🚀 ROLLING PIPELINE: Launching parallel workers and continuous sequential synthesis...", flush=True)
    print(f"    [i] Diagnostics: {len(sub_tasks)} tasks generated. Master document will compile continuously.", flush=True)

    worker_queue = queue.Queue()
    for ep in WORKER_ENDPOINTS:
        for _ in range(WORKER_PARALLEL_SLOTS): worker_queue.put(ep)

    orch_queue = queue.Queue()
    for ep in ORCHESTRATOR_ENDPOINTS:
        for _ in range(ORCH_PARALLEL_SLOTS): orch_queue.put(ep)

    def worker_wrapper(tid: int, prompt: str):
        last_result = None
        for _ in range(WORKER_RETRIES):
            endpoint = worker_queue.get()
            try:
                result = process_subtask(tid, prompt, endpoint, original_query, run_dir)
                if result["status"] == "success": return result
                last_result = result
            except Exception as e:
                last_result = {
                    "id": tid, "status": "error", "result": f"Worker Failed: {str(e)}", 
                    "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "elapsed": 0, "tps": 0
                }
            finally:
                worker_queue.put(endpoint)
            time.sleep(2)
        return last_result

    results_dict = {}
    master_synthesis_document = ""
    total_tasks = len(sub_tasks)
    next_needed_task_id = 1
    
    worker_p_tok, worker_c_tok = 0, 0
    orch_p_tok, orch_c_tok = 0, 0
    dispatch_start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_worker_concurrent) as worker_exec:
        future_to_task = {worker_exec.submit(worker_wrapper, i + 1, task): i + 1 for i, task in enumerate(sub_tasks)}
        
        # Monitor workers as they finish asynchronously
        for future in concurrent.futures.as_completed(future_to_task):
            task_result = future.result()
            tid = task_result["id"]
            results_dict[tid] = task_result
            
            worker_p_tok += task_result["prompt_tokens"]
            worker_c_tok += task_result["completion_tokens"]
            
            current_elapsed = time.time() - dispatch_start_time
            agg_tps = round(worker_c_tok / current_elapsed, 2) if current_elapsed > 0 else 0
            
            print(f"    <- [Worker-{tid:02d}] Finished in {task_result['elapsed']}s | "
                  f"Status: {task_result['status']} | "
                  f"Agg Worker TPS: {agg_tps}", flush=True)

            # Check if the next sequential task is ready for the Master Synthesis Document
            while next_needed_task_id in results_dict:
                target_task = results_dict.pop(next_needed_task_id)
                orch_endpoint = orch_queue.get()
                
                try:
                    print(f"    [>] Pipeline trigger: Routing Step {next_needed_task_id}/{total_tasks} into Master Document...", flush=True)
                    new_synth, p_tok, c_tok = rolling_synthesis_step(
                        next_needed_task_id, total_tasks, master_synthesis_document, 
                        target_task, orch_endpoint, original_query
                    )
                    
                    if new_synth:  
                        master_synthesis_document = new_synth
                        
                    orch_p_tok += p_tok
                    orch_c_tok += c_tok
                finally:
                    orch_queue.put(orch_endpoint)
                    
                next_needed_task_id += 1
                
    return master_synthesis_document, worker_p_tok, worker_c_tok, orch_p_tok, orch_c_tok

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
    
    # 3 & 4. Execution via Rolling Pipeline (Workers + Continuous Synthesis)
    final_output, w_p, w_c, o_p, o_c = execute_rolling_pipeline(fragments, target_query, run_directory)
    
    global_input_tokens += (w_p + o_p)
    global_output_tokens += (w_c + o_c)
    
    master_elapsed_time = time.time() - master_start_time
    
    print("\n[5] 💾 MASTER EXPORT: Saving master synthesis to disk...", flush=True)
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
