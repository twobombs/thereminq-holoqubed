import os
import sys
import json
import time
import concurrent.futures
from openai import OpenAI

# ==============================================================================
# Configuration & Endpoints
# ==============================================================================

# Orchestrator Node (Handles Decomposition & Synthesis)
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://127.0.0.1:8080/v1")
ORCHESTRATOR_MODEL = os.getenv("ORCHESTRATOR_MODEL", "nemotron-orchestrator-8b")

# Worker Nodes (Handles the parallel sub-tasks)
# In a real cluster, these might point to different IPs or ports
WORKER_ENDPOINTS = [
    "http://127.0.0.1:8081/v1",
    "http://127.0.0.1:8082/v1",
    "http://127.0.0.1:8083/v1"
]
WORKER_MODEL = os.getenv("WORKER_MODEL", "qwen-coder")

# Setup Orchestrator Client
orch_client = OpenAI(base_url=ORCHESTRATOR_URL, api_key="local-sk")

# ==============================================================================
# Phase 1: Ingress & Decomposition
# ==============================================================================

def decompose_query(user_query: str) -> list:
    """Asks the orchestrator to break down the query into smaller parallel tasks."""
    print(f"\n[1] 📥 INGRESS: Received complex query:\n    '{user_query}'")
    print(f"[2] 🪚 DECOMPOSITION: Orchestrator is breaking tasks down...")
    
    system_prompt = """You are the Decomposition Layer of an AI orchestrator.
Analyze the user's query and break it down into 3 to 5 independent, highly specific sub-tasks that can be researched or processed in parallel by worker nodes.
Output ONLY a raw JSON list of strings, where each string is a specific prompt for a worker.
Example format: ["Research aspect A", "Write code for B", "Analyze impact of C"]"""

    try:
        response = orch_client.chat.completions.create(
            model=ORCHESTRATOR_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            temperature=0.2,
            response_format={"type": "json_object"} # Force JSON if supported, otherwise rely on prompt
        )
        
        raw_output = response.choices[0].message.content.strip()
        
        # Clean up potential markdown formatting from LLM
        if raw_output.startswith("```json"):
            raw_output = raw_output[7:-3].strip()
        elif raw_output.startswith("```"):
            raw_output = raw_output[3:-3].strip()
            
        sub_tasks = json.loads(raw_output)
        
        # Ensure it's a list
        if isinstance(sub_tasks, dict) and len(sub_tasks.keys()) == 1:
            sub_tasks = list(sub_tasks.values())[0]
            
        print(f"    [+] Successfully decomposed into {len(sub_tasks)} parallel threads.")
        return sub_tasks

    except Exception as e:
        print(f"    [!] Error during decomposition: {e}")
        print("    [!] Falling back to single-task execution.")
        return [user_query]

# ==============================================================================
# Phase 2 & 3: Dispatch, Wait, & Check
# ==============================================================================

def process_subtask(task_id: int, task_prompt: str, endpoint: str) -> dict:
    """Worker thread function to execute a sub-task."""
    print(f"    -> [Thread-{task_id}] Dispatched to {endpoint} | Task: '{task_prompt[:50]}...'")
    
    worker_client = OpenAI(base_url=endpoint, api_key="local-sk")
    start_time = time.time()
    
    try:
        response = worker_client.chat.completions.create(
            model=WORKER_MODEL,
            messages=[
                {"role": "system", "content": "You are a specialized worker node. Provide a detailed, accurate response to the sub-task. Focus only on your specific task."},
                {"role": "user", "content": task_prompt}
            ],
            temperature=0.4,
            max_tokens=1024
        )
        result_text = response.choices[0].message.content.strip()
        status = "success"
        
        # Simple Check/Validation layer: Ensure the worker actually returned substance
        if len(result_text) < 20:
            status = "failed_validation (output too short)"
            
    except Exception as e:
        result_text = f"Worker Error: {str(e)}"
        status = "error"

    elapsed = round(time.time() - start_time, 2)
    print(f"    <- [Thread-{task_id}] Completed in {elapsed}s | Status: {status}")
    
    return {
        "id": task_id,
        "prompt": task_prompt,
        "result": result_text,
        "status": status
    }

def dispatch_and_gather(sub_tasks: list) -> list:
    """Distributes tasks across the worker pool using concurrent threads."""
    print(f"\n[3] 🚀 DISPATCH: Sending tasks to Worker Nodes...")
    
    results = []
    # Map tasks to available endpoints via simple round-robin
    tasks_with_endpoints = []
    for i, task in enumerate(sub_tasks):
        endpoint = WORKER_ENDPOINTS[i % len(WORKER_ENDPOINTS)]
        tasks_with_endpoints.append((i, task, endpoint))

    # Execute in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(WORKER_ENDPOINTS)) as executor:
        future_to_task = {
            executor.submit(process_subtask, tid, prompt, ep): tid 
            for (tid, prompt, ep) in tasks_with_endpoints
        }
        
        for future in concurrent.futures.as_completed(future_to_task):
            try:
                task_result = future.result()
                results.append(task_result)
            except Exception as exc:
                print(f"    [!] Thread generated an exception: {exc}")

    # Sort results back into their original decomposed order for logical flow
    results.sort(key=lambda x: x["id"])
    return results

# ==============================================================================
# Phase 4: Synthesis / Channel Output
# ==============================================================================

def synthesize_results(original_query: str, completed_tasks: list) -> str:
    """Takes all worker outputs and synthesizes the final comprehensive answer."""
    print(f"\n[4] 🧠 SYNTHESIS: Consolidating worker progress into final output...")
    
    # Build the context block from worker results
    context_blocks = []
    for t in completed_tasks:
        context_blocks.append(f"--- Sub-Task: {t['prompt']} ---\nSTATUS: {t['status']}\nRESULT:\n{t['result']}\n")
    
    consolidated_context = "\n".join(context_blocks)
    
    system_prompt = """You are the Synthesis Layer of a master AI orchestrator.
Your job is to read the original user query and the compiled reports from multiple worker nodes.
Merge these separate reports into a single, cohesive, well-formatted final response.
Resolve any contradictions between the workers smoothly, remove redundancies, and directly answer the user's original query."""

    user_prompt = f"""ORIGINAL QUERY: {original_query}
