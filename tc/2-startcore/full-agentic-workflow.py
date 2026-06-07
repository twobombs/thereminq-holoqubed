# This is a massive upgrade to the pipeline's capabilities. If your Qwen worker nodes are running as fully agentic instances (with tools, deep reasoning/Chain of Thought, and a 128k context window), they are no longer just simple function executors—they are autonomous sub-agents.
#To take full advantage of this, we need to alter how the orchestrator communicates with them. Instead of giving them *only* their micro-task, we should feed them the entire original query as background context so they can use that massive 128k window to see the "big picture" while they work.
#Here are the specific changes made to the engine:
# 1. **Big-Picture Context Injection:** The process_subtask function now accepts the original_query. The worker gets the full master prompt as background context, and its specific micro-task as its primary objective.
# 2. **Agentic System Prompt:** The worker's system prompt has been rewritten to explicitly authorize deep reasoning, step-by-step thought processes, and the use of external tools to accomplish their goals.
# 3. **Expanded Resource Ceilings:** Deep reasoning (like Qwen's <think> tags) and tool execution take time and consume massive amounts of tokens. I increased the worker's max_tokens from 1024 to 32768 and added a timeout of 1200 seconds (20 minutes) to prevent the network from dropping the connection while the worker is "thinking."
# 4. **Tooling Placeholder:** Added a commented-out tools=[] parameter in the worker's API call so you can easily inject your specific JSON tool schemas (e.g., web search, python interpreter) if you are passing them dynamically via the OpenAI spec.
#Here is the fully updated, complete script:
#```python
import os
import sys
import json
import time
import re
import argparse
import concurrent.futures
from datetime import datetime
from pathlib import Path
from openai import OpenAI

# ==============================================================================
# Configuration & Endpoints
# ==============================================================================

# Orchestrator Node (Handles Decomposition & Synthesis)
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://192.168.2.134:8080/v1")
ORCHESTRATOR_MODEL = os.getenv("ORCHESTRATOR_MODEL", "nvidia_Orchestrator-8B-Q6_K.gguf")
ORCH_API_KEY = os.getenv("ORCH_API_KEY", "local-sk")
MAX_RETRIES = 3

# Worker Nodes (Agentic Qwen instances with 128k context, reasoning, and tools)
WORKER_ENDPOINTS = [
    "http://192.168.2.136:8030/v1",
    "http://192.168.2.136:8031/v1",
    "http://192.168.2.136:8032/v1",
    "http://192.168.2.136:8033/v1",
    "http://192.168.2.136:8034/v1",
    "http://192.168.2.136:8035/v1"
]
WORKER_MODEL = os.getenv("WORKER_MODEL", "Qwen3.5-9B-IQ4_XS.gguf")
WORKER_API_KEY = os.getenv("WORKER_API_KEY", "local-sk")

# Setup Orchestrator Client
orch_client = OpenAI(base_url=ORCHESTRATOR_URL, api_key=ORCH_API_KEY)

# ==============================================================================
# Phase 1: Hyper-Granular Decomposition
# ==============================================================================

def extract_json_array(raw_text: str) -> str:
    """Uses regex to extract the first JSON array from a messy string."""
    cleaned_text = re.sub(r'```json\s*', '', raw_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'```\s*', '', cleaned_text)
    
    match = re.search(r'\[.*\]', cleaned_text, re.DOTALL)
    if match:
        return match.group(0)
    return ""

def decompose_to_atomic_pieces(large_query: str) -> list:
    """Forces the LLM to break a large query down into atomic micro-tasks with streaming."""
    print(f"\n[1] 📥 INGRESS: Analyzing massive query...\n    Length: {len(large_query)} characters")

    system_prompt = """You are an algorithmic micro-task decomposer.
Your sole purpose is to take a large, complex query or task and shatter it into atomic, independent pieces for parallel processing.
Output ONLY a valid, flat JSON array of strings. No markdown formatting, no conversational text.
Example: ["micro piece 1", "micro piece 2", "micro piece 3"]"""

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"[2] 🔬 DECOMPOSITION: Engaging atomic breakdown (Attempt {attempt}/{MAX_RETRIES})...")
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
            print("\n") 
            
            cleaned_output = extract_json_array(raw_output)
            if not cleaned_output:
                raise ValueError("Could not locate a JSON array in the LLM response.")
                
            atomic_pieces = json.loads(cleaned_output)
            if not isinstance(atomic_pieces, list):
                raise ValueError("LLM returned JSON, but it was not a flat array.")
                
            elapsed = round(time.time() - start_time, 2)
            print(f"    [+] Success! Shattered into {len(atomic_pieces)} distinct micro-pieces in {elapsed}s.")
            return atomic_pieces

        except Exception as e:
            print(f"\n    [!] Decomposition Error: {e}")
            if attempt < MAX_RETRIES:
                print("    [!] Retrying...")
                time.sleep(2)

    print("    [!] Fatal: Exhausted all retries. Falling back to single-task execution.")
    return [large_query]

# ==============================================================================
# Phase 2: Audit Trail / File Export
# ==============================================================================

def export_to_split_files(pieces: list):
    """Shatters the queue into individual markdown files for auditing."""
    if len(pieces) <= 1:
        return
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = Path(f"raw/decomposed_batch_{timestamp}")
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[3] 💾 QUEUE EXPORT: Saving tasks to disk for audit log...")
    
    for idx, piece in enumerate(pieces, start=1):
        filename = f"task_{idx:03d}.md"
        filepath = batch_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"{piece.strip()}\n")
            
    print(f"    [+] Saved {len(pieces)} files to {batch_dir.absolute()}/")

# ==============================================================================
# Phase 3: Parallel Dispatch (Agentic Qwen Workers)
# ==============================================================================

def process_subtask(task_id: int, task_prompt: str, endpoint: str, original_query: str) -> dict:
    """Worker thread function to execute an agentic sub-task with full context."""
    print(f"    -> [Thread-{task_id:02d}] Dispatched to {endpoint} | Task: '{task_prompt[:40]}...'")
    
    worker_client = OpenAI(base_url=endpoint, api_key=WORKER_API_KEY)
    start_time = time.time()
    
    # We now feed the worker the big-picture context, taking advantage of the 128k window
    system_instruction = (
        "You are an autonomous, highly-capable worker agent. "
        "You are equipped with advanced reasoning capabilities and external tools. "
        "Think step-by-step to formulate a plan. If you need to do additional work or research, use your tools. "
        "Provide a comprehensive, highly-detailed execution of your specific objective based on the broader context."
    )
    
    user_instruction = (
        f"BACKGROUND CONTEXT (The overall project):\n{original_query}\n\n"
        f"YOUR SPECIFIC OBJECTIVE:\n{task_prompt}"
    )
    
    try:
        response = worker_client.chat.completions.create(
            model=WORKER_MODEL,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_instruction}
            ],
            temperature=0.4,
            max_tokens=32768, # Massively increased to allow for deep reasoning/Chain of Thought output
            timeout=1200.0,   # 20 minute timeout to allow for lengthy tool usage and long-context processing
            # tools=[],       # Uncomment and add your JSON tool schemas here if required by your endpoint setup
        )
        result_text = response.choices[0].message.content.strip()
        status = "success"
        
        if len(result_text) < 20:
            status = "failed_validation (output too short)"
            
    except Exception as e:
        result_text = f"Worker Error: {str(e)}"
        status = "error"

    elapsed = round(time.time() - start_time, 2)
    print(f"    <- [Thread-{task_id:02d}] Completed in {elapsed}s | Status: {status}")
    
    return {
        "id": task_id,
        "prompt": task_prompt,
        "result": result_text,
        "status": status
    }

def dispatch_and_gather(sub_tasks: list, original_query: str) -> list:
    """Distributes tasks across the worker pool using concurrent threads."""
    print(f"\n[4] 🚀 DISPATCH: Firing agentic tasks across Worker Pool...")
    
    results = []
    tasks_with_endpoints = []
    for i, task in enumerate(sub_tasks):
        endpoint = WORKER_ENDPOINTS[i % len(WORKER_ENDPOINTS)]
        tasks_with_endpoints.append((i + 1, task, endpoint))

    # Pass the original_query into the threads so workers have 128k context awareness
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(WORKER_ENDPOINTS)) as executor:
        future_to_task = {
            executor.submit(process_subtask, tid, prompt, ep, original_query): tid 
            for (tid, prompt, ep) in tasks_with_endpoints
        }
        
        for future in concurrent.futures.as_completed(future_to_task):
            try:
                task_result = future.result()
                results.append(task_result)
            except Exception as exc:
                print(f"    [!] Thread generated an exception: {exc}")

    results.sort(key=lambda x: x["id"])
    return results

# ==============================================================================
# Phase 4: Synthesis
# ==============================================================================

def synthesize_results(original_query: str, completed_tasks: list) -> str:
    """Takes all worker outputs and synthesizes the final comprehensive answer."""
    print(f"\n[5] 🧠 SYNTHESIS: Consolidating worker progress into final output...")
    
    context_blocks = []
    for t in completed_tasks:
        context_blocks.append(f"--- Sub-Task: {t['prompt']} ---\nSTATUS: {t['status']}\nRESULT:\n{t['result']}\n")
    
    consolidated_context = "\n".join(context_blocks)
    
    system_prompt = """You are the Synthesis Layer of a master AI orchestrator.
Read the original user query and the compiled reports from multiple autonomous worker nodes.
Merge these separate reports into a single, cohesive, highly-detailed final response.
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
            max_tokens=8192 # Increased to handle potentially massive worker outputs
        )
        
        final_answer = response.choices[0].message.content.strip()
        print("    [+] Synthesis complete.")
        return final_answer
        
    except Exception as e:
        print(f"    [!] Error during synthesis: {e}")
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
            print(f"[!] Fatal Error: The file '{args.file}' does not exist.")
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8") as f:
            target_query = f.read()
        print(f"[*] Loaded query from file: {args.file}")
    elif args.prompt:
        target_query = args.prompt
        print("[*] Loaded query from command line argument.")
    else:
        print("[*] No input arguments provided. Using default complex query.")
        target_query = """
        Build a complete, secure, production-ready React and Node.js e-commerce application. 
        It needs a PostgreSQL database, user authentication via JWT, a product catalog with 
        search and filtering, a shopping cart, Stripe payment integration, order history, 
        and an admin dashboard to manage inventory. Write all the code, setup instructions, 
        and deployment scripts using Docker and AWS ECS.
        """
        
    print("\n=== STARTING UNIFIED ORCHESTRATOR ENGINE ===")
    
    # 1. Break the massive query into atomic pieces
    fragments = decompose_to_atomic_pieces(target_query)
    
    # 2. Write them to disk for logging/auditing
    export_to_split_files(fragments)
    
    # 3. Fire them off to the Qwen worker nodes (now passing the full original query for context)
    worker_results = dispatch_and_gather(fragments, target_query)
    
    # 4. Synthesize the final result via Nemotron
    final_output = synthesize_results(target_query, worker_results)
    
    print("\n==============================================================================")
    print("✨ FINAL SYNTHESIZED OUTPUT ✨\n")
    print(final_output)
    print("\n==============================================================================")

