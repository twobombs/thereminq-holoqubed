import os
import time
import json
import queue
import requests
import threading
import concurrent.futures
from datetime import datetime

# ==============================================================================
# Configuration & Directory Setup
# ==============================================================================

RAW_DIR = "raw"
WIKI_DIR = "wiki"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(WIKI_DIR, exist_ok=True)

RAW_INPUT_FILE = os.path.join(RAW_DIR, "slack_transcript_latest.txt")
WIKI_DOD_FILE = os.path.join(WIKI_DIR, "DEFINITION_OF_DONE.md")
STATE_FILE = os.path.join(WIKI_DIR, "project_state.json")
WIKI_SYNTHESIS_FILE = os.path.join(WIKI_DIR, "DAILY_SYNTHESIS.md")

VALID_STATES = {"active", "in_progress", "blocked", "completed", "invalid"}

# ==============================================================================
# State Management
# ==============================================================================

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {
        "tasks": {},
        "linting_violations": [],
        "failed_chunks": [],     
        "last_updated": None
    }

def save_state(state):
    state["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)

# ==============================================================================
# LLM Cluster & Session Management (LIVE)
# ==============================================================================

WORKER_ENDPOINTS = ["http://localhost:8033/v1/chat/completions"]
ORCHESTRATOR_ENDPOINT = "http://192.168.2.134:8033/v1/chat/completions"
MAX_SESSIONS_PER_ENDPOINT = 2

class LLMClusterManager:
    def __init__(self):
        self.worker_pool = queue.Queue()
        for endpoint in WORKER_ENDPOINTS:
            for _ in range(MAX_SESSIONS_PER_ENDPOINT):
                self.worker_pool.put(endpoint)

    def query(self, prompt, system_prompt="", is_orchestrator=False, max_retries=3, requires_json=False):
        endpoint = ORCHESTRATOR_ENDPOINT if is_orchestrator else self.worker_pool.get()
        
        payload = {
            "model": "local-model",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2 if requires_json else 0.7
        }

        for attempt in range(max_retries):
            try:
                print(f"      -> [LLM Cluster] Sending Request to: {endpoint}...")
                
                response = requests.post(endpoint, json=payload, timeout=120)
                response.raise_for_status()
                
                result = response.json()['choices'][0]['message']['content'].strip()
                
                if not is_orchestrator:
                    self.worker_pool.put(endpoint)
                return True, result
                
            except Exception as e:
                print(f"      -> [LLM Cluster] Node {endpoint} failed: {e}. Retrying...")
                time.sleep(2 ** attempt)
        
        if not is_orchestrator:
            self.worker_pool.put(endpoint)
        return False, None

cluster = LLMClusterManager()

# ==============================================================================
# Micro-Task Dispatcher
# ==============================================================================

def chunk_text(text, chunk_size=6, overlap=1):
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    chunks, i = [], 0
    while i < len(lines):
        chunks.append('\n'.join(lines[i:i + chunk_size]))
        i += (chunk_size - overlap)
        if i >= len(lines): break
    return chunks

def dispatch_jobs_in_chunks(large_text, prompt_template, system_prompt=""):
    chunks = chunk_text(large_text)
    print(f"      -> [Dispatcher] Processing {len(chunks)} chunks in parallel.")
    
    results, failed_chunks = [None] * len(chunks), []
    total_capacity = len(WORKER_ENDPOINTS) * MAX_SESSIONS_PER_ENDPOINT
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=total_capacity) as executor:
        future_to_index = {
            executor.submit(cluster.query, prompt_template.format(chunk=chunk), system_prompt): i 
            for i, chunk in enumerate(chunks)
        }
        
        for future in concurrent.futures.as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                success, output = future.result()
                if success:
                    results[idx] = output
                else:
                    failed_chunks.append({"type": "extraction_failure", "payload": chunks[idx]})
            except Exception as exc:
                failed_chunks.append({"type": "exception", "error": str(exc)})
                
    return [r for r in results if r is not None], failed_chunks

# ==============================================================================
# Core Operations
# ==============================================================================

def setup_mock_environment():
    """Only creates files if they are missing."""
    if not os.path.exists(RAW_INPUT_FILE):
        mock_raw = "[10:00] User A: Push to prod. [10:05] User B: Wait, tests failed."
        with open(RAW_INPUT_FILE, "w", encoding="utf-8") as f:
            f.write(mock_raw)

def ingest_and_merge_source(input_path, source_title):
    print(f"\n[*] INGESTING: {source_title}...")
    with open(input_path, "r", encoding="utf-8") as f:
        raw_content = f.read()

    content_prompt = "Extract bullet points of actionable tasks from this transcript segment:\n{chunk}"
    raw_tasks_list, extraction_failures = dispatch_jobs_in_chunks(raw_content, content_prompt)

    state = load_state()
    if extraction_failures:
        state["failed_chunks"].extend(extraction_failures)

    print("\n[*] MERGING: Orchestrator reconciling state...")
    
    existing_tasks_context = json.dumps(list(state["tasks"].values()), indent=2)
    new_signals_context = "\n".join(raw_tasks_list)

    merge_prompt = f"""
Reconcile these new signals with the current project state. 
Return a JSON array of objects with keys: id (slug), content, status, confidence.
STATE: {existing_tasks_context}
SIGNALS: {new_signals_context}
"""
    success, merged_output = cluster.query(
        prompt=merge_prompt,
        system_prompt="Return ONLY a JSON array.",
        is_orchestrator=True,
        requires_json=True
    )

    if success:
        try:
            # Clean up potential markdown code blocks in LLM output
            clean_json = re.sub(r'```json|```', '', merged_output).strip()
            merged_tasks = json.loads(clean_json)
            for task in merged_tasks:
                slug = task.get("id")
                if slug:
                    task["updated_at"] = datetime.now().isoformat()
                    state["tasks"][slug] = task
            print(f" [+] Reconciliation complete.")
        except Exception as e:
            print(f" [!] JSON Parse Error: {e}")

    save_state(state)

def generate_daily_synthesis():
    print("\n[*] SYNTHESIZING: Generating Daily Synthesis...")
    state = load_state()
    synthesis_prompt = f"Summarize this project state into a readable markdown report:\n{json.dumps(state['tasks'])}"
    success, response = cluster.query(synthesis_prompt, is_orchestrator=True)
    if success:
        with open(WIKI_SYNTHESIS_FILE, "w", encoding="utf-8") as f:
            f.write(response)
        print(f" [+] Synthesis saved to {WIKI_SYNTHESIS_FILE}")

import re # Needed for JSON cleaning

if __name__ == "__main__":
    print("=== STARTING LIVE AGENTIC CONTROL LOOP ===")
    setup_mock_environment()
    ingest_and_merge_source(RAW_INPUT_FILE, "Latest Sync")
    generate_daily_synthesis()
    print("\n=== PIPELINE FINISHED ===")
