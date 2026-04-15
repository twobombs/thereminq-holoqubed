import os
import time
import json
import queue
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

# Valid lifecycle states
VALID_STATES = {"active", "in_progress", "blocked", "completed", "invalid"}

# ==============================================================================
# State Management
# ==============================================================================

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {
        "tasks": {},             # { semantic_slug: { id, content, status, confidence, ... } }
        "linting_violations": [],
        "failed_chunks": [],     
        "last_updated": None
    }

def save_state(state):
    state["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)

# ==============================================================================
# LLM Cluster & Session Management 
# ==============================================================================

WORKER_ENDPOINTS = [f"http://worker-node-{i}:8080/v1/chat/completions" for i in range(1, 8)]
ORCHESTRATOR_ENDPOINT = "http://orchestrator-node:8080/v1/chat/completions"
MAX_SESSIONS_PER_ENDPOINT = 2

class LLMClusterManager:
    def __init__(self):
        self.worker_pool = queue.Queue()
        for endpoint in WORKER_ENDPOINTS:
            for _ in range(MAX_SESSIONS_PER_ENDPOINT):
                self.worker_pool.put(endpoint)

    def query(self, prompt, system_prompt="", is_orchestrator=False, max_retries=3, requires_json=False):
        endpoint = ORCHESTRATOR_ENDPOINT if is_orchestrator else self.worker_pool.get()

        for attempt in range(max_retries):
            try:
                print(f"      -> [LLM Cluster] Routing to: {endpoint} | Attempt {attempt+1}/{max_retries}")
                
                # Mocking the actual HTTP inference delay
                time.sleep(1) 
                
                # Mock Responses
                if is_orchestrator and requires_json:
                    result = """[
                        {"id": "update_api_v2", "content": "Update user API endpoints to v2", "status": "active", "confidence": 0.95},
                        {"id": "jest_tests_dashboard", "content": "Write Jest tests for React dashboard", "status": "blocked", "confidence": 0.88}
                    ]"""
                elif is_orchestrator:
                    result = "## Daily Synthesis\n### Progress\nAPI Updates planned.\n### Risk\nMissing tests."
                else:
                    result = "- Extracted raw context: Frontend pushed, backend changed API, tests missing."
                
                if not is_orchestrator:
                    self.worker_pool.put(endpoint)
                return True, result
                
            except Exception as e:
                print(f"      -> [LLM Cluster] Node {endpoint} failed: {e}. Backing off...")
                time.sleep(2 ** attempt)
        
        if not extraction_mode:
            self.worker_pool.put(endpoint)
        return False, prompt

cluster = LLMClusterManager()

# ==============================================================================
# Micro-Task Dispatcher
# ==============================================================================

def chunk_text(text, chunk_size=4, overlap=1):
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    chunks, i = [], 0
    while i < len(lines):
        chunks.append('\n'.join(lines[i:i + chunk_size]))
        i += (chunk_size - overlap)
        if i >= len(lines): break
    return chunks

def dispatch_jobs_in_chunks(large_text, prompt_template, system_prompt="", chunk_size=4, overlap=1):
    chunks = chunk_text(large_text, chunk_size, overlap)
    print(f"      -> [Dispatcher] Split job into {len(chunks)} sliding-window micro-tasks.")
    
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
                    failed_chunks.append({"type": "extraction_failure", "payload": chunks[idx], "timestamp": datetime.now().isoformat()})
            except Exception as exc:
                failed_chunks.append({"type": "unhandled_exception", "payload": chunks[idx], "error": str(exc)})
                
    return [r for r in results if r is not None], failed_chunks

# ==============================================================================
# Mock Data Generation
# ==============================================================================

def setup_mock_environment():
    if not os.path.exists(RAW_INPUT_FILE):
        mock_raw = """
[10:05 AM] Sarah (Frontend): I pushed the new React dashboard components.
[10:06 AM] John (Backend): The user data API endpoint changed from /api/v1/users to /api/v2/users.
[10:08 AM] Sarah: I'll update the fetch calls, but I haven't written the unit tests for the new components yet. Might push them tomorrow to save time.
[10:15 AM] Mike (DevOps): Reminder, pipeline is frozen this Friday for infrastructure updates.
[10:20 AM] Sarah: Okay, I will make sure the dashboard is merged by Thursday EOD.
[10:25 AM] John: I'll review your PR once the tests are in.
"""
        with open(RAW_INPUT_FILE, "w") as f:
            f.write(mock_raw.strip())
    
    if not os.path.exists(WIKI_DOD_FILE):
        mock_dod = "1. API endpoints must be versioned.\n2. React components need Jest tests.\n3. No deployments on Friday."
        with open(WIKI_DOD_FILE, "w") as f:
            f.write(mock_dod)

# ==============================================================================
# Core Agent Operations
# ==============================================================================

def ingest_and_merge_source(input_path, source_title):
    print(f"\n[*] INGESTING: {source_title}...")
    with open(input_path, "r") as f:
        raw_content = f.read()

    # Phase 1: Distributed Extraction
    content_prompt = "Extract tasks and blockers.\nContent:\n---\n{chunk}\n---"
    raw_tasks_list, extraction_failures = dispatch_jobs_in_chunks(raw_content, content_prompt, chunk_size=4, overlap=1)

    state = load_state()
    if extraction_failures:
        state["failed_chunks"].extend(extraction_failures)

    # Phase 2: Context-Aware Orchestrator Merge (Semantic Identity & Lifecycle)
    print("\n[*] MERGING: Orchestrator is reconciling new signals with existing state...")
    
    existing_tasks_context = json.dumps(list(state["tasks"].values()), indent=2)
    new_signals_context = chr(10).join(raw_tasks_list)

    merge_prompt = f"""
You are the Global Orchestrator. Reconcile NEW SIGNALS with the EXISTING STATE.
Generate a semantic slug for identity (e.g. 'update_dashboard_api').
Update statuses based on signals (active, in_progress, blocked, completed, invalid).
Calculate a confidence score (0.0-1.0) based on signal repetition.

EXISTING STATE:
{existing_tasks_context}

NEW SIGNALS:
{new_signals_context}

Output ONLY a valid JSON array of objects with keys: id (slug), content, status, confidence.
"""
    success, merged_output = cluster.query(
        prompt=merge_prompt,
        system_prompt="Return strict JSON.",
        is_orchestrator=True,
        requires_json=True
    )

    if not success:
        print(" [!] FATAL: Orchestrator failed to merge. Adding to dead-letter queue.")
        state["failed_chunks"].append({"type": "orchestrator_merge", "payload": new_signals_context})
        save_state(state)
        return

    # Phase 3: Update State Machine
    try:
        merged_tasks = json.loads(merged_output)
        for task in merged_tasks:
            slug = task.get("id")
            if not slug or task.get("status") not in VALID_STATES:
                continue
                
            task["updated_at"] = datetime.now().isoformat()
            
            # Confidence Propagation / State Update
            if slug in state["tasks"]:
                old_task = state["tasks"][slug]
                task["confidence"] = min(1.0, old_task["confidence"] + 0.05) # Boost confidence on reinforcement
                task["source"] = old_task.get("source", "") + f", {source_title}"
            else:
                task["source"] = source_title
                task["created_at"] = datetime.now().isoformat()
                
            state["tasks"][slug] = task
            
        print(f" [+] Reconciliation complete: {len(merged_tasks)} canonical tasks tracked.")
    except json.JSONDecodeError:
        print(" [!] Orchestrator produced invalid JSON. Sending to dead-letter queue.")
        state["failed_chunks"].append({"type": "invalid_json_merge", "payload": merged_output})

    save_state(state)

def active_dead_letter_recovery():
    print("\n[*] SELF-HEALING: Processing Dead-Letter Queue...")
    state = load_state()
    
    if not state["failed_chunks"]:
        print(" [+] Queue is empty. System is healthy.")
        return
        
    print(f" [!] Attempting to recover {len(state['failed_chunks'])} failed operations...")
    still_failed = []
    
    for failure in state["failed_chunks"]:
        # Fallback Strategy: Simplify prompt and increase context window
        if failure["type"] == "extraction_failure":
            recovery_prompt = f"Summarize any technical tasks found here:\n{failure['payload']}"
            success, output = cluster.query(recovery_prompt, max_retries=1) # Single desperate attempt
            
            if success:
                print("   [+] Successfully recovered an extraction failure.")
                # In a real system, we would route this recovered output back into the merge phase
            else:
                still_failed.append(failure)
        else:
            still_failed.append(failure) # Keep non-extraction errors in queue for manual review

    state["failed_chunks"] = still_failed
    save_state(state)

def generate_daily_synthesis():
    print("\n[*] SYNTHESIZING: Generating Automated Daily Synthesis...")
    state = load_state()

    active_tasks = [t for t in state["tasks"].values() if t["status"] in ("active", "in_progress", "blocked")]
    if not active_tasks:
        return

    synthesis_payload = {
        "tracked_tasks": active_tasks,
        "linting_violations": state["linting_violations"],
        "system_health_alerts": len(state['failed_chunks'])
    }

    synthesis_prompt = f"Review project state and output a markdown Daily Synthesis.\n{json.dumps(synthesis_payload)}"
    success, response = cluster.query(synthesis_prompt, is_orchestrator=True)

    if success:
        with open(WIKI_SYNTHESIS_FILE, "w") as f:
            f.write(response)
        print(" [+] Synthesis complete: Dashboard updated.")

# ==============================================================================
# Main Execution Pipeline
# ==============================================================================

if __name__ == "__main__":
    print("=== STARTING AUTONOMOUS AGENTIC CONTROL LOOP ===")
    setup_mock_environment()
    
    # 1. Ingest, Extract, Reconcile State
    ingest_and_merge_source(RAW_INPUT_FILE, "Sprint 4 Sync")
    
    # 2. Self-Healing Phase
    active_dead_letter_recovery()
    
    # 3. Report Generation
    generate_daily_synthesis()
    
    print("\n=== PIPELINE FINISHED ===")
