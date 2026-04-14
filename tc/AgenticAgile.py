import os
import google.generativeai as genai
from datetime import datetime

# =====================================================================
# Configuration & Directory Setup
# =====================================================================
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set. Please set it to run the script.")

genai.configure(api_key=API_KEY)

# Define architecture directories
RAW_DIR = "raw"
WIKI_DIR = "wiki"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(WIKI_DIR, exist_ok=True)

# Define file paths
RAW_INPUT_FILE = os.path.join(RAW_DIR, "slack_transcript_latest.txt")
WIKI_TASKS_FILE = os.path.join(WIKI_DIR, "ACTIVE_TASKS.md")
WIKI_INDEX_FILE = os.path.join(WIKI_DIR, "index.md")
WIKI_LOG_FILE = os.path.join(WIKI_DIR, "log.md")
WIKI_DOD_FILE = os.path.join(WIKI_DIR, "DEFINITION_OF_DONE.md")
WIKI_SYNTHESIS_FILE = os.path.join(WIKI_DIR, "DAILY_SYNTHESIS.md")

# =====================================================================
# Mock Data Generation (Immutable Raw Sources & Wiki Rules)
# =====================================================================
def setup_mock_environment():
    """Creates raw sources and project rules for the agent to process."""
    if not os.path.exists(RAW_INPUT_FILE):
        mock_raw = """
        [10:05 AM] Sarah (Frontend): I pushed the new React dashboard components. 
        [10:06 AM] John (Backend): The user data API endpoint changed from /api/v1/users to /api/v2/users.
        [10:08 AM] Sarah: I'll update the fetch calls, but I haven't written the unit tests for the new components yet. Might push them tomorrow to save time.
        """
        with open(RAW_INPUT_FILE, "w") as f:
            f.write(mock_raw)

    if not os.path.exists(WIKI_DOD_FILE):
        mock_dod = """
        # Definition of Done (DoD)
        1. All API endpoints must be strictly versioned (e.g., v2).
        2. All new React components MUST have accompanying Jest unit tests before merging.
        3. No code is deployed on Fridays.
        """
        with open(WIKI_DOD_FILE, "w") as f:
            f.write(mock_dod)

# =====================================================================
# Core Agent Operations
# =====================================================================
def ingest_source_to_wiki(input_path, source_title):
    """Step 1 & 2: Continuous Ingest and Structuring the Wiki."""
    print(f"[*] INGESTING: {source_title}...")
    
    with open(input_path, "r") as f:
        raw_content = f.read()

    model = genai.GenerativeModel('gemini-1.5-pro')

    # Generate Task Content
    content_prompt = f"""
    Analyze the raw transcript and extract actionable tasks, architectural decisions, and blockers.
    Output strictly as Markdown suitable for 'ACTIVE_TASKS.md'. Do not use markdown codeblocks.
    Raw Transcript: {raw_content}
    """
    tasks_response = model.generate_content(content_prompt)
    
    # Generate Index Entry
    index_prompt = f"""
    Write a single-line summary of the transcript for the index.
    Format: - [[ACTIVE_TASKS.md]] - [One sentence summary]
    Raw Transcript: {raw_content}
    """
    index_response = model.generate_content(index_prompt)

    # Apply Updates
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_date = datetime.now().strftime("%Y-%m-%d")

    with open(WIKI_TASKS_FILE, "a") as f:
        f.write(f"\n\n## Auto-Update: {current_time}\n")
        f.write(tasks_response.text.strip())

    index_exists = os.path.exists(WIKI_INDEX_FILE) and os.path.getsize(WIKI_INDEX_FILE) > 0
    with open(WIKI_INDEX_FILE, "a") as f:
        if not index_exists:
            f.write("# Wiki Index\n\n### Core Project Files\n")
        f.write(f"{index_response.text.strip()}\n")

    with open(WIKI_LOG_FILE, "a") as f:
        f.write(f"## [{current_date}] ingest | {source_title}\n")

    print("[+] Ingest complete: Wiki state updated.")

def agentic_linting():
    """Step 3: Agentic Linting & Quality Control."""
    print("[*] LINTING: Checking Wiki against Definition of Done...")
    
    try:
        with open(WIKI_TASKS_FILE, "r") as f:
            active_tasks = f.read()
        with open(WIKI_DOD_FILE, "r") as f:
            dod_rules = f.read()
    except FileNotFoundError:
        print("[-] Skipping linting: Required files missing.")
        return

    model = genai.GenerativeModel('gemini-1.5-pro')
    lint_prompt = f"""
    You are the Agentic Linter. Compare the ACTIVE TASKS against the DEFINITION OF DONE.
    Identify any contradictions, policy violations, or quality risks.
    
    ACTIVE TASKS:
    {active_tasks}
    
    DEFINITION OF DONE:
    {dod_rules}
    
    Output a markdown report starting with "### Linting Report". 
    If a violation is found, clearly state the contradiction (e.g., "CONTRADICTION: Frontend tests missing vs DoD Rule 2").
    Do not use markdown codeblocks.
    """
    
    lint_response = model.generate_content(lint_prompt)
    
    with open(WIKI_TASKS_FILE, "a") as f:
        f.write(f"\n\n{lint_response.text.strip()}\n")
        
    print("[+] Linting complete: Contradictions and quality checks appended to tasks.")

def generate_daily_synthesis():
    """Steps 5 & 6: Generating Actionable Insights and Daily Synthesis."""
    print("[*] SYNTHESIZING: Generating Automated Daily Synthesis...")
    
    try:
        with open(WIKI_TASKS_FILE, "r") as f:
            active_tasks = f.read()
    except FileNotFoundError:
        print("[-] Skipping synthesis: Active tasks missing.")
        return

    model = genai.GenerativeModel('gemini-1.5-pro')
    synthesis_prompt = f"""
    You are the Global Agile Orchestrator. Review the current project state and generate an Automated Daily Synthesis.
    
    PROJECT STATE (Active Tasks & Linting Reports):
    {active_tasks}
    
    Format the output exactly with these markdown headers:
    ## Daily Synthesis: {datetime.now().strftime("%Y-%m-%d")}
    ### 📊 Daily Progress
    ### 🎯 Current Focus
    ### ⚠️ Risk Mitigation & Blockers
    ### 🚀 Velocity Suggestion (Analyze current pace and suggest adjustments)
    
    Do not use markdown codeblocks. Keep it concise, actionable, and aligned for a team sync.
    """
    
    synthesis_response = model.generate_content(synthesis_prompt)
    
    # Overwrite or prepend to keep the synthesis file acting as a dashboard
    with open(WIKI_SYNTHESIS_FILE, "w") as f:
        f.write(synthesis_response.text.strip())
        
    # Also log the synthesis generation
    with open(WIKI_LOG_FILE, "a") as f:
        f.write(f"## [{datetime.now().strftime('%Y-%m-%d')}] synthesis | Generated Daily Agile Report\n")

    print(f"[+] Synthesis complete: Dashboard updated at {WIKI_SYNTHESIS_FILE}")

# =====================================================================
# Main Execution Pipeline
# =====================================================================
if __name__ == "__main__":
    print("=== STARTING AGENTIC PROJECT ORCHESTRATION ===")
    
    # 0. Set up initial environment
    setup_mock_environment()
    
    # 1 & 2. Ingest raw data into the Living Wiki
    ingest_source_to_wiki(
        input_path=RAW_INPUT_FILE, 
        source_title="Slack Transcript: Sprint 4 Sync"
    )
    
    # 3. Run Quality Control & Linting
    agentic_linting()
    
    # 4 & 5 & 6. Generate Agile Insights and Daily Dashboard
    generate_daily_synthesis()
    
    print("=== ORCHESTRATION PIPELINE FINISHED ===")
