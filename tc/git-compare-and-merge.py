import os
import sys
import subprocess
import openai
import re

# --- 1. CONFIGURATION ---
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://192.168.2.134:8033/v1")
REASONING_URL = os.getenv("REASONING_URL", "http://192.168.2.137:8033/v1")
CLIENT_TIMEOUT = (10.0, 600.0)

# Client Setup
orch_client = openai.OpenAI(
    base_url=ORCHESTRATOR_URL, 
    api_key="not-needed",
    timeout=CLIENT_TIMEOUT
)
reason_client = openai.OpenAI(
    base_url=REASONING_URL, 
    api_key="not-needed",
    timeout=CLIENT_TIMEOUT
)

# --- 2. GIT HELPER FUNCTIONS ---
def run_git_command(args, check=True):
    """Executes a git command and returns the output."""
    try:
        result = subprocess.run(
            ["git"] + args,
            text=True,
            capture_output=True,
            check=check
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        if check:
            print(f"❌ [Git Error] Failed running 'git {' '.join(args)}'")
            print(f"Error Output: {e.stderr.strip()}")
            sys.exit(1)
        return e.stdout.strip()

def get_current_branch():
    return run_git_command(["rev-parse", "--abbrev-ref", "HEAD"])

def get_conflicting_files():
    """Finds all files currently in a conflicted state."""
    # diff-filter=U gets unmerged files
    output = run_git_command(["diff", "--name-only", "--diff-filter=U"])
    if not output:
        return []
    return output.split('\n')

# --- 3. THE AI RESOLUTION ENGINE ---
def clean_llm_output(text):
    """Strips markdown code blocks to prevent corrupting the source files."""
    text = text.strip()
    # Remove standard markdown code blocks
    text = re.sub(r'^```[\w]*\n', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n```$', '', text, flags=re.MULTILINE)
    return text

def resolve_file_with_ai(file_path):
    """Reads a conflicted file, sends it to the Reasoner and Orchestrator, and writes the fix."""
    print(f"\n📄 Processing conflict in: {file_path}")
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"⚠️ Could not read {file_path}: {e}")
        return False

    if "<<<<<<<" not in content or "=======" not in content:
        print(f"💡 No standard conflict markers found in {file_path}. Skipping.")
        return False

    # STEP A: Reasoner Drafts the Fix
    print("   🧠 [Reasoner] Analyzing markers and drafting merge strategy...")
    reasoner_prompt = f"""
    You are an expert developer. The following file has a git merge conflict.
    Resolve the conflict intelligently. Consolidate documentation, keep the best code features from both branches, or fix breaking changes between them.
    
    FILE CONTENT:
    {content}
    
    INSTRUCTIONS: Output ONLY the fully resolved, merged file content. Do not include markdown code blocks. Do not explain your choices. Just the raw, valid code/text.
    """
    
    try:
        reasoning_response = reason_client.chat.completions.create(
            model="qwen-3.5-35b",
            messages=[
                {"role": "system", "content": "You are a code merging engine. Output raw code only."},
                {"role": "user", "content": reasoner_prompt}
            ],
            max_tokens=8192 # Increased for large code files
        )
        draft = reasoning_response.choices[0].message.content
    except Exception as e:
        print(f"   ❌ [Reasoner Error]: {e}")
        return False

    # STEP B: Orchestrator Sanity Check
    print("   ⚖️ [Orchestrator] Verifying code integrity and marker removal...")
    editor_prompt = f"""
    Review this drafted code resolution for a git merge conflict.
    
    DRAFT:
    {draft}
    
    TASK: Ensure there are absolutely NO leftover git markers (<<<<<<<, =======, >>>>>>>) in the code. Ensure the code is syntactically logical. 
    Output ONLY the final code. DO NOT wrap it in ```markdown``` or ```python``` blocks.
    """

    try:
        verification_response = orch_client.chat.completions.create(
            model="nemotron-orchestrator-8b",
            messages=[
                {"role": "system", "content": "You are a strict code formatter. Output raw code/text only."},
                {"role": "user", "content": editor_prompt}
            ]
        )
        final_code = clean_llm_output(verification_response.choices[0].message.content)
    except Exception as e:
        print(f"   ❌ [Orchestrator Error]: {e}")
        return False

    # Check for failure states
    if "<<<<<<<" in final_code or "=======" in final_code:
        print("   ❌ [Error] AI failed to remove conflict markers. Manual intervention required.")
        return False

    # Write the fixed content back to the file
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(final_code)
        print(f"   ✅ Successfully resolved and saved {file_path}")
        return True
    except Exception as e:
        print(f"   ❌ Failed to write to {file_path}: {e}")
        return False

# --- 4. THE WORKFLOW ---
def perform_ai_merge(target_branch, source_branch):
    print("🤖 Starting AI-Driven Git Merge")
    print("=" * 40)
    
    # Ensure working directory is clean before starting
    status = run_git_command(["status", "--porcelain"])
    if status:
        print("❌ Your working directory is not clean. Please commit or stash your changes first.")
        sys.exit(1)

    # Create the new branch name
    safe_source = source_branch.replace("/", "-")
    safe_target = target_branch.replace("/", "-")
    merge_branch_name = f"ai-merge-{safe_source}-into-{safe_target}"

    print(f"🔄 Setting up branches...")
    print(f"   Target: {target_branch}")
    print(f"   Source: {source_branch}")
    
    # Checkout target and pull latest (assuming local exists, otherwise adjust)
    run_git_command(["checkout", target_branch])
    
    # Create and checkout the new isolation branch
    print(f"🌿 Creating isolated branch: {merge_branch_name}")
    run_git_command(["checkout", "-b", merge_branch_name])

    # Attempt the merge
    print(f"🔀 Attempting standard git merge of '{source_branch}'...")
    merge_output = run_git_command(["merge", source_branch], check=False)

    conflicting_files = get_conflicting_files()
    
    if not conflicting_files:
        print("✅ Merge completed successfully with no conflicts! Branch is ready.")
        sys.exit(0)

    print(f"⚠️ Conflicts detected in {len(conflicting_files)} file(s). Engaging AI Resolver...")
    
    # Process each conflicted file
    resolution_failures = 0
    for file_path in conflicting_files:
        success = resolve_file_with_ai(file_path)
        if success:
            run_git_command(["add", file_path])
        else:
            resolution_failures += 1

    print("\n" + "=" * 40)
    if resolution_failures == 0:
        print("🎉 All conflicts resolved by AI. Committing results...")
        run_git_command(["commit", "-m", f"AI automated merge resolution: {source_branch} into {target_branch}"])
        print(f"✅ Success! Your resolved code is waiting on branch: {merge_branch_name}")
    else:
        print(f"⚠️ AI resolved some files, but {resolution_failures} files failed or require manual review.")
        print(f"   The branch '{merge_branch_name}' is currently in a merging state.")
        print("   Please review the files, resolve manually, and run 'git commit'.")

# --- 5. CLI EXECUTION ---
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python ai_merge.py <target_branch> <source_branch>")
        print("Example: python ai_merge.py main feature/new-ui")
        sys.exit(1)

    target = sys.argv[1]
    source = sys.argv[2]
    
    perform_ai_merge(target, source)
