# ThereminQ Holoqubed Tool Cupboard (tc/)

A collection of auxiliary utilities and autonomous workflows for the ThereminQ Holoqubed project.

## Files

### `deep-local-research.py`
**Functional Description:** An autonomous local LLM web research script that performs deep web scraping and generates formatted PDF reports.
**Internal Workings:** It queries DuckDuckGo (`ddgs`) and scrapes webpages using `requests` and `BeautifulSoup`, filtering out video and image domains (e.g., YouTube, TikTok) while truncating text to manage context windows effectively. The workflow uses an Orchestrator model (`nemotron-orchestrator-8b`) for tool planning and final editing, and a Reasoner model (`qwen-3.5-35b`) for deep logic analysis. It utilizes streaming inference, a token leash to prevent runaway loops, and outputs the final result as a PDF report using `fpdf`.

### `git-compare-and-merge.py`
**Functional Description:** AI-Driven Git Merge Conflict Resolver.
**Internal Workings:** It detects git merge conflicts by parsing `git diff` and `MERGE_MSG`, or can manually create an isolated AI merge branch (`ai-merge-<source>-into-<target>`). Conflicted files are processed by sending the file content with standard git conflict markers (`<<<<<<<`, `=======`) to a Reasoner model (`qwen-3.5-35b`) to draft an intelligent resolution. The draft is then passed to an Orchestrator model (`nemotron-orchestrator-8b`) to ensure code integrity and the complete removal of conflict markers. The script cleans the LLM output (e.g., stripping markdown code blocks) before writing the resolved code back to the file.

### `AgenticAgile.py`
**Functional Description:** Agentic Project Orchestration pipeline that manages project state based on raw transcripts and generates an automated daily synthesis.
**Internal Workings:** The script processes raw text (e.g., mock Slack logs) by chunking it and using a `concurrent.futures.ThreadPoolExecutor` to dispatch sliding-window micro-tasks across a simulated cluster of local LLM worker nodes. An Orchestrator node reconciles the newly extracted signals (tasks, blockers) with the existing project state managed via a JSON state file (`project_state.json`), updating semantic identity, lifecycle status, and confidence scores. It features a self-healing dead-letter queue for recovering failed extraction chunks and outputs a final project summary in `DAILY_SYNTHESIS.md`.

### `llm-wiki.py`
**Functional Description:** Automated knowledge compiler for an LLM Wiki.
**Internal Workings:** It sequentially reads raw documents (`.txt`, `.md`, `.csv`) and prompts a local LLM (via an OpenAI-compatible API) to synthesize the raw text into structured wiki pages. It uses regular expressions to parse expected YAML frontmatter from the LLM's response, determining the document's type. Based on this type, the script dynamically routes the compiled markdown file to a specific subdirectory (`concepts`, `entities`, `sources`, `comparisons`), updates an `index.md`, and logs the operation to `log.md`.

### `local-discord-bot.py`
**Functional Description:** An integration bridging a local Discord bot to a local LLM.
**Internal Workings:** Using the `discord.py` library, the bot listens for direct mentions. It cleans the user's message and asynchronously queries a local server (e.g., `llama-server` on port 8033) via `asyncio.to_thread`. This threaded approach ensures non-blocking interactions while the bot waits for the synchronous HTTP POST response. It then parses the JSON response to reply back within the Discord channel and handles connection exceptions.

<img width="2816" height="1536" alt="Gemini_Generated_Image_j6xa7dj6xa7dj6xa" src="https://github.com/user-attachments/assets/db95e0fa-c8bb-4e96-9aac-9c9224aa1ed2" />
<img width="2816" height="1536" alt="gemini_generated_image_j0jsqnj0jsqnj0js" src="https://github.com/user-attachments/assets/79406291-6eba-4621-aebd-387f852a714a" />
<img width="2528" height="1696" alt="gemini_generated_image_x8xdflx8xdflx8xd" src="https://github.com/user-attachments/assets/62637080-7f76-4b5f-b701-04d35eceb793" />
