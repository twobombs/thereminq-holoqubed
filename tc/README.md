# ThereminQ Holoqubed Tool Cupboard (tc/)

A collection of auxiliary utilities and autonomous workflows for the ThereminQ Holoqubed project.

## Files

*   `deep-local-research.py`: An autonomous local LLM web research script. Powered by local LLMs (orchestrator and reasoning models via external endpoints), it performs deep web scraping via DuckDuckGo (filtering out video/image sites like YouTube and TikTok), executes a multi-step analysis (Tool Planning, Reasoning, Verification), and outputs a formatted PDF report using `fpdf`. It incorporates streaming inference and safe truncations to manage context windows effectively.
*   `git-compare-and-merge.py`: AI-Driven Git Merge Conflict Resolver. A script that detects git merge conflicts and delegates resolution to local AI models. It reads conflicted files, identifies git conflict markers (`<<<<<<<`, `=======`), sends the file to a Reasoner model to intelligently draft a resolution, and then passes it to an Orchestrator model to verify code integrity and ensure marker removal before automatically writing the resolved file back to disk.
*   `AgenticAgile.py`: Agentic Project Orchestration pipeline. Utilizing the Google Gemini API (`gemini-1.5-pro`), it manages a "Living Wiki" by continuously ingesting raw transcripts (e.g., mock Slack logs) and extracting actionable tasks, architectural decisions, and blockers. It includes an Agentic Linter to verify `ACTIVE_TASKS.md` against project rules (`DEFINITION_OF_DONE.md`), generating automated linting reports, and finally synthesizes an Automated Daily Agile Dashboard summarizing progress, risks, and velocity adjustments.
*   `local-discord-bot.py`: A simple integration bridging a local Discord bot (via `discord.py`) to a local LLM. The bot responds to direct mentions, asynchronously querying a local server (e.g., `llama-server` on port 8033) via `asyncio.to_thread` to ensure non-blocking interactions while streaming AI responses seamlessly into Discord channels.
*   `llm-wiki.py`: An automated knowledge compiler for an LLM Wiki. It uses an OpenAI-compatible local API to read raw source text files (`.txt`, `.md`, `.csv`) and synthesizes them into structured Markdown wiki pages with YAML frontmatter. It also dynamically routes files to appropriate directories based on type and manages an index and log of ingested documents.
*   `mcp-workspace-bridge.py`: A FastMCP server script acting as a bridge to the local workspace. It exposes read-only resources (Agile project state, Wiki index, Daily Synthesis) and actionable tools (document ingestion, local orchestrator query) over standard input/output (stdio) for secure external client access.

<img width="2816" height="1536" alt="Gemini_Generated_Image_3cnxrm3cnxrm3cnx" src="https://github.com/user-attachments/assets/100a743b-a893-42b5-8e37-3dbc221ed72f" />
<img width="2816" height="1536" alt="Gemini_Generated_Image_j6xa7dj6xa7dj6xa" src="https://github.com/user-attachments/assets/db95e0fa-c8bb-4e96-9aac-9c9224aa1ed2" />
<img width="2816" height="1536" alt="gemini_generated_image_j0jsqnj0jsqnj0js" src="https://github.com/user-attachments/assets/79406291-6eba-4621-aebd-387f852a714a" />
<img width="2528" height="1696" alt="gemini_generated_image_x8xdflx8xdflx8xd" src="https://github.com/user-attachments/assets/62637080-7f76-4b5f-b701-04d35eceb793" />
