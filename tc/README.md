# ThereminQ Holoqubed Tool Cupboard (tc/)

A collection of auxiliary utilities and autonomous workflows for the ThereminQ Holoqubed project.

## Files

*   `deep-local-research.py`: An autonomous local LLM web research script. Powered by local LLMs (orchestrator and reasoning models via external endpoints), it performs deep web scraping via DuckDuckGo (filtering out video/image sites like YouTube and TikTok), executes a multi-step analysis (Tool Planning, Reasoning, Verification), and outputs a formatted PDF report using `fpdf`. It incorporates streaming inference and safe truncations to manage context windows effectively.
*   `git-compare-and-merge.py`: AI-Driven Git Merge Conflict Resolver. A script that detects git merge conflicts and delegates resolution to local AI models. It reads conflicted files, identifies git conflict markers (`<<<<<<<`, `=======`), sends the file to a Reasoner model to intelligently draft a resolution, and then passes it to an Orchestrator model to verify code integrity and ensure marker removal before automatically writing the resolved file back to disk.
