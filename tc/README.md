# ThereminQ Holoqubed Tool Cupboard (tc/)

A collection of auxiliary utilities and autonomous workflows for the ThereminQ Holoqubed project.

## Files

*   `deep-local-research.py`: An autonomous local LLM web research script. Powered by local LLMs (orchestrator and reasoning models via external endpoints), it performs deep web scraping via DuckDuckGo (filtering out video/image sites like YouTube and TikTok), executes a multi-step analysis (Tool Planning, Reasoning, Verification), and outputs a formatted PDF report using `fpdf`. It incorporates streaming inference and safe truncations to manage context windows effectively.
*   `git-compare-and-merge.py`: AI-Driven Git Merge Conflict Resolver. A script that detects git merge conflicts and delegates resolution to local AI models. It reads conflicted files, identifies git conflict markers (`<<<<<<<`, `=======`), sends the file to a Reasoner model to intelligently draft a resolution, and then passes it to an Orchestrator model to verify code integrity and ensure marker removal before automatically writing the resolved file back to disk.


<img width="2816" height="1536" alt="gemini_generated_image_j0jsqnj0jsqnj0js" src="https://github.com/user-attachments/assets/79406291-6eba-4621-aebd-387f852a714a" />
<img width="2528" height="1696" alt="gemini_generated_image_x8xdflx8xdflx8xd" src="https://github.com/user-attachments/assets/62637080-7f76-4b5f-b701-04d35eceb793" />
<img width="1408" height="768" alt="1776199644594" src="https://github.com/user-attachments/assets/239f4d4c-e2b0-4f58-bbbb-af68ef67848b" />
