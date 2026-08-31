# Autonomous Food-Waste Reduction Agent

A cyclical, multi-agent state machine designed to autonomously audit leftover ingredients, retrieve real-time nutritional data, and generate zero-waste recipes strictly adhering to user dietary constraints. 

Engineered for edge deployment, this system runs entirely locally using quantized LLMs to ensure zero cloud API costs and complete data privacy.

## System Architecture

The pipeline implements a **Planner-Worker-Critic** triad using LangGraph, overriding the linear and non-deterministic limitations of standard LLM prompts.

1. **Food Safety Auditor (Planner):** A deterministic node that evaluates raw user input, flags high-risk items, and establishes a strict "Verified Safe" baseline.
2. **Nutritional Researcher (Tool Worker):** Executes a custom Python `requests` tool to query the live Open Food Facts REST API, utilizing custom headers to bypass bot-blockers and fetch real-world allergen data.
3. **Executive Chef (Generator):** Drafts zero-waste recipes based exclusively on the safe list and API data. Engineered with graceful degradation to handle API `503 Service Unavailable` rate-limiting dynamically.
4. **Quality Control Auditor (Critic):** Evaluates the generated recipe against original constraints. If rule violations or LLM hallucinations are detected, a conditional edge routes the state backward for self-correction.

## Tech Stack
* **Core Logic:** Python, LangChain, LangGraph
* **LLM Engine:** Llama 3.2 (3B Parameter) via Ollama
* **Frontend:** Streamlit
* **External Integration:** Open Food Facts Global API

## Local Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Flinty21/leftover-food-agent.git
   cd leftover-food-agent