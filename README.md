# orbitResearch

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]((https://orbit-research.streamlit.app/))

Multi-agent AI research assistant that orchestrates specialized agents to search, analyze, and synthesize information from multiple sources — built with LangGraph, LangChain, and Groq-hosted LLMs.

## Overview

orbitResearch breaks down research queries into tasks handled by a coordinated team of agents rather than a single LLM call. Each agent has access to a specific tool (arXiv, GitHub, web search, Wikipedia, YouTube) and the system routes queries dynamically, aggregates findings, and returns synthesized results through a Streamlit interface.

## Features

- **Multi-agent orchestration** — LangGraph-based state machine coordinates specialized research agents instead of relying on a single monolithic prompt
- **Multi-source research tools:**
  - arXiv — academic paper search
  - GitHub — repository and code search
  - Web Search — general information retrieval
  - Wikipedia — reference and background knowledge
  - YouTube — video content search
- **Dynamic tool-name guide** — tool names and routing are generated at runtime, reducing agent-tool mismatches
- **Groq-hosted LLM inference** — fast, low-latency responses via Groq's LPU infrastructure
- **Streamlit multi-page UI** for interacting with the research pipeline

## Tech Stack

| Component | Technology |
|---|---|
| Agent Orchestration | LangGraph |
| LLM Framework | LangChain |
| LLM Inference | Groq (LLaMA models) |
| Frontend | Streamlit |
| Language | Python |

## Screenshots

<!-- Add screenshots here -->

## Installation

```bash
git clone https://github.com/Agent-Orbit/orbitResearch.git
cd orbitResearch
pip install -r requirements.txt
```

## Configuration

Create a `.streamlit/secrets.toml` file with your API keys:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

*(Add any additional keys your tools require — e.g. GitHub token, search API key.)*

## Usage

```bash
streamlit run app.py
```

Enter a research query in the UI. The system routes it through the agent graph, calls the relevant tools, and returns a synthesized response.

## How It Works

1. **Query intake** — user submits a research question via the Streamlit UI
2. **Routing** — the graph determines which agent(s)/tool(s) are relevant using the dynamic tool-name guide
3. **Tool execution** — specialized agents query arXiv, GitHub, Wikipedia, YouTube, or the web as needed
4. **Synthesis** — results are aggregated and returned as a coherent answer

## Author

**Ali Akbar**
Self-taught ML/AI Developer — building AI-powered tools and autonomous agent systems.

- LinkedIn: [linkedin.com/in/ali-akbar-3148b940b](https://www.linkedin.com/in/ali-akbar-3148b940b/)
- GitHub: [@Agent-Orbit](https://github.com/Agent-Orbit)

## License

MIT
