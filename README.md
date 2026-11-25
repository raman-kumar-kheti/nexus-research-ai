# NexusResearch AI (`nexus-research-ai`)

> **Multi-agent research and automation platform** built with LangGraph, Model Context Protocol (MCP), FastAPI, Streamlit, and PostgreSQL/SQLite checkpointing.

## 1. Problem Statement
Unconstrained LLM agents often enter infinite loops, fabricate citations, mix speculation with verified facts, and lack human oversight before persisting or exporting results. NexusResearch AI solves this with a stateful 6-agent LangGraph workflow, bounded retries/budgets, MCP tool integration, and human-in-the-loop approval gates.

## 2. Project Overview
This repository transforms [nzfadhiya/agentic-research-assistant](https://github.com/nzfadhiya/agentic-research-assistant) into a structured, testable multi-agent research platform.

### Upstream vs. Original Contributions
- **Inherited from `agentic-research-assistant`:** FastAPI + Streamlit split, PostgreSQL/SQLite persistence layer, JWT auth structure, DuckDuckGo & Wikipedia search tools, and ReportLab/DOCX export utilities.
- **Original Additions:**
  - Explicit **6-agent LangGraph workflow** (`Planner` → parallel `Web Search` → `Extractor` → `Analyst` → `Critic/Fact-Checker` → `Approval` → `Reporter`).
  - **LangGraph `SqliteSaver` checkpointing** and **Human-in-the-Loop (`interrupt` / `Command`)** approval before finalizing reports.
  - **Real MCP Server & Client (`app/mcp_server/server.py`)** replacing synchronous self-HTTP loopbacks.
  - **Execution Guardrails (`app/core/telemetry.py`):** Per-agent latency tracing, token/cost budget tracking, bounded retries, wall-clock timeouts, and `MAX_ITERATIONS` loop limits.
  - **Fact-vs-Interpretation Separation & Verified-Only Citations:** Never fabricates URLs; only cites retrieved sources.
  - **Agent Evaluation Framework (`app/core/evaluator.py`)** and **XML-escaped PDF + Markdown export**.
  - **Self-contained Pytest suite**, **Dockerfile**, and **GitHub Actions CI**.

## 3. Six-Agent Architecture
1. **Research Planning Agent (`planner`):** Decomposes the topic into focused sub-queries.
2. **Web Search Agent (`search`):** Executes parallel DuckDuckGo & Wikipedia lookups via `MCPToolClient`.
3. **Information Extraction Agent (`extractor`):** Extracts verifiable facts linked to source indices.
4. **Research Analysis Agent (`analyst`):** Synthesizes findings while separating verified facts from interpretations.
5. **Critic & Fact-Checking Agent (`critic`):** Audits claims against retrieved sources and controls bounded iteration.
6. **Report Generation Agent (`reporter`):** Compiles structured Markdown/PDF reports after the **Human-in-the-Loop (`approval`)** gate.

## 4. Technology Stack
- **Orchestration & Protocol:** LangGraph, LangChain, Model Context Protocol (`mcp`)
- **Backend & UI:** Python 3.12, FastAPI, Pydantic, Streamlit
- **Persistence & Auth:** PostgreSQL (`psycopg2-binary`) or SQLite WAL, `SqliteSaver` checkpoints, JWT (`python-jose`)
- **Tools & Exports:** `ddgs`, `wikipedia`, PyMuPDF, ReportLab, `python-docx`
- **Testing & DevOps:** Pytest, Docker, Docker Compose, GitHub Actions

## 5. Installation & Environment Configuration
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## 6. How to Run Locally
```bash
# FastAPI backend
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Streamlit UI (in another terminal)
streamlit run streamlit_app.py
```

## 7. How to Run Tests
```bash
LLM_PROVIDER=mock pytest tests/ -v
```

## 8. Docker Instructions
```bash
docker compose up --build
```

## 9. Original Repository Attribution & License
- **Original Upstream Repository:** [nzfadhiya/agentic-research-assistant](https://github.com/nzfadhiya/agentic-research-assistant) (no `LICENSE` file was present in upstream).
- Upstream contributions are preserved and attributed; all extensions above are original.
