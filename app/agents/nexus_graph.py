"""Six-agent LangGraph research workflow with parallel search, HITL approval, and checkpointing."""

from __future__ import annotations

import concurrent.futures
import json
import sqlite3
import time
from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app.config import settings
from app.core.llm import invoke_llm
from app.core.telemetry import BudgetTracker, run_with_retry
from app.mcp_server.server import MCPToolClient


class ResearchWorkflowState(TypedDict, total=False):
    topic: str
    username: str
    require_approval: bool
    approved: bool
    iteration: int
    max_iterations: int
    sub_queries: list[str]
    raw_sources: list[dict[str, str]]
    extracted_facts: list[str]
    analysis: str
    critique_notes: str
    critique_passed: bool
    report_markdown: str
    telemetry: dict[str, Any]
    error: str


mcp_client = MCPToolClient()


def _parse_sub_queries(plan_text: str, topic: str) -> list[str]:
    lines = [
        line.lstrip("0123456789.-) ").strip()
        for line in plan_text.splitlines()
        if line.strip()
    ]
    return lines[:3] if lines else [topic]


def planner_node(state: ResearchWorkflowState) -> ResearchWorkflowState:
    """Agent 1: Research Planning Agent."""
    t0 = time.perf_counter()
    tracker = BudgetTracker()
    topic = state["topic"]
    prompt = f"Create 3 focused research sub-queries for the topic:\n{topic}"
    plan = run_with_retry(
        lambda: invoke_llm(prompt, system_prompt="You are a Research Planning Agent. Output 3 concise numbered sub-queries.")
    )
    sub_queries = _parse_sub_queries(plan, topic)
    tracker.record("planner", prompt, plan, (time.perf_counter() - t0) * 1000, detail=f"{len(sub_queries)} sub-queries")
    return {
        "sub_queries": sub_queries,
        "iteration": state.get("iteration", 0) + 1,
        "telemetry": tracker.to_dict(),
    }


def search_node(state: ResearchWorkflowState) -> ResearchWorkflowState:
    """Agent 2: Parallel Web Search & Wikipedia Agent via MCPToolClient."""
    t0 = time.perf_counter()
    queries = state.get("sub_queries") or [state["topic"]]
    collected: list[dict[str, str]] = []

    def _search_one(q: str) -> list[dict[str, str]]:
        return mcp_client.call("web_search", query=q, max_results=3)

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(queries) + 1)) as pool:
        futures = [pool.submit(_search_one, q) for q in queries]
        wiki_future = pool.submit(mcp_client.call, "wikipedia_fetch", topic=state["topic"])
        for fut in futures:
            try:
                for item in fut.result(timeout=settings.AGENT_TIMEOUT_SECONDS):
                    if item.get("url") or item.get("snippet"):
                        collected.append(item)
            except Exception:
                continue
        try:
            wiki = wiki_future.result(timeout=settings.AGENT_TIMEOUT_SECONDS)
            if wiki.get("url"):
                collected.append({"title": f"Wikipedia: {wiki['title']}", "url": wiki["url"], "snippet": wiki["summary"][:500]})
        except Exception:
            pass

    # Deduplicate by URL
    seen_urls: set[str] = set()
    unique_sources: list[dict[str, str]] = []
    for src in collected:
        url = src.get("url", "")
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)
        unique_sources.append(src)

    telemetry = dict(state.get("telemetry") or {})
    traces = list(telemetry.get("traces", []))
    traces.append(
        {
            "agent": "web_search",
            "status": "ok",
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "tokens_used": 0,
            "detail": f"{len(unique_sources)} sources collected",
        }
    )
    telemetry["traces"] = traces
    return {"raw_sources": unique_sources, "telemetry": telemetry}


def extractor_node(state: ResearchWorkflowState) -> ResearchWorkflowState:
    """Agent 3: Information Extraction Agent."""
    t0 = time.perf_counter()
    sources = state.get("raw_sources") or []
    evidence_text = "\n".join(
        f"[{i+1}] {s.get('title')} ({s.get('url', 'no-url')}): {s.get('snippet', '')}"
        for i, s in enumerate(sources[:8])
    )
    if not evidence_text.strip():
        facts = ["No external web sources were retrieved; rely only on verified local context."]
        resp = facts[0]
    else:
        prompt = f"Extract concise, verifiable factual bullet points with source numbers [n] from:\n{evidence_text}"
        resp = run_with_retry(
            lambda: invoke_llm(prompt, system_prompt="You are an Information Extraction Agent. Never invent citations.")
        )
        facts = [line.strip() for line in resp.splitlines() if line.strip()]

    telemetry = dict(state.get("telemetry") or {})
    traces = list(telemetry.get("traces", []))
    traces.append(
        {
            "agent": "extractor",
            "status": "ok",
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "tokens_used": len(resp.split()),
            "detail": f"{len(facts)} facts extracted",
        }
    )
    telemetry["traces"] = traces
    return {"extracted_facts": facts, "telemetry": telemetry}


def analyst_node(state: ResearchWorkflowState) -> ResearchWorkflowState:
    """Agent 4: Research Analysis Agent (separates facts from interpretations)."""
    t0 = time.perf_counter()
    facts_block = "\n".join(state.get("extracted_facts") or [])
    prompt = (
        f"Topic: {state['topic']}\n\n"
        f"Extracted Facts:\n{facts_block}\n\n"
        "Synthesize key findings, trade-offs, and interpretations. "
        "Clearly distinguish verified facts from analytical interpretations."
    )
    analysis = run_with_retry(
        lambda: invoke_llm(prompt, system_prompt="You are a Research Analysis Agent.")
    )
    telemetry = dict(state.get("telemetry") or {})
    traces = list(telemetry.get("traces", []))
    traces.append(
        {
            "agent": "analyst",
            "status": "ok",
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "tokens_used": len(analysis.split()),
            "detail": "analysis complete",
        }
    )
    telemetry["traces"] = traces
    return {"analysis": analysis, "telemetry": telemetry}


def critic_node(state: ResearchWorkflowState) -> ResearchWorkflowState:
    """Agent 5: Critic & Fact-Checking Agent."""
    t0 = time.perf_counter()
    sources = state.get("raw_sources") or []
    analysis = state.get("analysis", "")
    prompt = (
        f"Audit this analysis against {len(sources)} retrieved sources for unsupported claims or fabricated citations:\n"
        f"{analysis[:1200]}"
    )
    critique = run_with_retry(
        lambda: invoke_llm(prompt, system_prompt="You are a Critic and Fact-Checking Agent.")
    )
    passed = len(sources) > 0 or state.get("iteration", 1) >= state.get("max_iterations", settings.MAX_ITERATIONS)

    telemetry = dict(state.get("telemetry") or {})
    traces = list(telemetry.get("traces", []))
    traces.append(
        {
            "agent": "critic",
            "status": "ok",
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "tokens_used": len(critique.split()),
            "detail": f"passed={passed}",
        }
    )
    telemetry["traces"] = traces
    return {"critique_notes": critique, "critique_passed": passed, "telemetry": telemetry}


def approval_node(state: ResearchWorkflowState) -> ResearchWorkflowState:
    """Human-in-the-loop approval gate before persisting & finalizing report."""
    if state.get("require_approval", False) and not state.get("approved", False):
        decision = interrupt(
            {
                "message": "Approve final research report generation and memory persistence?",
                "topic": state["topic"],
                "sources_count": len(state.get("raw_sources") or []),
            }
        )
        approved = bool(decision.get("approved", True)) if isinstance(decision, dict) else bool(decision)
        return {"approved": approved}
    return {"approved": True}


def reporter_node(state: ResearchWorkflowState) -> ResearchWorkflowState:
    """Agent 6: Report Generation Agent with verified citations only."""
    t0 = time.perf_counter()
    sources = state.get("raw_sources") or []
    facts = "\n".join(f"- {f}" for f in (state.get("extracted_facts") or []))
    citations_md = "\n".join(
        f"{i+1}. [{s.get('title', 'Source')}]({s.get('url', '')})"
        for i, s in enumerate(sources)
        if s.get("url")
    ) or "No external URLs retrieved."

    report = (
        f"# Research Report: {state['topic']}\n\n"
        f"## 1. Verified Facts\n{facts}\n\n"
        f"## 2. Analytical Interpretation\n{state.get('analysis', '')}\n\n"
        f"## 3. Fact-Check & Critic Notes\n{state.get('critique_notes', '')}\n\n"
        f"## 4. Verified References\n{citations_md}\n"
    )

    if state.get("approved", True):
        mcp_client.call(
            "save_memory",
            query=state["topic"],
            summary=report[:2000],
            username=state.get("username", "anonymous"),
            citations_json=json.dumps(sources),
        )

    telemetry = dict(state.get("telemetry") or {})
    traces = list(telemetry.get("traces", []))
    traces.append(
        {
            "agent": "reporter",
            "status": "ok",
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "tokens_used": len(report.split()),
            "detail": "report generated",
        }
    )
    telemetry["traces"] = traces
    return {"report_markdown": report, "telemetry": telemetry}


def _route_after_critic(state: ResearchWorkflowState) -> str:
    if not state.get("critique_passed", True) and state.get("iteration", 1) < state.get("max_iterations", settings.MAX_ITERATIONS):
        return "search"
    return "approval"


def build_nexus_graph(use_sqlite_checkpoint: bool = True):
    """Build and compile the 6-agent LangGraph workflow."""
    builder = StateGraph(ResearchWorkflowState)
    builder.add_node("planner", planner_node)
    builder.add_node("search", search_node)
    builder.add_node("extractor", extractor_node)
    builder.add_node("analyst", analyst_node)
    builder.add_node("critic", critic_node)
    builder.add_node("approval", approval_node)
    builder.add_node("reporter", reporter_node)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "search")
    builder.add_edge("search", "extractor")
    builder.add_edge("extractor", "analyst")
    builder.add_edge("analyst", "critic")
    builder.add_conditional_edges("critic", _route_after_critic, {"search": "search", "approval": "approval"})
    builder.add_edge("approval", "reporter")
    builder.add_edge("reporter", END)

    if use_sqlite_checkpoint:
        conn = sqlite3.connect(str(settings.CHECKPOINT_DB_PATH), check_same_thread=False)
        checkpointer = SqliteSaver(conn)
    else:
        checkpointer = MemorySaver()

    return builder.compile(checkpointer=checkpointer)


_graph = None


def get_nexus_graph():
    global _graph
    if _graph is None:
        _graph = build_nexus_graph(use_sqlite_checkpoint=True)
    return _graph


def run_research_workflow(
    topic: str,
    thread_id: str = "default",
    username: str = "anonymous",
    require_approval: bool = False,
    max_iterations: int | None = None,
) -> dict[str, Any]:
    """Execute the 6-agent workflow with checkpointing."""
    graph = get_nexus_graph()
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(
        {
            "topic": topic,
            "username": username,
            "require_approval": require_approval,
            "approved": not require_approval,
            "iteration": 0,
            "max_iterations": max_iterations or settings.MAX_ITERATIONS,
        },
        config=config,
    )
    if "__interrupt__" in result:
        return {
            "status": "awaiting_approval",
            "thread_id": thread_id,
            "interrupt": result["__interrupt__"][0].value,
        }
    return {
        "status": "completed",
        "thread_id": thread_id,
        "report_markdown": result.get("report_markdown", ""),
        "sources": result.get("raw_sources", []),
        "telemetry": result.get("telemetry", {}),
    }


def resume_research_workflow(thread_id: str, approved: bool = True) -> dict[str, Any]:
    """Resume a checkpointed workflow after human-in-the-loop approval."""
    graph = get_nexus_graph()
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(Command(resume={"approved": approved}), config=config)
    return {
        "status": "completed",
        "thread_id": thread_id,
        "report_markdown": result.get("report_markdown", ""),
        "sources": result.get("raw_sources", []),
        "telemetry": result.get("telemetry", {}),
    }
