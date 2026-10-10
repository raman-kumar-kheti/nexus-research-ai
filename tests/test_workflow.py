"""Tests for NexusResearch AI 6-agent LangGraph workflow, HITL, MCP, and FastAPI."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

os.environ["LLM_PROVIDER"] = "mock"

from app.main import app
from app.tools.doc_generator import create_pdf


def _mock_mcp_call(tool_name: str, **kwargs):
    if tool_name == "web_search":
        return [
            {
                "title": "LangGraph Production Guide",
                "url": "https://example.com/langgraph",
                "snippet": "LangGraph provides stateful multi-agent orchestration with checkpointing.",
            }
        ]
    if tool_name == "wikipedia_fetch":
        return {
            "title": "Multi-agent system",
            "url": "https://en.wikipedia.org/wiki/Multi-agent_system",
            "summary": "A multi-agent system is composed of multiple interacting intelligent agents.",
        }
    if tool_name == "save_memory":
        return {"status": "saved"}
    return []


def test_health_and_auth_flow() -> None:
    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json()["status"] == "healthy"


def test_six_agent_workflow_and_hitl_approval() -> None:
    with patch("app.agents.nexus_graph.mcp_client.call", side_effect=_mock_mcp_call):
        with TestClient(app) as client:
            # 1. Start workflow with HITL approval enabled
            res = client.post(
                "/research",
                json={
                    "topic": "Stateful LangGraph multi-agent systems",
                    "thread_id": "test-hitl-1",
                    "require_approval": True,
                    "max_iterations": 2,
                },
            )
            assert res.status_code == 200
            body = res.json()
            assert body["status"] == "awaiting_approval"
            assert body["thread_id"] == "test-hitl-1"

            # 2. Resume with approval
            approve_res = client.post(
                "/research/approve",
                json={"thread_id": "test-hitl-1", "approved": True},
            )
            assert approve_res.status_code == 200
            final = approve_res.json()
            assert final["status"] == "completed"
            assert "## 1. Verified Facts" in final["report_markdown"]
            assert "## 4. Verified References" in final["report_markdown"]
            assert "https://example.com/langgraph" in final["report_markdown"]
            assert final["evaluation"]["structure_score"] == 1.0
            assert len(final["telemetry"]["traces"]) == 6


def test_pdf_export_handles_xml_special_chars(tmp_path: Path) -> None:
    out_pdf = tmp_path / "test_report.pdf"
    create_pdf("Report <Alpha & Beta>", "## Section\n- Item with 1 < 2 & 3 > 1", out_pdf)
    assert out_pdf.exists()
    assert out_pdf.stat().st_size > 500
