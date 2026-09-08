"""Agent workflow evaluation framework."""

from __future__ import annotations

from typing import Any


def evaluate_workflow_output(result: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a completed research workflow for citation coverage, structure, and latency."""
    report = result.get("report_markdown", "")
    sources = result.get("sources", [])
    traces = result.get("telemetry", {}).get("traces", [])

    has_facts_section = "## 1. Verified Facts" in report
    has_interp_section = "## 2. Analytical Interpretation" in report
    has_refs_section = "## 4. Verified References" in report
    total_latency_ms = round(sum(float(t.get("latency_ms", 0.0)) for t in traces), 2)
    agents_executed = [t.get("agent") for t in traces]

    return {
        "source_count": len(sources),
        "has_verified_facts_section": has_facts_section,
        "has_interpretation_section": has_interp_section,
        "has_references_section": has_refs_section,
        "structure_score": round(
            (int(has_facts_section) + int(has_interp_section) + int(has_refs_section)) / 3.0,
            4,
        ),
        "agents_executed": agents_executed,
        "total_latency_ms": total_latency_ms,
    }
