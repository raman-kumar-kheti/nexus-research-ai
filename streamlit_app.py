"""Streamlit UI for NexusResearch AI."""

from __future__ import annotations

import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="NexusResearch AI", layout="wide")
st.title("NexusResearch AI")
st.caption("Multi-agent research and automation platform (LangGraph + MCP)")

with st.sidebar:
    st.header("Workflow Controls")
    require_approval = st.checkbox("Require Human-in-the-Loop Approval", value=False)
    max_iterations = st.slider("Max Research Iterations", min_value=1, max_value=5, value=2)

topic = st.text_input("Enter a research topic:", placeholder="e.g., Production deployment of Small Language Models on CPU")

if st.button("Run Multi-Agent Research", type="primary") and topic.strip():
    with st.spinner("Running 6-agent LangGraph research workflow..."):
        res = requests.post(
            f"{API_URL}/research",
            json={
                "topic": topic.strip(),
                "require_approval": require_approval,
                "max_iterations": max_iterations,
            },
            timeout=120,
        )
        st.session_state["last_result"] = res.json()

result = st.session_state.get("last_result")
if result:
    if result.get("status") == "awaiting_approval":
        st.warning(f"Approval required for thread `{result['thread_id']}`: {result.get('interrupt')}")
        col1, col2 = st.columns(2)
        if col1.button("Approve & Finalize"):
            res = requests.post(
                f"{API_URL}/research/approve",
                json={"thread_id": result["thread_id"], "approved": True},
                timeout=60,
            )
            st.session_state["last_result"] = res.json()
            st.rerun()
        if col2.button("Reject Persistence"):
            res = requests.post(
                f"{API_URL}/research/approve",
                json={"thread_id": result["thread_id"], "approved": False},
                timeout=60,
            )
            st.session_state["last_result"] = res.json()
            st.rerun()
    else:
        st.markdown(result.get("report_markdown", ""))
        if result.get("evaluation"):
            st.subheader("Workflow Evaluation & Telemetry")
            st.json(result["evaluation"])
        if result.get("telemetry", {}).get("traces"):
            st.subheader("Agent Execution Trace")
            st.dataframe(result["telemetry"]["traces"], use_container_width=True)
