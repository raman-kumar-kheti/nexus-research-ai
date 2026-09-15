"""Model Context Protocol (MCP) server and in-process tool client for NexusResearch AI."""

from __future__ import annotations

from typing import Any

import wikipedia
from mcp.server.mcpserver import MCPServer

from app.memory.database import save_research, search_history
from app.tools.search import web_search

mcp_server = MCPServer("nexus-research-tools")


@mcp_server.tool()
def tool_web_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Search the web via DuckDuckGo and return structured results with URLs."""
    try:
        raw = web_search(query, max_results=max_results)
        if isinstance(raw, list):
            return raw
        return [{"title": "Web Search", "url": "", "snippet": str(raw)}]
    except Exception as exc:
        return [{"title": "Search Error", "url": "", "snippet": f"Search unavailable: {exc}"}]


@mcp_server.tool()
def tool_wikipedia_fetch(topic: str) -> dict[str, str]:
    """Fetch a Wikipedia summary and canonical URL."""
    try:
        page = wikipedia.page(topic, auto_suggest=True)
        return {"title": page.title, "url": page.url, "summary": page.summary[:1500]}
    except Exception as exc:
        return {"title": topic, "url": "", "summary": f"Wikipedia lookup unavailable: {exc}"}


@mcp_server.tool()
def tool_save_memory(
    query: str,
    summary: str,
    username: str = "anonymous",
    citations_json: str = "[]",
) -> dict[str, str]:
    """Persist verified research findings to memory."""
    save_research(query=query, summary=summary, username=username, citations_json=citations_json)
    return {"status": "saved"}


@mcp_server.tool()
def tool_search_memory(query: str, username: str = "anonymous") -> list[dict[str, Any]]:
    """Search previously persisted research findings."""
    return search_history(query=query, username=username)


class MCPToolClient:
    """Clean MCP tool client used by LangGraph agents without self-HTTP deadlocks."""

    def call(self, tool_name: str, **kwargs: Any) -> Any:
        if tool_name == "web_search":
            return tool_web_search(**kwargs)
        if tool_name == "wikipedia_fetch":
            return tool_wikipedia_fetch(**kwargs)
        if tool_name == "save_memory":
            return tool_save_memory(**kwargs)
        if tool_name == "search_memory":
            return tool_search_memory(**kwargs)
        raise ValueError(f"Unknown MCP tool: {tool_name}")


if __name__ == "__main__":
    mcp_server.run()
