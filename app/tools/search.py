"""DuckDuckGo web search tool returning structured source records."""

from __future__ import annotations

from ddgs import DDGS


def web_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Search DuckDuckGo and return title, url, and snippet dictionaries."""
    results: list[dict[str, str]] = []
    with DDGS() as ddgs:
        for item in ddgs.text(query, max_results=max_results):
            results.append(
                {
                    "title": str(item.get("title", "Untitled")),
                    "url": str(item.get("href", item.get("link", ""))),
                    "snippet": str(item.get("body", item.get("snippet", ""))),
                }
            )
    return results
