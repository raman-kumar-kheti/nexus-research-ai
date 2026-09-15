"""Agent execution tracing, token/cost budgets, and retry/timeout wrappers."""

from __future__ import annotations

import concurrent.futures
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from app.config import settings

T = TypeVar("T")


def estimate_tokens(text: str) -> int:
    return max(1, len(re.findall(r"\S+", text))) if text.strip() else 0


@dataclass
class TraceEvent:
    """Single agent execution span."""

    agent: str
    status: str
    latency_ms: float
    tokens_used: int = 0
    detail: str = ""


@dataclass
class BudgetTracker:
    """Tracks token and estimated USD cost against configured limits."""

    max_tokens: int = settings.MAX_TOKEN_BUDGET
    max_cost_usd: float = settings.MAX_COST_USD
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    traces: list[TraceEvent] = field(default_factory=list)

    def record(self, agent: str, prompt: str, response: str, latency_ms: float, status: str = "ok", detail: str = "") -> None:
        pt = estimate_tokens(prompt)
        ct = estimate_tokens(response)
        self.prompt_tokens += pt
        self.completion_tokens += ct
        # Conservative blended rate ($0.15 / 1M tokens)
        self.cost_usd += (pt + ct) * 0.00000015
        self.traces.append(
            TraceEvent(
                agent=agent,
                status=status,
                latency_ms=round(latency_ms, 2),
                tokens_used=pt + ct,
                detail=detail,
            )
        )

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def within_budget(self) -> bool:
        return self.total_tokens <= self.max_tokens and self.cost_usd <= self.max_cost_usd

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(self.cost_usd, 6),
            "within_budget": self.within_budget(),
            "traces": [
                {
                    "agent": t.agent,
                    "status": t.status,
                    "latency_ms": t.latency_ms,
                    "tokens_used": t.tokens_used,
                    "detail": t.detail,
                }
                for t in self.traces
            ],
        }


def run_with_retry(
    fn: Callable[[], T],
    max_retries: int | None = None,
    timeout_seconds: float | None = None,
) -> T:
    """Execute a callable with bounded retries and wall-clock timeout."""
    retries = settings.MAX_RETRIES if max_retries is None else max_retries
    timeout = settings.AGENT_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
    last_exc: Exception | None = None

    for attempt in range(retries + 1):
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(fn)
                return future.result(timeout=timeout)
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(min(0.5 * (attempt + 1), 2.0))
    raise RuntimeError(f"Operation failed after {retries + 1} attempts: {last_exc}") from last_exc
