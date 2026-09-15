"""Configurable LLM wrapper supporting Groq, OpenAI-compatible APIs, and offline mock mode."""

from __future__ import annotations

import httpx

from app.config import settings


def invoke_llm(prompt: str, system_prompt: str = "", provider: str | None = None) -> str:
    """Invoke the configured LLM provider with safe fallback."""
    selected = (provider or settings.LLM_PROVIDER).lower()

    if selected == "groq":
        if not settings.GROQ_API_KEY:
            selected = "mock"
        else:
            from langchain_core.messages import HumanMessage, SystemMessage
            from langchain_groq import ChatGroq

            llm = ChatGroq(api_key=settings.GROQ_API_KEY, model=settings.GROQ_MODEL, temperature=0.2)
            msgs = []
            if system_prompt:
                msgs.append(SystemMessage(content=system_prompt))
            msgs.append(HumanMessage(content=prompt))
            return str(llm.invoke(msgs).content).strip()

    if selected == "openai_compatible":
        if not settings.OPENAI_API_BASE or not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_BASE and OPENAI_API_KEY are required.")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        resp = httpx.post(
            f"{settings.OPENAI_API_BASE.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}"},
            json={"model": settings.OPENAI_MODEL, "messages": messages, "temperature": 0.2},
            timeout=settings.AGENT_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        return str(resp.json()["choices"][0]["message"]["content"]).strip()

    # Deterministic offline mock mode for CPU-only / no-API-key testing
    if "sub-queries" in prompt.lower() or "plan" in system_prompt.lower():
        topic = prompt.splitlines()[-1][:120]
        return f"1. {topic} overview and architecture\n2. {topic} benchmarks and limitations\n3. {topic} production best practices"
    if "extract" in system_prompt.lower():
        return f"Extracted key facts from evidence:\n{prompt[:400]}"
    if "critic" in system_prompt.lower() or "fact-check" in system_prompt.lower():
        return "VERIFIED: Claims align with retrieved sources. Interpretations are clearly separated from verified facts."
    return f"Structured synthesis based on verified evidence:\n{prompt[:500]}"
