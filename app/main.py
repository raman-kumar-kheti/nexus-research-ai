"""FastAPI application for NexusResearch AI."""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from app.agents.nexus_graph import resume_research_workflow, run_research_workflow
from app.auth.auth import create_token, decode_token, hash_password, verify_password
from app.config import settings
from app.core.evaluator import evaluate_workflow_output
from app.memory.database import (
    create_user,
    delete_session,
    get_all_history,
    get_chat_history,
    get_user,
    init_db,
    list_sessions,
    save_chat_message,
)
from app.tools.doc_generator import create_markdown, create_pdf

security = HTTPBearer(auto_error=False)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Multi-agent research and automation platform powered by LangGraph and MCP.",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def global_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": f"Internal error: {exc}"})


def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(security),
) -> str:
    if creds is None:
        return "anonymous"
    username = decode_token(creds.credentials)
    if not username:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")
    return username


class AuthRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=6, max_length=128)


class ResearchRequest(BaseModel):
    topic: str = Field(min_length=3, max_length=1000)
    thread_id: str = Field(default="")
    require_approval: bool = False
    max_iterations: int = Field(default=2, ge=1, le=5)


class ApprovalRequest(BaseModel):
    thread_id: str
    approved: bool = True


class ExportRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1)
    format: Literal["markdown", "pdf"] = "markdown"


@app.get("/", tags=["System"])
@app.get("/health", tags=["System"])
def health() -> dict[str, str]:
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "llm_provider": settings.LLM_PROVIDER,
    }


@app.post("/auth/register", tags=["Auth"])
def register(req: AuthRequest) -> dict[str, str]:
    if not create_user(req.username, hash_password(req.password)):
        raise HTTPException(status_code=409, detail="Username already exists.")
    return {"access_token": create_token(req.username), "token_type": "bearer"}


@app.post("/auth/login", tags=["Auth"])
def login(req: AuthRequest) -> dict[str, str]:
    user = get_user(req.username)
    if not user or not verify_password(req.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials.")
    return {"access_token": create_token(req.username), "token_type": "bearer"}


@app.post("/research", tags=["Research"])
def run_research(req: ResearchRequest, username: str = Depends(get_current_user)) -> dict[str, Any]:
    thread_id = req.thread_id or uuid.uuid4().hex[:12]
    save_chat_message(thread_id, "user", req.topic, username=username)

    result = run_research_workflow(
        topic=req.topic,
        thread_id=thread_id,
        username=username,
        require_approval=req.require_approval,
        max_iterations=req.max_iterations,
    )
    if result["status"] == "completed":
        save_chat_message(thread_id, "assistant", result["report_markdown"], username=username)
        result["evaluation"] = evaluate_workflow_output(result)
    return result


@app.post("/research/approve", tags=["Research"])
def approve_research(req: ApprovalRequest, username: str = Depends(get_current_user)) -> dict[str, Any]:
    result = resume_research_workflow(thread_id=req.thread_id, approved=req.approved)
    save_chat_message(req.thread_id, "assistant", result.get("report_markdown", ""), username=username)
    result["evaluation"] = evaluate_workflow_output(result)
    return result


@app.post("/export-report", tags=["Export"])
def export_report(req: ExportRequest) -> FileResponse:
    file_id = uuid.uuid4().hex[:8]
    if req.format == "pdf":
        out_path = settings.OUTPUTS_DIR / f"report_{file_id}.pdf"
        create_pdf(req.title, req.content, out_path)
        return FileResponse(path=str(out_path), filename=out_path.name, media_type="application/pdf")

    out_path = settings.OUTPUTS_DIR / f"report_{file_id}.md"
    create_markdown(req.title, req.content, out_path)
    return FileResponse(path=str(out_path), filename=out_path.name, media_type="text/markdown")


@app.get("/history", tags=["Memory"])
def history(limit: int = 20, username: str = Depends(get_current_user)) -> list[dict[str, Any]]:
    return get_all_history(username=username, limit=limit)


@app.get("/sessions", tags=["Sessions"])
def sessions(limit: int = 20, username: str = Depends(get_current_user)) -> list[dict[str, Any]]:
    return list_sessions(username=username, limit=limit)


@app.get("/chat/{session_id}/history", tags=["Sessions"])
def session_history(session_id: str, username: str = Depends(get_current_user)) -> list[dict[str, Any]]:
    return get_chat_history(session_id=session_id, username=username)


@app.delete("/chat/{session_id}", tags=["Sessions"])
def remove_session(session_id: str, username: str = Depends(get_current_user)) -> dict[str, int | str]:
    deleted = delete_session(session_id=session_id, username=username)
    return {"status": "deleted", "deleted_rows": deleted}
