"""Database persistence supporting PostgreSQL and SQLite with proper indexing."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone

from app.config import settings

DATABASE_URL = os.getenv("DATABASE_URL", settings.DATABASE_URL)


def is_postgres() -> bool:
    return bool(DATABASE_URL and DATABASE_URL.startswith("postgres"))


def get_connection():
    if is_postgres():
        import psycopg2

        return psycopg2.connect(DATABASE_URL)
    conn = sqlite3.connect(str(settings.DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def ph() -> str:
    return "%s" if is_postgres() else "?"


def serial() -> str:
    return "SERIAL PRIMARY KEY" if is_postgres() else "INTEGER PRIMARY KEY AUTOINCREMENT"


def init_db() -> None:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS users (
            id {serial()},
            username TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS research_history (
            id {serial()},
            username TEXT NOT NULL DEFAULT 'anonymous',
            query TEXT NOT NULL,
            summary TEXT NOT NULL,
            citations_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL
        )
        """
    )
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id {serial()},
            session_id TEXT NOT NULL,
            username TEXT NOT NULL DEFAULT 'anonymous',
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_session_user ON chat_sessions(session_id, username)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_research_user ON research_history(username)")
    conn.commit()
    conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_user(username: str, hashed_password: str) -> bool:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            f"INSERT INTO users (username, hashed_password, created_at) VALUES ({ph()}, {ph()}, {ph()})",
            (username, hashed_password, _now()),
        )
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()


def get_user(username: str) -> dict | None:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(f"SELECT id, username, hashed_password FROM users WHERE username = {ph()}", (username,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {"id": row[0], "username": row[1], "hashed_password": row[2]}


def save_research(query: str, summary: str, username: str = "anonymous", citations_json: str = "[]") -> None:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        f"INSERT INTO research_history (username, query, summary, citations_json, created_at) VALUES ({ph()}, {ph()}, {ph()}, {ph()}, {ph()})",
        (username, query, summary, citations_json, _now()),
    )
    conn.commit()
    conn.close()


def search_history(query: str, username: str = "anonymous") -> list[dict]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        f"SELECT query, summary, created_at FROM research_history WHERE username = {ph()} AND query LIKE {ph()} ORDER BY id DESC LIMIT 3",
        (username, f"%{query[:80]}%"),
    )
    rows = cur.fetchall()
    conn.close()
    return [{"query": r[0], "summary": r[1], "created_at": r[2]} for r in rows]


def get_all_history(username: str = "anonymous", limit: int = 20) -> list[dict]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        f"SELECT id, query, summary, citations_json, created_at FROM research_history WHERE username = {ph()} ORDER BY id DESC LIMIT {ph()}",
        (username, limit),
    )
    rows = cur.fetchall()
    conn.close()
    return [
        {"id": r[0], "query": r[1], "summary": r[2], "citations_json": r[3], "created_at": r[4]}
        for r in rows
    ]


def save_chat_message(session_id: str, role: str, content: str, username: str = "anonymous") -> None:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        f"INSERT INTO chat_sessions (session_id, username, role, content, created_at) VALUES ({ph()}, {ph()}, {ph()}, {ph()}, {ph()})",
        (session_id, username, role, content, _now()),
    )
    conn.commit()
    conn.close()


def get_chat_history(session_id: str, username: str = "anonymous", limit: int = 20) -> list[dict]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        f"SELECT role, content, created_at FROM chat_sessions WHERE session_id = {ph()} AND username = {ph()} ORDER BY id DESC LIMIT {ph()}",
        (session_id, username, limit),
    )
    rows = cur.fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1], "created_at": r[2]} for r in reversed(rows)]


def list_sessions(username: str = "anonymous", limit: int = 20) -> list[dict]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        f"SELECT session_id, MAX(created_at) as last_active, COUNT(*) as msg_count FROM chat_sessions WHERE username = {ph()} GROUP BY session_id ORDER BY last_active DESC LIMIT {ph()}",
        (username, limit),
    )
    rows = cur.fetchall()
    conn.close()
    return [{"session_id": r[0], "last_active": r[1], "message_count": r[2]} for r in rows]


def delete_session(session_id: str, username: str = "anonymous") -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        f"DELETE FROM chat_sessions WHERE session_id = {ph()} AND username = {ph()}",
        (session_id, username),
    )
    deleted = cur.rowcount
    conn.commit()
    conn.close()
    return deleted
