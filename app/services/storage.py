"""SQLite-хранилище AI Radar.

Реализовано на aiosqlite с простой схемой и миграциями через
`CREATE TABLE IF NOT EXISTS`. Все runtime-настройки (мастер-пароль,
порог VETO, список LLM-провайдеров, API-ключи, папки, файлы, отчёты
сканирований) живут здесь.
"""

from __future__ import annotations

import json
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import aiosqlite

from app.config import PROJECT_ROOT, get_settings


# --------------------------------------------------------------------------- #
#  Схема                                                                       #
# --------------------------------------------------------------------------- #

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_providers (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL,
    base_url   TEXT NOT NULL,
    model      TEXT NOT NULL,
    api_key    TEXT NOT NULL,
    role       TEXT NOT NULL DEFAULT 'ensemble',  -- 'ensemble' | 'judge'
    enabled    INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    UNIQUE(name)
);

CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    login           TEXT NOT NULL UNIQUE,
    password_hash   TEXT NOT NULL,           -- sha256(salt + password)
    password_salt   TEXT NOT NULL,
    role            TEXT NOT NULL DEFAULT 'teacher',  -- 'admin' | 'teacher'
    display_name    TEXT,
    created_at      TEXT NOT NULL,
    last_login_at   TEXT
);

CREATE TABLE IF NOT EXISTS api_keys (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT NOT NULL,
    key_prefix   TEXT NOT NULL,
    key_hash     TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    last_used_at TEXT,
    revoked      INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS folders (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL,
    parent_id  INTEGER,
    owner_id   INTEGER,  -- id пользователя-владельца (NULL = admin/shared)
    created_at TEXT NOT NULL,
    FOREIGN KEY(parent_id) REFERENCES folders(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS files (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    folder_id       INTEGER,
    name            TEXT NOT NULL,
    mime            TEXT,
    size_bytes      INTEGER,
    text            TEXT,
    created_at      TEXT NOT NULL,
    FOREIGN KEY(folder_id) REFERENCES folders(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS scan_reports (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id      INTEGER,
    source       TEXT NOT NULL,         -- 'quick' | 'deep'
    title        TEXT NOT NULL,
    text         TEXT NOT NULL,
    ai_score     INTEGER NOT NULL,
    stat_score   INTEGER NOT NULL,
    veto         INTEGER NOT NULL,
    veto_reason  TEXT,
    integrity    TEXT NOT NULL,
    judge_status TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    fragments_json TEXT NOT NULL,
    judge_raw_json TEXT,
    created_at   TEXT NOT NULL,
    FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS scan_tasks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    title         TEXT NOT NULL,
    source        TEXT NOT NULL,          -- 'quick' | 'deep'
    status        TEXT NOT NULL DEFAULT 'pending',  -- pending|running|done|error
    progress      INTEGER NOT NULL DEFAULT 0,       -- 0..100
    progress_msg  TEXT,
    steps_json    TEXT,                   -- JSON array of step states
    text          TEXT,                   -- текст работы (для quick)
    file_id       INTEGER,               -- для deep
    report_id     INTEGER,               -- заполнится при done
    error         TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE SET NULL
);
"""


# --------------------------------------------------------------------------- #
#  Pydantic-модели для API                                                      #
# --------------------------------------------------------------------------- #

from pydantic import BaseModel, Field


class LlmProviderIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    base_url: str = Field(..., min_length=1)
    model: str = Field(..., min_length=1)
    api_key: str = Field(..., min_length=1)
    role: str = Field(default="ensemble", pattern="^(ensemble|judge)$")
    enabled: bool = True


class LlmProviderOut(BaseModel):
    id: int
    name: str
    base_url: str
    model: str
    api_key_masked: str
    role: str  # 'ensemble' | 'judge'
    enabled: bool
    created_at: str


class ApiKeyOut(BaseModel):
    id: int
    name: str
    key_prefix: str
    created_at: str
    last_used_at: str | None
    revoked: bool


class ApiKeyCreated(BaseModel):
    id: int
    name: str
    key: str
    created_at: str


class ScanReportOut(BaseModel):
    id: int | None
    source: str
    title: str
    ai_score: int
    stat_score: int
    veto: bool
    veto_reason: str | None
    academic_integrity: str
    judge_status: str
    metrics: dict[str, Any]
    suspicious_fragments: list[str]
    text_preview: str
    created_at: str


# --------------------------------------------------------------------------- #
#  Dataclass-обёртки для внутренней логики                                       #
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class LlmProvider:
    id: int
    name: str
    base_url: str
    model: str
    api_key: str
    role: str  # 'ensemble' | 'judge'
    enabled: bool
    created_at: str


@dataclass(slots=True)
class ApiKeyRecord:
    id: int
    name: str
    key_prefix: str
    key_hash: str
    created_at: str
    last_used_at: str | None
    revoked: bool


@dataclass(slots=True)
class User:
    id: int
    login: str
    password_hash: str
    password_salt: str
    role: str  # 'admin' | 'teacher'
    display_name: str | None
    created_at: str
    last_login_at: str | None


class UserOut(BaseModel):
    id: int
    login: str
    role: str
    display_name: str | None
    created_at: str
    last_login_at: str | None


class UserCreateIn(BaseModel):
    login: str = Field(..., min_length=2, max_length=60)
    password: str = Field(..., min_length=4, max_length=200)
    role: str = Field(default="teacher", pattern="^(admin|teacher)$")
    display_name: str | None = None


class PasswordResetIn(BaseModel):
    new_password: str = Field(..., min_length=4, max_length=200)


# --------------------------------------------------------------------------- #
#  Хранилище                                                                    #
# --------------------------------------------------------------------------- #


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _mask(key: str) -> str:
    """Маска для отображения API-ключа в UI."""
    if len(key) <= 8:
        return "*" * len(key)
    return key[:4] + "*" * (len(key) - 8) + key[-4:]


def _hash_key(raw: str) -> str:
    import hashlib

    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _hash_password(password: str, salt: str) -> str:
    """SHA-256(salt + password) — простая, но достаточная для локального деплоя."""
    import hashlib

    return hashlib.sha256(f"{salt}:{password}".encode("utf-8")).hexdigest()


def _verify_password(password: str, salt: str, expected_hash: str) -> bool:
    import hmac

    actual = _hash_password(password, salt)
    return hmac.compare_digest(actual, expected_hash)


def _gen_api_key() -> str:
    return "air_" + secrets.token_hex(16)


def _gen_prefix(raw: str) -> str:
    return raw[:8] + "..." if len(raw) > 8 else raw


class Storage:
    """Асинхронное SQLite-хранилище."""

    def __init__(self, db_uri: str | None = None) -> None:
        self.db_uri = db_uri or get_settings().database_uri
        self._conn: aiosqlite.Connection | None = None

    # --- connection lifecycle ---

    async def init(self) -> None:
        Path(self.db_uri).parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.db_uri)
        self._conn.row_factory = sqlite3.Row
        await self._conn.executescript(SCHEMA_SQL)
        await self._conn.commit()
        await self._apply_migrations()
        await self._ensure_default_settings()
        await self._conn.commit()

    async def _apply_migrations(self) -> None:
        """Простые миграции для совместимости со старыми базами."""
        # Добавить колонку role в llm_providers, если её нет (старые базы)
        cur = await self.conn.execute("PRAGMA table_info(llm_providers)")
        cols = {row["name"] for row in await cur.fetchall()}
        if "role" not in cols:
            try:
                await self.conn.execute(
                    "ALTER TABLE llm_providers ADD COLUMN role TEXT NOT NULL DEFAULT 'ensemble'"
                )
                await self.conn.commit()
            except Exception:
                pass

        # Добавить колонку owner_id в folders, если её нет
        cur = await self.conn.execute("PRAGMA table_info(folders)")
        cols = {row["name"] for row in await cur.fetchall()}
        if "owner_id" not in cols:
            try:
                await self.conn.execute(
                    "ALTER TABLE folders ADD COLUMN owner_id INTEGER"
                )
                await self.conn.commit()
            except Exception:
                pass

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Storage not initialized. Call init() first.")
        return self._conn

    async def _ensure_default_settings(self) -> None:
        """Записать дефолты из .env в SQLite при первом старте."""
        s = get_settings()
        defaults: dict[str, str] = {
            "master_password": s.master_password,
            "veto_threshold": str(s.veto_threshold),
            "judge_mode": s.judge_mode,
            "judge_max_retries": str(s.judge_max_retries),
            "judge_http_timeout_sec": str(s.judge_http_timeout_sec),
            "docs_enabled": "true" if s.docs_enabled else "false",
        }
        for k, v in defaults.items():
            await self.conn.execute(
                "INSERT OR IGNORE INTO settings(key, value) VALUES(?, ?)", (k, v)
            )

        # Дефолтный LLM-провайдер из .env (если задан) — роль 'ensemble'
        if s.default_llm_base_url and s.default_llm_model and s.default_llm_api_key:
            existing = await self.conn.execute(
                "SELECT id FROM llm_providers WHERE name = ?", ("default",)
            )
            if (await existing.fetchone()) is None:
                await self.conn.execute(
                    "INSERT INTO llm_providers(name, base_url, model, api_key, role, enabled, created_at) "
                    "VALUES(?, ?, ?, ?, 'ensemble', 1, ?)",
                    (
                        "default",
                        s.default_llm_base_url,
                        s.default_llm_model,
                        s.default_llm_api_key,
                        _now(),
                    ),
                )

        # Дефолтный admin-пользователь при первом старте
        existing_user = await self.conn.execute(
            "SELECT id FROM users WHERE login = ?", ("admin",)
        )
        if (await existing_user.fetchone()) is None:
            salt = secrets.token_hex(8)
            master = s.master_password or "airadar"
            await self.conn.execute(
                "INSERT INTO users(login, password_hash, password_salt, role, display_name, created_at) "
                "VALUES(?, ?, ?, 'admin', 'Администратор', ?)",
                ("admin", _hash_password(master, salt), salt, _now()),
            )

    # --- settings (key-value) ---

    async def get_setting(self, key: str, default: str | None = None) -> str | None:
        cur = await self.conn.execute("SELECT value FROM settings WHERE key = ?", (key,))
        row = await cur.fetchone()
        return row["value"] if row else default

    async def set_setting(self, key: str, value: str) -> None:
        await self.conn.execute(
            "INSERT INTO settings(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        await self.conn.commit()

    async def get_all_settings(self) -> dict[str, str]:
        cur = await self.conn.execute("SELECT key, value FROM settings")
        rows = await cur.fetchall()
        return {r["key"]: r["value"] for r in rows}

    async def get_master_password(self) -> str:
        return (await self.get_setting("master_password")) or get_settings().master_password

    async def set_master_password(self, new_password: str) -> None:
        await self.set_setting("master_password", new_password)

    async def get_veto_threshold(self) -> int:
        v = await self.get_setting("veto_threshold")
        return int(v) if v else get_settings().veto_threshold

    async def set_veto_threshold(self, value: int) -> None:
        await self.set_setting("veto_threshold", str(value))

    async def get_judge_max_retries(self) -> int:
        v = await self.get_setting("judge_max_retries")
        return int(v) if v is not None else get_settings().judge_max_retries

    # --- LLM providers ---

    async def list_llm_providers(self, role: str | None = None) -> list[LlmProvider]:
        if role:
            cur = await self.conn.execute(
                "SELECT id, name, base_url, model, api_key, role, enabled, created_at "
                "FROM llm_providers WHERE role=? ORDER BY id",
                (role,),
            )
        else:
            cur = await self.conn.execute(
                "SELECT id, name, base_url, model, api_key, role, enabled, created_at "
                "FROM llm_providers ORDER BY id"
            )
        rows = await cur.fetchall()
        return [LlmProvider(**dict(r)) for r in rows]

    async def list_enabled_llm_providers(self, role: str | None = None) -> list[LlmProvider]:
        return [p for p in await self.list_llm_providers(role) if p.enabled]

    async def add_llm_provider(self, p: LlmProviderIn) -> LlmProvider:
        created = _now()
        cur = await self.conn.execute(
            "INSERT INTO llm_providers(name, base_url, model, api_key, role, enabled, created_at) "
            "VALUES(?, ?, ?, ?, ?, ?, ?)",
            (p.name, p.base_url, p.model, p.api_key, p.role, int(p.enabled), created),
        )
        await self.conn.commit()
        return LlmProvider(
            id=cur.lastrowid,
            name=p.name,
            base_url=p.base_url,
            model=p.model,
            api_key=p.api_key,
            role=p.role,
            enabled=p.enabled,
            created_at=created,
        )

    async def update_llm_provider(self, provider_id: int, p: LlmProviderIn) -> LlmProvider | None:
        await self.conn.execute(
            "UPDATE llm_providers SET name=?, base_url=?, model=?, api_key=?, role=?, enabled=? WHERE id=?",
            (p.name, p.base_url, p.model, p.api_key, p.role, int(p.enabled), provider_id),
        )
        await self.conn.commit()
        cur = await self.conn.execute(
            "SELECT id, name, base_url, model, api_key, role, enabled, created_at FROM llm_providers WHERE id=?",
            (provider_id,),
        )
        row = await cur.fetchone()
        return LlmProvider(**dict(row)) if row else None

    async def delete_llm_provider(self, provider_id: int) -> None:
        await self.conn.execute("DELETE FROM llm_providers WHERE id=?", (provider_id,))
        await self.conn.commit()

    async def set_llm_provider_enabled(self, provider_id: int, enabled: bool) -> None:
        await self.conn.execute(
            "UPDATE llm_providers SET enabled=? WHERE id=?", (int(enabled), provider_id)
        )
        await self.conn.commit()

    # --- API keys ---

    async def create_api_key(self, name: str) -> tuple[ApiKeyRecord, str]:
        raw = _gen_api_key()
        prefix = _gen_prefix(raw)
        h = _hash_key(raw)
        created = _now()
        cur = await self.conn.execute(
            "INSERT INTO api_keys(name, key_prefix, key_hash, created_at) VALUES(?, ?, ?, ?)",
            (name, prefix, h, created),
        )
        await self.conn.commit()
        rec = ApiKeyRecord(
            id=cur.lastrowid,
            name=name,
            key_prefix=prefix,
            key_hash=h,
            created_at=created,
            last_used_at=None,
            revoked=False,
        )
        return rec, raw

    async def list_api_keys(self) -> list[ApiKeyRecord]:
        cur = await self.conn.execute(
            "SELECT id, name, key_prefix, key_hash, created_at, last_used_at, revoked "
            "FROM api_keys ORDER BY id DESC"
        )
        rows = await cur.fetchall()
        return [
            ApiKeyRecord(
                id=r["id"],
                name=r["name"],
                key_prefix=r["key_prefix"],
                key_hash=r["key_hash"],
                created_at=r["created_at"],
                last_used_at=r["last_used_at"],
                revoked=bool(r["revoked"]),
            )
            for r in rows
        ]

    async def verify_api_key(self, raw: str) -> ApiKeyRecord | None:
        """Вернуть запись о валидном (не отозванном) ключе или None."""
        if not raw.startswith("air_"):
            return None
        h = _hash_key(raw)
        cur = await self.conn.execute(
            "SELECT id, name, key_prefix, key_hash, created_at, last_used_at, revoked "
            "FROM api_keys WHERE key_hash = ?",
            (h,),
        )
        row = await cur.fetchone()
        if not row or row["revoked"]:
            return None
        await self.conn.execute(
            "UPDATE api_keys SET last_used_at = ? WHERE id = ?", (_now(), row["id"])
        )
        await self.conn.commit()
        return ApiKeyRecord(
            id=row["id"],
            name=row["name"],
            key_prefix=row["key_prefix"],
            key_hash=row["key_hash"],
            created_at=row["created_at"],
            last_used_at=row["last_used_at"],
            revoked=bool(row["revoked"]),
        )

    async def revoke_api_key(self, key_id: int) -> None:
        await self.conn.execute("UPDATE api_keys SET revoked=1 WHERE id=?", (key_id,))
        await self.conn.commit()

    async def rotate_api_key(self, key_id: int) -> tuple[ApiKeyRecord, str] | None:
        await self.revoke_api_key(key_id)
        cur = await self.conn.execute(
            "SELECT name FROM api_keys WHERE id=?", (key_id,)
        )
        row = await cur.fetchone()
        if not row:
            return None
        return await self.create_api_key(row["name"])

    # --- users ---

    async def list_users(self) -> list[User]:
        cur = await self.conn.execute(
            "SELECT id, login, password_hash, password_salt, role, display_name, "
            "created_at, last_login_at FROM users ORDER BY id"
        )
        return [User(**dict(r)) for r in await cur.fetchall()]

    async def get_user_by_login(self, login: str) -> User | None:
        cur = await self.conn.execute(
            "SELECT id, login, password_hash, password_salt, role, display_name, "
            "created_at, last_login_at FROM users WHERE login = ?",
            (login,),
        )
        row = await cur.fetchone()
        return User(**dict(row)) if row else None

    async def get_user(self, user_id: int) -> User | None:
        cur = await self.conn.execute(
            "SELECT id, login, password_hash, password_salt, role, display_name, "
            "created_at, last_login_at FROM users WHERE id = ?",
            (user_id,),
        )
        row = await cur.fetchone()
        return User(**dict(row)) if row else None

    async def verify_user(self, login: str, password: str) -> User | None:
        user = await self.get_user_by_login(login)
        if not user:
            return None
        if not _verify_password(password, user.password_salt, user.password_hash):
            return None
        await self.conn.execute(
            "UPDATE users SET last_login_at = ? WHERE id = ?", (_now(), user.id)
        )
        await self.conn.commit()
        return user

    async def create_user(self, login: str, password: str, role: str = "teacher",
                          display_name: str | None = None) -> User:
        salt = secrets.token_hex(8)
        created = _now()
        cur = await self.conn.execute(
            "INSERT INTO users(login, password_hash, password_salt, role, display_name, created_at) "
            "VALUES(?, ?, ?, ?, ?, ?)",
            (login, _hash_password(password, salt), salt, role, display_name, created),
        )
        await self.conn.commit()
        return await self.get_user(cur.lastrowid)  # type: ignore[return-value]

    async def set_user_password(self, user_id: int, new_password: str) -> None:
        salt = secrets.token_hex(8)
        await self.conn.execute(
            "UPDATE users SET password_hash = ?, password_salt = ? WHERE id = ?",
            (_hash_password(new_password, salt), salt, user_id),
        )
        await self.conn.commit()

    async def delete_user(self, user_id: int) -> None:
        await self.conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        await self.conn.commit()

    # --- folders / files ---

    async def list_folders(self, parent_id: int | None = None,
                           owner_id: int | None = None,
                           include_shared: bool = True) -> list[dict[str, Any]]:
        """Список папок. owner_id ограничивает до папок владельца + shared."""
        conditions = []
        params: list = []
        if parent_id is None:
            conditions.append("parent_id IS NULL")
        else:
            conditions.append("parent_id = ?")
            params.append(parent_id)
        if owner_id is not None:
            if include_shared:
                conditions.append("(owner_id = ? OR owner_id IS NULL)")
                params.append(owner_id)
            else:
                conditions.append("owner_id = ?")
                params.append(owner_id)
        where = " AND ".join(conditions)
        cur = await self.conn.execute(
            f"SELECT id, name, parent_id, owner_id, created_at FROM folders WHERE {where} ORDER BY name",
            params,
        )
        return [dict(r) for r in await cur.fetchall()]

    async def create_folder(self, name: str, parent_id: int | None = None,
                             owner_id: int | None = None) -> dict[str, Any]:
        created = _now()
        cur = await self.conn.execute(
            "INSERT INTO folders(name, parent_id, owner_id, created_at) VALUES(?, ?, ?, ?)",
            (name, parent_id, owner_id, created),
        )
        await self.conn.commit()
        return {"id": cur.lastrowid, "name": name, "parent_id": parent_id,
                "owner_id": owner_id, "created_at": created}

    async def get_folder(self, folder_id: int) -> dict[str, Any] | None:
        cur = await self.conn.execute(
            "SELECT id, name, parent_id, owner_id, created_at FROM folders WHERE id=?",
            (folder_id,),
        )
        row = await cur.fetchone()
        return dict(row) if row else None

    async def delete_folder(self, folder_id: int) -> None:
        await self.conn.execute("DELETE FROM folders WHERE id=?", (folder_id,))
        await self.conn.commit()

    async def list_files(self, folder_id: int) -> list[dict[str, Any]]:
        cur = await self.conn.execute(
            "SELECT id, folder_id, name, mime, size_bytes, created_at FROM files WHERE folder_id=? ORDER BY name",
            (folder_id,),
        )
        return [dict(r) for r in await cur.fetchall()]

    async def add_file(
        self,
        folder_id: int,
        name: str,
        text: str,
        mime: str | None = None,
        size_bytes: int | None = None,
    ) -> dict[str, Any]:
        created = _now()
        cur = await self.conn.execute(
            "INSERT INTO files(folder_id, name, mime, size_bytes, text, created_at) VALUES(?, ?, ?, ?, ?, ?)",
            (folder_id, name, mime, size_bytes, text, created),
        )
        await self.conn.commit()
        return {
            "id": cur.lastrowid,
            "folder_id": folder_id,
            "name": name,
            "mime": mime,
            "size_bytes": size_bytes,
            "created_at": created,
        }

    async def get_file(self, file_id: int) -> dict[str, Any] | None:
        cur = await self.conn.execute(
            "SELECT id, folder_id, name, mime, size_bytes, text, created_at FROM files WHERE id=?",
            (file_id,),
        )
        row = await cur.fetchone()
        return dict(row) if row else None

    async def delete_file(self, file_id: int) -> None:
        await self.conn.execute("DELETE FROM files WHERE id=?", (file_id,))
        await self.conn.commit()

    async def list_recent_files(self, limit: int = 10) -> list[dict[str, Any]]:
        cur = await self.conn.execute(
            "SELECT id, folder_id, name, mime, size_bytes, created_at FROM files ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in await cur.fetchall()]

    async def list_recent_reports(self, limit: int = 10) -> list[dict[str, Any]]:
        cur = await self.conn.execute(
            "SELECT id, file_id, source, title, ai_score, stat_score, veto, veto_reason, "
            "integrity, judge_status, created_at FROM scan_reports ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in await cur.fetchall()]

    async def add_scan_report(self, data: ScanReportOut) -> int:
        cur = await self.conn.execute(
            "INSERT INTO scan_reports(file_id, source, title, text, ai_score, stat_score, "
            "veto, veto_reason, integrity, judge_status, metrics_json, fragments_json, "
            "judge_raw_json, created_at) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                data.id,
                data.source,
                data.title,
                data.text_preview,
                data.ai_score,
                data.stat_score,
                int(data.veto),
                data.veto_reason,
                data.academic_integrity,
                data.judge_status,
                json.dumps(data.metrics, ensure_ascii=False),
                json.dumps(data.suspicious_fragments, ensure_ascii=False),
                json.dumps(data.metrics.get("signal2_panel", {}), ensure_ascii=False) if data.metrics else None,
                data.created_at,
            ),
        )
        await self.conn.commit()
        return cur.lastrowid

    async def get_scan_report(self, report_id: int) -> dict[str, Any] | None:
        """Полная запись отчёта по id (для страницы детализации)."""
        cur = await self.conn.execute(
            "SELECT id, file_id, source, title, text, ai_score, stat_score, "
            "veto, veto_reason, integrity, judge_status, metrics_json, "
            "fragments_json, judge_raw_json, created_at FROM scan_reports WHERE id=?",
            (report_id,),
        )
        row = await cur.fetchone()
        if not row:
            return None
        d = dict(row)
        try:
            d["metrics"] = json.loads(d.pop("metrics_json")) if d.get("metrics_json") else {}
        except json.JSONDecodeError:
            d["metrics"] = {}
            d.pop("metrics_json", None)
        try:
            d["suspicious_fragments"] = json.loads(d.pop("fragments_json")) if d.get("fragments_json") else []
        except json.JSONDecodeError:
            d["suspicious_fragments"] = []
            d.pop("fragments_json", None)
        try:
            d["judge_raw"] = json.loads(d.pop("judge_raw_json")) if d.get("judge_raw_json") else None
        except json.JSONDecodeError:
            d["judge_raw"] = None
            d.pop("judge_raw_json", None)
        return d

    # --- scan tasks (асинхронные фоновые проверки) ---

    async def create_scan_task(self, *, title: str, source: str, text: str | None = None,
                                 file_id: int | None = None) -> dict[str, Any]:
        created = _now()
        cur = await self.conn.execute(
            "INSERT INTO scan_tasks(title, source, status, progress, progress_msg, "
            "text, file_id, created_at, updated_at) VALUES(?, ?, 'pending', 0, ?, ?, ?, ?, ?)",
            (title, source, "Ожидание...", text, file_id, created, created),
        )
        await self.conn.commit()
        return {"id": cur.lastrowid, "title": title, "source": source,
                "status": "pending", "progress": 0, "progress_msg": "Ожидание...",
                "created_at": created}

    async def update_scan_task(self, task_id: int, *, status: str | None = None,
                                 progress: int | None = None, progress_msg: str | None = None,
                                 steps_json: str | None = None, report_id: int | None = None,
                                 error: str | None = None) -> None:
        sets = []
        params: list = []
        if status is not None:
            sets.append("status = ?"); params.append(status)
        if progress is not None:
            sets.append("progress = ?"); params.append(progress)
        if progress_msg is not None:
            sets.append("progress_msg = ?"); params.append(progress_msg)
        if steps_json is not None:
            sets.append("steps_json = ?"); params.append(steps_json)
        if report_id is not None:
            sets.append("report_id = ?"); params.append(report_id)
        if error is not None:
            sets.append("error = ?"); params.append(error)
        if not sets:
            return
        sets.append("updated_at = ?"); params.append(_now())
        params.append(task_id)
        await self.conn.execute(
            f"UPDATE scan_tasks SET {', '.join(sets)} WHERE id = ?", params
        )
        await self.conn.commit()

    async def get_scan_task(self, task_id: int) -> dict[str, Any] | None:
        cur = await self.conn.execute(
            "SELECT id, title, source, status, progress, progress_msg, steps_json, "
            "text, file_id, report_id, error, created_at, updated_at FROM scan_tasks WHERE id = ?",
            (task_id,),
        )
        row = await cur.fetchone()
        if not row:
            return None
        d = dict(row)
        try:
            d["steps"] = json.loads(d.pop("steps_json") or "[]")
        except (json.JSONDecodeError, TypeError):
            d["steps"] = []
            d.pop("steps_json", None)
        return d

    async def list_scan_tasks(self, limit: int = 20) -> list[dict[str, Any]]:
        cur = await self.conn.execute(
            "SELECT id, title, source, status, progress, progress_msg, file_id, "
            "report_id, error, created_at, updated_at FROM scan_tasks "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in await cur.fetchall()]


# --------------------------------------------------------------------------- #
#  Module-level singleton                                                       #
# --------------------------------------------------------------------------- #


_storage: Storage | None = None


async def get_storage() -> Storage:
    global _storage
    if _storage is None:
        _storage = Storage()
        await _storage.init()
    return _storage


async def close_storage() -> None:
    global _storage
    if _storage is not None:
        await _storage.close()
        _storage = None


def reset_storage_for_tests() -> None:
    """Сбросить синглтон (используется в тестах)."""
    global _storage
    _storage = None
