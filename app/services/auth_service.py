"""Сервис авторизации AI Radar.

Поддерживает два типа учётных данных:
  * **Session-токены** (JWT access + refresh) — для UI-сессий после ввода
    мастер-пароля.
  * **API-ключи** (`air_<32hex>`) — для программного доступа через
    `Authorization: Bearer air_...` или `X-API-Key: air_...`.

Унифицированная FastAPI-dependency `current_principal` различает формат
токена и возвращает объект `Principal` с информацией о текущем субъекте
доступа.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Literal

from fastapi import Depends, Header, HTTPException, Request, status

from app.config import get_settings
from app.services.storage import ApiKeyRecord, Storage, get_storage


# --------------------------------------------------------------------------- #
#  JWT (HMAC-SHA256)                                                            #
# --------------------------------------------------------------------------- #


def _b64encode(raw: bytes) -> str:
    import base64

    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64decode(s: str) -> bytes:
    import base64

    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def _sign(payload: dict, secret: str) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    h = _b64encode(json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode())
    p = _b64encode(json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode())
    signing_input = f"{h}.{p}".encode()
    sig = hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()
    return f"{h}.{p}.{_b64encode(sig)}"


def _verify(token: str, secret: str) -> dict:
    try:
        h_b64, p_b64, sig_b64 = token.split(".")
    except ValueError as exc:
        raise ValueError("malformed JWT") from exc
    signing_input = f"{h_b64}.{p_b64}".encode()
    expected = hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()
    actual = _b64decode(sig_b64)
    if not hmac.compare_digest(expected, actual):
        raise ValueError("bad signature")
    payload = json.loads(_b64decode(p_b64))
    if "exp" in payload and int(payload["exp"]) < int(time.time()):
        raise ValueError("token expired")
    return payload


def _make_session( principal_id: str, kind: Literal["access", "refresh"], ttl_sec: int
) -> str:
    s = get_settings()
    payload = {
        "sub": principal_id,
        "typ": kind,
        "iat": int(time.time()),
        "exp": int(time.time()) + ttl_sec,
    }
    return _sign(payload, s.jwt_secret)


# --------------------------------------------------------------------------- #
#  Public API                                                                  #
# --------------------------------------------------------------------------- #


async def issue_session_tokens(principal_id: str = "admin", role: str = "admin",
                                 user_id: int | None = None) -> dict:
    """Сгенерировать пару access+refresh JWT-токенов для UI-сессии."""
    s = get_settings()
    payload_access = {
        "sub": principal_id,
        "typ": "access",
        "role": role,
        "uid": user_id,
        "iat": int(time.time()),
        "exp": int(time.time()) + s.access_token_ttl_min * 60,
    }
    payload_refresh = {
        "sub": principal_id,
        "typ": "refresh",
        "role": role,
        "uid": user_id,
        "iat": int(time.time()),
        "exp": int(time.time()) + s.refresh_token_ttl_days * 86400,
    }
    return {
        "access_token": _sign(payload_access, s.jwt_secret),
        "refresh_token": _sign(payload_refresh, s.jwt_secret),
        "token_type": "Bearer",
        "expires_in": s.access_token_ttl_min * 60,
        "role": role,
    }


async def verify_master_password(password: str, storage: Storage) -> bool:
    expected = await storage.get_master_password()
    return hmac.compare_digest(password or "", expected or "")


def decode_access_token(token: str) -> dict:
    s = get_settings()
    payload = _verify(token, s.jwt_secret)
    if payload.get("typ") != "access":
        raise ValueError("not an access token")
    return payload


def decode_refresh_token(token: str) -> dict:
    s = get_settings()
    payload = _verify(token, s.jwt_secret)
    if payload.get("typ") != "refresh":
        raise ValueError("not a refresh token")
    return payload


# --------------------------------------------------------------------------- #
#  Principal / dependency                                                      #
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Principal:
    """Унифицированный субъект доступа (session или api_key)."""

    type: str  # 'session' | 'api_key'
    name: str
    user_id: int | None = None
    role: str = "admin"  # 'admin' | 'teacher'
    is_admin: bool = True

    @property
    def display(self) -> str:
        return f"{self.type}:{self.name}"


async def _extract_token(
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> str | None:
    if x_api_key:
        return x_api_key.strip()
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token:
            return None
        return token.strip()
    return None


async def current_principal(
    request: Request,
    token: str | None = Depends(_extract_token),
    storage: Storage = Depends(get_storage),
) -> Principal:
    """FastAPI dependency: валидирует token/api-key и возвращает Principal."""
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Требуется авторизация: Bearer <token> или X-API-Key: air_...",
            headers={"WWW-Authenticate": 'Bearer realm="ai-radar"'},
        )
    # 1) Попытка валидировать как JWT-сессию
    if token.count(".") == 2:
        try:
            payload = decode_access_token(token)
            role = payload.get("role", "admin")
            uid = payload.get("uid")
            request.state.principal = Principal(
                type="session",
                name=payload.get("sub", "session"),
                user_id=uid,
                role=role,
                is_admin=(role == "admin"),
            )
            return request.state.principal
        except ValueError:
            pass  # не JWT или невалидный — пробуем как API-ключ
    # 2) Попытка валидировать как API-ключ
    if token.startswith("air_"):
        rec: ApiKeyRecord | None = await storage.verify_api_key(token)
        if rec is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API-ключ недействителен или отозван",
                headers={"WWW-Authenticate": 'Bearer realm="ai-radar"'},
            )
        # API-ключи имеют admin-права (для программного доступа)
        request.state.principal = Principal(
            type="api_key", name=rec.name, user_id=None, role="admin", is_admin=True
        )
        return request.state.principal
    # 3) Ни то, ни другое
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Неизвестный формат токена",
        headers={"WWW-Authenticate": 'Bearer realm="ai-radar"'},
    )


async def require_admin(principal: Principal = Depends(current_principal)) -> Principal:
    """Dependency: требует админ-прав (пока все — админы)."""
    if not principal.is_admin:
        raise HTTPException(status_code=403, detail="Требуется администратор")
    return principal
