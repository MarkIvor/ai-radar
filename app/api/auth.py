"""Маршруты авторизации."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.services.auth_service import (
    Principal,
    current_principal,
    decode_refresh_token,
    issue_session_tokens,
    verify_master_password,
)
from app.services.storage import Storage, get_storage


router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginIn(BaseModel):
    login: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class TokenOut(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_in: int
    role: str = "admin"


class RefreshIn(BaseModel):
    refresh_token: str


class WhoamiOut(BaseModel):
    type: str
    name: str
    role: str
    is_admin: bool
    user_id: int | None = None


@router.post("/login", response_model=TokenOut)
async def login(payload: LoginIn, storage: Storage = Depends(get_storage)) -> TokenOut:
    """Войти в систему по логину и паролю."""
    user = await storage.verify_user(payload.login, payload.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный логин или пароль",
            headers={"WWW-Authenticate": 'Bearer realm="ai-radar"'},
        )
    tokens = await issue_session_tokens(
        principal_id=user.login, role=user.role, user_id=user.id
    )
    return TokenOut(**tokens)


@router.post("/refresh", response_model=TokenOut)
async def refresh(payload: RefreshIn) -> TokenOut:
    """Обновить access-токен по refresh-токену."""
    try:
        claims = decode_refresh_token(payload.refresh_token)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Невалидный refresh-токен: {exc}",
        )
    tokens = await issue_session_tokens(
        principal_id=claims.get("sub", "admin"),
        role=claims.get("role", "admin"),
        user_id=claims.get("uid"),
    )
    return TokenOut(**tokens)


@router.post("/logout", status_code=204)
async def logout(principal: Principal = Depends(current_principal)) -> None:
    """Выход из системы (stateless — клиент просто удаляет токен)."""
    _ = principal
    return None


@router.get("/whoami", response_model=WhoamiOut)
async def whoami(principal: Principal = Depends(current_principal)) -> WhoamiOut:
    """Диагностический эндпоинт: тип и имя текущего субъекта доступа."""
    return WhoamiOut(
        type=principal.type,
        name=principal.name,
        role=principal.role,
        is_admin=principal.is_admin,
        user_id=principal.user_id,
    )
