"""Админ-маршруты: настройки, LLM-провайдеры, API-ключи."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.services.auth_service import Principal, current_principal, require_admin
from app.services.storage import (
    ApiKeyCreated,
    ApiKeyOut,
    LlmProviderIn,
    LlmProviderOut,
    PasswordResetIn,
    Storage,
    UserCreateIn,
    UserOut,
    get_storage,
)


router = APIRouter(prefix="/api/admin", tags=["admin"])


# --------------------------------------------------------------------------- #
#  Настройки                                                                    #
# --------------------------------------------------------------------------- #


class SettingsOut(BaseModel):
    master_password_set: bool
    veto_threshold: int
    judge_mode: str
    judge_max_retries: int
    judge_http_timeout_sec: int
    docs_enabled: bool


class SettingsIn(BaseModel):
    new_master_password: str | None = Field(default=None, min_length=6)
    veto_threshold: int | None = Field(default=None, ge=50, le=95)
    judge_mode: str | None = Field(default=None, pattern="^(ensemble|single)$")
    judge_max_retries: int | None = Field(default=None, ge=0, le=5)
    judge_http_timeout_sec: int | None = Field(default=None, ge=5, le=300)
    docs_enabled: bool | None = None


@router.get("/settings", response_model=SettingsOut)
async def get_settings_route(
    principal: Principal = Depends(require_admin),
    storage: Storage = Depends(get_storage),
) -> SettingsOut:
    _ = principal
    mp = await storage.get_setting("master_password")
    return SettingsOut(
        master_password_set=bool(mp),
        veto_threshold=await storage.get_veto_threshold(),
        judge_mode=(await storage.get_setting("judge_mode")) or "ensemble",
        judge_max_retries=await storage.get_judge_max_retries(),
        judge_http_timeout_sec=int(
            (await storage.get_setting("judge_http_timeout_sec")) or 60
        ),
        docs_enabled=((await storage.get_setting("docs_enabled")) or "true").lower() == "true",
    )


@router.put("/settings", response_model=SettingsOut)
async def update_settings(
    payload: SettingsIn,
    principal: Principal = Depends(require_admin),
    storage: Storage = Depends(get_storage),
) -> SettingsOut:
    _ = principal
    if payload.new_master_password:
        await storage.set_master_password(payload.new_master_password)
    if payload.veto_threshold is not None:
        await storage.set_veto_threshold(payload.veto_threshold)
    if payload.judge_mode:
        await storage.set_setting("judge_mode", payload.judge_mode)
    if payload.judge_max_retries is not None:
        await storage.set_setting("judge_max_retries", str(payload.judge_max_retries))
    if payload.judge_http_timeout_sec is not None:
        await storage.set_setting(
            "judge_http_timeout_sec", str(payload.judge_http_timeout_sec)
        )
    if payload.docs_enabled is not None:
        await storage.set_setting("docs_enabled", "true" if payload.docs_enabled else "false")
    return await get_settings_route(principal, storage)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
#  LLM-провайдеры                                                               #
# --------------------------------------------------------------------------- #


def _provider_to_out(p) -> LlmProviderOut:
    mask = p.api_key[:4] + "*" * (len(p.api_key) - 8) + p.api_key[-4:]
    return LlmProviderOut(
        id=p.id,
        name=p.name,
        base_url=p.base_url,
        model=p.model,
        api_key_masked=mask,
        role=p.role,
        enabled=p.enabled,
        created_at=p.created_at,
    )


@router.get("/llm-providers", response_model=list[LlmProviderOut])
async def list_llm_providers(
    principal: Principal = Depends(require_admin),
    storage: Storage = Depends(get_storage),
) -> list[LlmProviderOut]:
    _ = principal
    items = await storage.list_llm_providers()
    return [_provider_to_out(p) for p in items]


@router.post("/llm-providers", response_model=LlmProviderOut, status_code=201)
async def add_llm_provider(
    payload: LlmProviderIn,
    principal: Principal = Depends(require_admin),
    storage: Storage = Depends(get_storage),
) -> LlmProviderOut:
    _ = principal
    p = await storage.add_llm_provider(payload)
    return _provider_to_out(p)


@router.put("/llm-providers/{provider_id}", response_model=LlmProviderOut)
async def update_llm_provider(
    provider_id: int,
    payload: LlmProviderIn,
    principal: Principal = Depends(require_admin),
    storage: Storage = Depends(get_storage),
) -> LlmProviderOut:
    _ = principal
    p = await storage.update_llm_provider(provider_id, payload)
    if not p:
        raise HTTPException(404, "LLM-провайдер не найден.")
    return _provider_to_out(p)


@router.delete("/llm-providers/{provider_id}", status_code=204)
async def delete_llm_provider(
    provider_id: int,
    principal: Principal = Depends(require_admin),
    storage: Storage = Depends(get_storage),
) -> None:
    _ = principal
    await storage.delete_llm_provider(provider_id)


@router.post("/llm-providers/{provider_id}/toggle", response_model=LlmProviderOut)
async def toggle_llm_provider(
    provider_id: int,
    enabled: bool,
    principal: Principal = Depends(require_admin),
    storage: Storage = Depends(get_storage),
) -> LlmProviderOut:
    _ = principal
    await storage.set_llm_provider_enabled(provider_id, enabled)
    items = await storage.list_llm_providers()
    for p in items:
        if p.id == provider_id:
            return _provider_to_out(p)
    raise HTTPException(404, "LLM-провайдер не найден.")


# --------------------------------------------------------------------------- #
#  API-ключи                                                                   #
# --------------------------------------------------------------------------- #


class ApiKeyIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)


@router.get("/api-keys", response_model=list[ApiKeyOut])
async def list_api_keys(
    principal: Principal = Depends(require_admin),
    storage: Storage = Depends(get_storage),
) -> list[ApiKeyOut]:
    _ = principal
    items = await storage.list_api_keys()
    return [
        ApiKeyOut(
            id=k.id,
            name=k.name,
            key_prefix=k.key_prefix,
            created_at=k.created_at,
            last_used_at=k.last_used_at,
            revoked=k.revoked,
        )
        for k in items
    ]


@router.post("/api-keys", response_model=ApiKeyCreated, status_code=201)
async def create_api_key(
    payload: ApiKeyIn,
    principal: Principal = Depends(require_admin),
    storage: Storage = Depends(get_storage),
) -> ApiKeyCreated:
    _ = principal
    rec, raw = await storage.create_api_key(payload.name)
    return ApiKeyCreated(id=rec.id, name=rec.name, key=raw, created_at=rec.created_at)


@router.delete("/api-keys/{key_id}", status_code=204)
async def revoke_api_key(
    key_id: int,
    principal: Principal = Depends(require_admin),
    storage: Storage = Depends(get_storage),
) -> None:
    _ = principal
    await storage.revoke_api_key(key_id)


@router.post("/api-keys/{key_id}/rotate", response_model=ApiKeyCreated)
async def rotate_api_key(
    key_id: int,
    principal: Principal = Depends(require_admin),
    storage: Storage = Depends(get_storage),
) -> ApiKeyCreated:
    _ = principal
    result = await storage.rotate_api_key(key_id)
    if not result:
        raise HTTPException(404, "API-ключ не найден.")
    rec, raw = result
    return ApiKeyCreated(id=rec.id, name=rec.name, key=raw, created_at=rec.created_at)


# --------------------------------------------------------------------------- #
#  Статус судьи                                                                 #
# --------------------------------------------------------------------------- #


class JudgeStatusOut(BaseModel):
    status: str  # OFFLINE | ENSEMBLE_ONLY | FULL
    label: str
    ensemble_count: int
    judge_count: int


@router.get("/judge-status", response_model=JudgeStatusOut)
async def judge_status(
    principal: Principal = Depends(current_principal),
    storage: Storage = Depends(get_storage),
) -> JudgeStatusOut:
    _ = principal
    from app.services.signal2_judge import judge_status as _status

    data = await _status(storage)
    return JudgeStatusOut(**data)


# --------------------------------------------------------------------------- #
#  Users (управление преподавателями) — только админ                           #
# --------------------------------------------------------------------------- #


def _user_to_out(u) -> UserOut:
    return UserOut(
        id=u.id,
        login=u.login,
        role=u.role,
        display_name=u.display_name,
        created_at=u.created_at,
        last_login_at=u.last_login_at,
    )


@router.get("/users", response_model=list[UserOut])
async def list_users(
    principal: Principal = Depends(require_admin),
    storage: Storage = Depends(get_storage),
) -> list[UserOut]:
    _ = principal
    return [_user_to_out(u) for u in await storage.list_users()]


@router.post("/users", response_model=UserOut, status_code=201)
async def create_user(
    payload: UserCreateIn,
    principal: Principal = Depends(require_admin),
    storage: Storage = Depends(get_storage),
) -> UserOut:
    _ = principal
    existing = await storage.get_user_by_login(payload.login)
    if existing:
        raise HTTPException(409, f"Пользователь с логином '{payload.login}' уже существует.")
    user = await storage.create_user(
        login=payload.login,
        password=payload.password,
        role=payload.role,
        display_name=payload.display_name,
    )
    return _user_to_out(user)


@router.post("/users/{user_id}/reset-password", status_code=204)
async def reset_user_password(
    user_id: int,
    payload: PasswordResetIn,
    principal: Principal = Depends(require_admin),
    storage: Storage = Depends(get_storage),
) -> None:
    _ = principal
    user = await storage.get_user(user_id)
    if not user:
        raise HTTPException(404, "Пользователь не найден.")
    await storage.set_user_password(user_id, payload.new_password)


@router.delete("/users/{user_id}", status_code=204)
async def delete_user(
    user_id: int,
    principal: Principal = Depends(require_admin),
    storage: Storage = Depends(get_storage),
) -> None:
    _ = principal
    user = await storage.get_user(user_id)
    if not user:
        raise HTTPException(404, "Пользователь не найден.")
    if user.login == "admin":
        raise HTTPException(400, "Нельзя удалить встроенного администратора.")
    await storage.delete_user(user_id)


@router.put("/users/me/password", status_code=204)
async def change_own_password(
    payload: PasswordResetIn,
    principal: Principal = Depends(current_principal),
    storage: Storage = Depends(get_storage),
) -> None:
    """Смена собственного пароля (доступно и teacher, и admin)."""
    if not principal.user_id:
        raise HTTPException(400, "Смена пароля доступна только для пользователей с сессией.")
    await storage.set_user_password(principal.user_id, payload.new_password)
