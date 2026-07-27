"""CRUD-маршруты для папок и файлов."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.services.auth_service import Principal, current_principal
from app.services.file_parser import FileParseError, parse_file
from app.services.storage import Storage, get_storage


router = APIRouter(prefix="/api/folders", tags=["folders"])


class FolderIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    parent_id: int | None = None


@router.get("")
@router.get("/", response_model=list[dict], include_in_schema=False)
async def list_folders(
    parent_id: int | None = None,
    principal: Principal = Depends(current_principal),
    storage: Storage = Depends(get_storage),
) -> list[dict]:
    """Получить список папок. Teacher видит только свои + shared, admin — все."""
    owner = None if principal.is_admin else principal.user_id
    return await storage.list_folders(parent_id=parent_id, owner_id=owner)


@router.post("", response_model=dict, status_code=201)
async def create_folder(
    payload: FolderIn,
    principal: Principal = Depends(current_principal),
    storage: Storage = Depends(get_storage),
) -> dict:
    # Teacher создаёт папку в свой owner_id; admin — без owner (shared)
    owner = None if principal.is_admin else principal.user_id
    return await storage.create_folder(payload.name, payload.parent_id, owner_id=owner)


@router.delete("/{folder_id}", status_code=204)
async def delete_folder(
    folder_id: int,
    principal: Principal = Depends(current_principal),
    storage: Storage = Depends(get_storage),
) -> None:
    folder = await storage.get_folder(folder_id)
    if not folder:
        raise HTTPException(404, "Папка не найдена.")
    if not principal.is_admin and folder.get("owner_id") != principal.user_id:
        raise HTTPException(403, "Нет прав на удаление этой папки.")
    await storage.delete_folder(folder_id)


@router.get("/{folder_id}/files")
async def list_files(
    folder_id: int,
    principal: Principal = Depends(current_principal),
    storage: Storage = Depends(get_storage),
) -> list[dict]:
    _ = principal
    return await storage.list_files(folder_id)


@router.post("/{folder_id}/files", status_code=201)
async def upload_file(
    folder_id: int,
    file: UploadFile = File(...),
    principal: Principal = Depends(current_principal),
    storage: Storage = Depends(get_storage),
) -> dict:
    _ = principal
    content = await file.read()
    try:
        text = parse_file(filename=file.filename or "upload.txt", content=content)
    except FileParseError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return await storage.add_file(
        folder_id=folder_id,
        name=file.filename or "upload.txt",
        text=text,
        size_bytes=len(content),
    )


@router.get("/files/recent")
async def recent_files(
    limit: int = 10,
    principal: Principal = Depends(current_principal),
    storage: Storage = Depends(get_storage),
) -> list[dict]:
    _ = principal
    return await storage.list_recent_files(limit)


@router.delete("/files/{file_id}", status_code=204)
async def delete_file(
    file_id: int,
    principal: Principal = Depends(current_principal),
    storage: Storage = Depends(get_storage),
) -> None:
    _ = principal
    await storage.delete_file(file_id)
