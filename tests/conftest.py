"""Pytest configuration и общие фикстуры."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

# Гарантировать, что используется временная БД, а не production .env.
# Делаем до импорта app.*
_TMPDIR = Path(tempfile.mkdtemp(prefix="airadar-test-"))
os.environ["DATABASE_PATH"] = str(_TMPDIR / "test.db")
os.environ["JWT_SECRET"] = "test-secret-only-for-pytest"
os.environ["MASTER_PASSWORD"] = "airadar"
os.environ.setdefault("DOCS_ENABLED", "false")
# Тесты НЕ должны подтягивать LLM-пресет из .env — иначе появляется
# провайдер по умолчанию и degraded-mode проверить нельзя.
for _k in ("DEFAULT_LLM_BASE_URL", "DEFAULT_LLM_MODEL", "DEFAULT_LLM_API_KEY"):
    os.environ.pop(_k, None)
    os.environ[_k] = ""

import asyncio  # noqa: E402

from app.config import reset_settings_cache  # noqa: E402
from app.services.storage import close_storage, reset_storage_for_tests  # noqa: E402


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session", autouse=True)
def _cleanup(request):
    yield
    try:
        asyncio.get_event_loop().run_until_complete(close_storage())
    except Exception:
        pass
    reset_settings_cache()
    reset_storage_for_tests()
    shutil.rmtree(_TMPDIR, ignore_errors=True)


@pytest.fixture
def fresh_db():
    """Каждый тест получает чистую БД."""
    import asyncio

    async def _reset():
        await close_storage()
        reset_storage_for_tests()
        # Удалить файл БД
        db_path = Path(os.environ["DATABASE_PATH"])
        if db_path.exists():
            db_path.unlink()
        from app.services.storage import get_storage

        storage = await get_storage()
        return storage

    return asyncio.get_event_loop().run_until_complete(_reset())
