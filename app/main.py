"""FastAPI приложение AI Radar."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api import admin, auth, folders, scan
from app.config import PROJECT_ROOT, get_settings
from app.services.storage import close_storage, get_storage


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("ai-radar")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Инициализация и graceful-shutdown хранилища."""
    log.info("AI Radar v%s — запуск", __version__)
    storage = await get_storage()
    log.info("SQLite инициализирован: %s", storage.db_uri)
    providers = await storage.list_enabled_llm_providers()
    log.info("Активных LLM-провайдеров: %d", len(providers))
    for p in providers:
        log.info("  • %s (%s)", p.name, p.model)
    yield
    log.info("AI Radar — остановка, закрываем storage...")
    await close_storage()


def create_app() -> FastAPI:
    s = get_settings()
    app = FastAPI(
        title="AI Radar",
        description=(
            "Двухсигнальная автономная система детекции ИИ-контента и "
            "академической честности с AI Semantic Judge. "
            "Авторизация: Bearer &lt;jwt&gt; (UI-сессия) или Bearer &lt;api-key&gt; / X-API-Key (программный доступ)."
        ),
        version=__version__,
        docs_url="/docs" if s.docs_enabled else None,
        redoc_url="/redoc" if s.docs_enabled else None,
        lifespan=lifespan,
    )

    # --- Security-схемы для Swagger ---
    from fastapi.security import HTTPBearer, APIKeyHeader

    bearer_scheme = HTTPBearer(bearerFormat="JWT", auto_error=False)
    apikey_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)

    app.dependency_overrides[None] = None  # noop placeholder

    # --- Роутеры ---
    app.include_router(auth.router)
    app.include_router(scan.router)
    app.include_router(folders.router)
    app.include_router(admin.router)

    # --- Static + index.html ---
    static_dir = PROJECT_ROOT / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    templates_dir = PROJECT_ROOT / "app" / "templates"

    @app.get("/", include_in_schema=False)
    async def index():
        idx = templates_dir / "index.html"
        if idx.exists():
            return FileResponse(str(idx))
        return JSONResponse(
            status_code=404,
            content={"detail": "index.html не найден. Соберите фронтенд."},
        )

    @app.get("/report/{report_id}", include_in_schema=False)
    async def report_page(report_id: int):
        """Страница детализированного отчёта (HTML)."""
        tpl = templates_dir / "report.html"
        if not tpl.exists():
            return JSONResponse(status_code=404, content={"detail": "report.html не найден"})
        html = tpl.read_text(encoding="utf-8")
        # Подставить report_id (значение уже int, безопасно)
        html = html.replace("{REPORT_ID}", str(report_id))
        return HTMLResponse(html)

    @app.get("/docs-page", include_in_schema=False)
    async def docs_page():
        """Страница документации API в стиле системы (HTML)."""
        tpl = templates_dir / "docs.html"
        if not tpl.exists():
            return JSONResponse(status_code=404, content={"detail": "docs.html не найден"})
        return FileResponse(str(tpl))

    @app.get("/health", tags=["meta"])
    async def health() -> dict:
        return {"status": "ok", "version": __version__}

    return app


app = create_app()
