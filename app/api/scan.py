"""Маршруты сканирования AI Radar.

Поддерживает:
  * Синхронные quick/json, quick (multipart), deep — для программного API.
  * Асинхронные задачи (start/status/list) — переживают перезагрузку страницы.
  * SSE-стриминг прогресса по задаче с детальными статусами по каждой модели.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.services.auth_service import Principal, current_principal
from app.services.file_parser import FileParseError, parse_file
from app.services.pipeline import run_full_scan
from app.services.signal1_stat import analyze as analyze_stat
from app.services.signal2_judge import (
    DetectorVerdict,
    JudgeResult,
    _call_detector,
    _call_judge,
    _aggregate_ensemble,
    _offline,
)
from app.services.storage import ScanReportOut, Storage, get_storage, LlmProvider


log = logging.getLogger("ai-radar.scan")
router = APIRouter(prefix="/api/scan", tags=["scan"])


class QuickTextIn(BaseModel):
    text: str = Field(..., min_length=10)
    title: str | None = None


# --------------------------------------------------------------------------- #
#  Синхронные эндпоинты (для API/скриптов)                                     #
# --------------------------------------------------------------------------- #


@router.post("/quick", response_model=ScanReportOut)
async def quick_scan(
    principal: Principal = Depends(current_principal),
    text: str | None = Form(default=None),
    title: str | None = Form(default=None),
    file: UploadFile | None = File(default=None),
    storage: Storage = Depends(get_storage),
) -> ScanReportOut:
    if file is not None:
        content = await file.read()
        try:
            parsed_text = parse_file(filename=file.filename or "upload.txt", content=content)
        except FileParseError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if not parsed_text.strip():
            raise HTTPException(status_code=400, detail="Файл пуст или не содержит текста.")
        return await run_full_scan(
            text=parsed_text, title=title or file.filename or "scan.txt",
            source="quick", storage=storage,
        )
    if not text:
        raise HTTPException(400, "Нужно передать file или text.")
    return await run_full_scan(
        text=text, title=title or "scan.txt", source="quick", storage=storage,
    )


@router.post("/quick/json", response_model=ScanReportOut)
async def quick_scan_json(
    payload: QuickTextIn,
    principal: Principal = Depends(current_principal),
    storage: Storage = Depends(get_storage),
) -> ScanReportOut:
    _ = principal
    return await run_full_scan(
        text=payload.text, title=payload.title or "scan.txt",
        source="quick", storage=storage,
    )


@router.post("/deep/{file_id}", response_model=ScanReportOut)
async def deep_scan(
    file_id: int,
    principal: Principal = Depends(current_principal),
    storage: Storage = Depends(get_storage),
) -> ScanReportOut:
    _ = principal
    file = await storage.get_file(file_id)
    if not file:
        raise HTTPException(status_code=404, detail="Файл не найден.")
    text = file.get("text") or ""
    if not text.strip():
        raise HTTPException(status_code=400, detail="В файле нет распознанного текста.")
    return await run_full_scan(
        text=text, title=file.get("name") or f"file-{file_id}",
        source="deep", storage=storage,
    )


# --------------------------------------------------------------------------- #
#  Асинхронные задачи (переживают перезагрузку страницы)                       #
# --------------------------------------------------------------------------- #


class StartScanIn(BaseModel):
    text: str | None = None
    title: str | None = None
    file_id: int | None = None


@router.post("/tasks", status_code=201)
async def start_scan_task(
    payload: StartScanIn,
    principal: Principal = Depends(current_principal),
    storage: Storage = Depends(get_storage),
) -> dict:
    """Запустить асинхронную задачу сканирования. Возвращает task_id."""
    _ = principal
    text = payload.text
    file_id = payload.file_id
    if file_id:
        f = await storage.get_file(file_id)
        if not f:
            raise HTTPException(404, "Файл не найден.")
        text = f.get("text") or ""
        if not text.strip():
            raise HTTPException(400, "В файле нет текста.")
    elif not text:
        raise HTTPException(400, "Нужно передать text или file_id.")
    task = await storage.create_scan_task(
        title=payload.title or "Проверка",
        source="deep" if file_id else "quick",
        text=text,
        file_id=file_id,
    )
    # Запустить фоновую обработку
    asyncio.create_task(_run_scan_background(task["id"], text, payload.title or "Проверка",
                                              "deep" if file_id else "quick", storage))
    return task


@router.post("/tasks/upload", status_code=201)
async def start_scan_task_upload(
    principal: Principal = Depends(current_principal),
    file: UploadFile = File(...),
    storage: Storage = Depends(get_storage),
) -> dict:
    """Запустить асинхронную задачу сканирования загруженного файла."""
    _ = principal
    content = await file.read()
    try:
        text = parse_file(filename=file.filename or "upload.txt", content=content)
    except FileParseError as exc:
        raise HTTPException(400, detail=str(exc))
    if not text.strip():
        raise HTTPException(400, "Файл пуст или не содержит текста.")
    task = await storage.create_scan_task(
        title=file.filename or "Проверка",
        source="quick",
        text=text,
    )
    asyncio.create_task(_run_scan_background(task["id"], text, file.filename or "Проверка",
                                              "quick", storage))
    return task


@router.get("/tasks")
async def list_scan_tasks(
    limit: int = 20,
    principal: Principal = Depends(current_principal),
    storage: Storage = Depends(get_storage),
) -> list[dict]:
    _ = principal
    return await storage.list_scan_tasks(limit=limit)


@router.get("/tasks/{task_id}")
async def get_scan_task(
    task_id: int,
    principal: Principal = Depends(current_principal),
    storage: Storage = Depends(get_storage),
) -> dict:
    _ = principal
    task = await storage.get_scan_task(task_id)
    if not task:
        raise HTTPException(404, "Задача не найдена.")
    return task


@router.get("/tasks/{task_id}/stream")
async def stream_scan_task(
    task_id: int,
    principal: Principal = Depends(current_principal),
    storage: Storage = Depends(get_storage),
):
    """SSE-стриминг прогресса задачи с детальными статусами."""
    _ = principal
    task = await storage.get_scan_task(task_id)
    if not task:
        raise HTTPException(404, "Задача не найдена.")

    async def event_gen():
        last_progress = -1
        last_msg = ""
        idle_count = 0
        while True:
            t = await storage.get_scan_task(task_id)
            if not t:
                yield f"data: {json.dumps({'event': 'error', 'message': 'Задача исчезла'})}\n\n"
                break
            changed = (t["progress"] != last_progress or t["progress_msg"] != last_msg)
            if changed or idle_count == 0:
                payload = {
                    "event": "progress",
                    "task_id": task_id,
                    "status": t["status"],
                    "progress": t["progress"],
                    "message": t["progress_msg"],
                    "steps": t.get("steps", []),
                }
                if t["status"] == "done":
                    payload["event"] = "final"
                    payload["report_id"] = t.get("report_id")
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    break
                elif t["status"] == "error":
                    payload["event"] = "error"
                    payload["message"] = t.get("error") or "Ошибка"
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    break
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                last_progress = t["progress"]
                last_msg = t["progress_msg"]
            idle_count += 1
            await asyncio.sleep(0.5)

    return StreamingResponse(event_gen(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache",
                                       "Connection": "keep-alive",
                                       "X-Accel-Buffering": "no"})


# --------------------------------------------------------------------------- #
#  Фоновая обработка задачи с детальным прогрессом                              #
# --------------------------------------------------------------------------- #


async def _run_scan_background(task_id: int, text: str, title: str, source: str,
                                 storage: Storage) -> None:
    """Запустить конвейер сканирования в фоне, обновляя прогресс в БД."""
    import json as _json

    steps: list[dict[str, Any]] = []

    async def progress_cb(event_name: str, data: dict[str, Any]):
        msg = data.get("message", event_name)
        pct = data.get("progress", 0)
        steps.append({"event": event_name, "message": msg, "data": data})
        await storage.update_scan_task(
            task_id, status="running", progress=pct, progress_msg=msg,
            steps_json=_json.dumps(steps, ensure_ascii=False),
        )

    try:
        await storage.update_scan_task(task_id, status="running", progress=2,
                                         progress_msg="Подготовка текста...")

        # --- Сигнал 1: считаем метрики по одной для прогресса ---
        await progress_cb("signal1_start", {"message": "Анализирую текст: 20 статистических метрик...", "progress": 5})

        # Считаем метрики (единым вызовом, но с прогрессом)
        metric_names = [
            ("Burstiness (Всплесковость)", 10),
            ("Lexical Diversity (TTR)", 12),
            ("Sentence Length Variance", 14),
            ("Perplexity Proxy", 16),
            ("Cliche Density", 18),
            ("Punctuation Entropy", 20),
            ("Starter Entropy", 22),
            ("Yule's K / Honore's R / Sichel's S", 24),
            ("N-gram Repetition / Rhythm Entropy", 26),
            ("Остальные метрики", 28),
        ]
        for name, pct in metric_names:
            await storage.update_scan_task(task_id, progress=pct,
                                             progress_msg=f"Сигнал 1: считаю {name}...")
            await asyncio.sleep(0.05)  # небольшая задержка для UI

        signal1 = analyze_stat(text)
        await progress_cb("signal1_done", {
            "message": f"Сигнал 1 готов: stat_score={signal1.stat_score}/100, burstiness={signal1.metrics.burstiness:.2f}, TTR={signal1.metrics.lexical_diversity_ttr:.2f}",
            "progress": 30,
            "stat_score": signal1.stat_score,
        })

        # --- Сигнал 2: ансамбль ---
        ensemble_providers = await storage.list_enabled_llm_providers(role="ensemble")
        judge_providers = await storage.list_enabled_llm_providers(role="judge")
        veto_threshold = await storage.get_veto_threshold()

        if not ensemble_providers and not judge_providers:
            await progress_cb("skipped", {
                "message": "LLM-провайдеры не настроены — работаю в режиме статистики",
                "progress": 50,
            })
            judge = _offline(signal1)
        else:
            if not ensemble_providers:
                await progress_cb("skipped", {
                    "message": "Ансамбль пуст — работаю в режиме статистики",
                    "progress": 50,
                })
                judge = _offline(signal1)
            else:
                # Опрос ансамбля с детальным прогрессом
                await progress_cb("ensemble_start", {
                    "message": f"Опрашиваю ансамбль из {len(ensemble_providers)} детекторов...",
                    "progress": 35,
                    "ensemble_count": len(ensemble_providers),
                    "judge_count": len(judge_providers),
                })

                from app.config import get_settings
                s = get_settings()
                timeout = s.judge_http_timeout_sec

                # Параллельный опрос с обновлением прогресса по каждой модели
                # Используем простой подход: запускаем все, обновляем прогресс при завершении каждой
                pending: list[tuple[int, LlmProvider, asyncio.Task]] = []
                for idx, p in enumerate(ensemble_providers):
                    task = asyncio.create_task(_call_detector(p, text, signal1, timeout))
                    pending.append((idx, p, task))

                verdicts: list[DetectorVerdict] = [None] * len(ensemble_providers)  # type: ignore
                total = len(pending)
                done_count = 0
                # Ждём завершения каждой, обновляя прогресс
                for idx, p, task in pending:
                    try:
                        v = await task
                    except Exception as exc:
                        v = DetectorVerdict(
                            provider_name=p.name, model=p.model,
                            ai_score=0, veto_triggered=False, veto_reason=None,
                            academic_integrity="ПОДТВЕРЖДЕНА",
                            error=f"exception: {exc!s}",
                        )
                    verdicts[idx] = v
                    done_count += 1
                    pct = 35 + int(30 * done_count / total)
                    score_str = f"вердикт={v.ai_score}" if v.error is None else f"ошибка: {v.error[:50]}"
                    await progress_cb("ensemble_progress", {
                        "message": f"{p.name} ({p.model}) ответил: {score_str}",
                        "progress": pct,
                        "model": p.model,
                        "ai_score": v.ai_score,
                        "error": v.error,
                        "done": done_count,
                        "total": total,
                    })

                ensemble = _aggregate_ensemble(verdicts, veto_threshold)
                await progress_cb("ensemble_done", {
                    "message": f"Ансамбль завершён: median ai_score={ensemble.ai_score}, veto-голосов={sum(1 for v in verdicts if v.veto_triggered and v.error is None)}/{len([v for v in verdicts if v.error is None])}",
                    "progress": 70,
                    "ai_score": ensemble.ai_score,
                })

                # --- Метасудья ---
                if judge_providers and ensemble.available:
                    await progress_cb("judge_start", {
                        "message": f"Метасудья ({judge_providers[0].model}) оценивает вердикты ансамбля...",
                        "progress": 75,
                    })
                    try:
                        judge_v = await _call_judge(judge_providers[0], signal1, ensemble, timeout)
                    except Exception as exc:
                        judge_v = DetectorVerdict(
                            provider_name=judge_providers[0].name,
                            model=judge_providers[0].model,
                            ai_score=0, veto_triggered=False, veto_reason=None,
                            academic_integrity="ПОДТВЕРЖДЕНА",
                            error=f"exception: {exc!s}",
                        )

                    if judge_v.error is not None and s.judge_max_retries > 0:
                        for _ in range(await storage.get_judge_max_retries()):
                            try:
                                rv = await _call_judge(judge_providers[0], signal1, ensemble, timeout)
                                if rv.error is None:
                                    judge_v = rv
                                    break
                            except Exception:
                                continue

                    if judge_v.error is not None:
                        # Судья упал — используем ансамбль
                        judge = JudgeResult(
                            ai_score=ensemble.ai_score,
                            veto_triggered=ensemble.veto_triggered,
                            veto_reason=ensemble.veto_reason,
                            academic_integrity=ensemble.academic_integrity,
                            suspicious_fragments=ensemble.suspicious_fragments,
                            judge_status="ENSEMBLE_ONLY",
                            ensemble_panel=ensemble.panel,
                            judge_verdict=judge_v,
                            raw_json={"ensemble": ensemble.to_summary(),
                                       "judge": {"error": judge_v.error}},
                        )
                    else:
                        # Судья отработал
                        fragments = []
                        seen = set()
                        for f in judge_v.suspicious_fragments + ensemble.suspicious_fragments:
                            k = f.strip().lower()
                            if k and k not in seen:
                                fragments.append(f)
                                seen.add(k)
                        veto_triggered = judge_v.veto_triggered or ensemble.veto_triggered
                        veto_reason = judge_v.veto_reason or ensemble.veto_reason
                        if veto_triggered and not veto_reason:
                            veto_reason = f"ВЕТО метасудьи (score={judge_v.ai_score})."
                        integrity = "НАРУШЕНА" if (veto_triggered or judge_v.ai_score >= veto_threshold) else "ПОДТВЕРЖДЕНА"
                        judge = JudgeResult(
                            ai_score=judge_v.ai_score,
                            veto_triggered=veto_triggered,
                            veto_reason=veto_reason,
                            academic_integrity=integrity,
                            suspicious_fragments=fragments[:20],
                            judge_status="FULL",
                            ensemble_panel=ensemble.panel,
                            judge_verdict=judge_v,
                            raw_json={"ensemble": ensemble.to_summary(),
                                       "judge": {"model": judge_v.model, "ai_score": judge_v.ai_score,
                                                  "veto": judge_v.veto_triggered,
                                                  "veto_reason": judge_v.veto_reason,
                                                  "integrity": judge_v.academic_integrity,
                                                  "fragments": judge_v.suspicious_fragments}},
                        )
                        await progress_cb("judge_done", {
                            "message": f"Метасудья вынес вердикт: ai_score={judge_v.ai_score}, veto={judge_v.veto_triggered}",
                            "progress": 90,
                            "ai_score": judge_v.ai_score,
                        })
                else:
                    # Судьи нет — финал = ансамбль
                    judge = JudgeResult(
                        ai_score=ensemble.ai_score,
                        veto_triggered=ensemble.veto_triggered,
                        veto_reason=ensemble.veto_reason,
                        academic_integrity=ensemble.academic_integrity,
                        suspicious_fragments=ensemble.suspicious_fragments,
                        judge_status="ENSEMBLE_ONLY",
                        ensemble_panel=ensemble.panel,
                        judge_verdict=None,
                        raw_json={"ensemble": ensemble.to_summary(), "judge": None},
                    )

        # --- Финальный отчёт ---
        await progress_cb("finalizing", {
            "message": "Формирую финальный отчёт...",
            "progress": 95,
        })

        from app.services.pipeline import _now, _preview
        # Вычисляем финальные значения (как в pipeline, но уже с готовым judge)
        if judge.judge_status == "OFFLINE":
            final_ai_score = judge.ai_score
            final_veto = False
            final_veto_reason = None
            integrity = "ПОДТВЕРЖДЕНА"
        else:
            final_ai_score = max(0, min(100, judge.ai_score))
            final_veto = judge.veto_triggered or final_ai_score >= veto_threshold
            final_veto_reason = judge.veto_reason
            if final_veto and not final_veto_reason:
                final_veto_reason = f"Итоговый ai_score ({final_ai_score}%) ≥ порогу VETO ({veto_threshold}%)."
            integrity = "НАРУШЕНА" if (final_veto or final_ai_score >= veto_threshold) else "ПОДТВЕРЖДЕНА"

        fragments = list(judge.suspicious_fragments)
        seen_f = {f.strip().lower() for f in fragments}
        for f in signal1.suspicious_stat_fragments:
            k = f.strip().lower()
            if k and k not in seen_f:
                fragments.append(f)
                seen_f.add(k)
            if len(fragments) >= 20:
                break

        report = ScanReportOut(
            id=None, source=source, title=title,
            ai_score=final_ai_score, stat_score=signal1.stat_score,
            veto=final_veto, veto_reason=final_veto_reason,
            academic_integrity=integrity, judge_status=judge.judge_status,
            metrics={
                "signal1": signal1.metrics.model_dump(),
                "signal1_interpretation": signal1.interpretation,
                "signal2_ensemble_panel": judge.raw_json or {},
                "judge_status": judge.judge_status,
            },
            suspicious_fragments=fragments,
            text_preview=_preview(text),
            created_at=_now(),
        )
        report_id = await storage.add_scan_report(report)
        report.id = report_id

        await storage.update_scan_task(
            task_id, status="done", progress=100,
            progress_msg=f"Готово! ai_score={final_ai_score}, честность={integrity}",
            report_id=report_id,
            steps_json=_json.dumps(steps, ensure_ascii=False),
        )
    except Exception as exc:
        log.exception("Фоновая задача %s упала", task_id)
        await storage.update_scan_task(
            task_id, status="error", error=str(exc),
            progress_msg=f"Ошибка: {exc!s}",
        )


# --------------------------------------------------------------------------- #
#  Отчёты                                                                      #
# --------------------------------------------------------------------------- #


@router.get("/reports/recent", response_model=list[dict])
async def recent_reports(
    limit: int = 10,
    principal: Principal = Depends(current_principal),
    storage: Storage = Depends(get_storage),
) -> list[dict]:
    _ = principal
    return await storage.list_recent_reports(limit=limit)


@router.get("/reports/{report_id}")
async def get_report(
    report_id: int,
    principal: Principal = Depends(current_principal),
    storage: Storage = Depends(get_storage),
) -> dict:
    _ = principal
    r = await storage.get_scan_report(report_id)
    if not r:
        raise HTTPException(status_code=404, detail="Отчёт не найден.")
    return r
