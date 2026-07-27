"""Оркестратор полного конвейера сканирования AI Radar.

Конвейер:
  1. Сигнал 1 — 20 статистических метрик + stat_score.
  2. Сигнал 2:
     a. Ансамбль детекторов анализирует текст.
     b. Метасудья оценивает вердикты ансамбля (без исходного текста).
  3. Финальный ai_score:
     - FULL: вердикт метасудьи.
     - ENSEMBLE_ONLY: медиана ансамбля.
     - OFFLINE: stat_score (degraded).
  4. VETO активируется только при judge_status in {FULL, ENSEMBLE_ONLY}.
  5. Финальная academic_integrity учитывает VETO и порог.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from app.services.signal1_stat import Signal1Result, analyze as analyze_stat
from app.services.signal2_judge import JudgeResult, run_judge
from app.services.storage import ScanReportOut, Storage


log = logging.getLogger("ai-radar.pipeline")


# Тип callback прогресса: (event_name, data_dict) -> None
ProgressCallback = Callable[[str, dict[str, Any]], Awaitable[None]]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _preview(text: str, max_len: int = 100000) -> str:
    """Текст для отчёта. Большой лимит, чтобы почти весь текст был виден."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "\n\n[...текст обрезан для отображения...]"


async def run_full_scan(
    *,
    text: str,
    title: str,
    source: str,
    storage: Storage,
    progress: ProgressCallback | None = None,
) -> ScanReportOut:
    """Полный двухсигнальный конвейер.

    `progress` — опциональный async-callback для стриминга прогресса в UI.
    """
    text_clean = (text or "").strip()
    if not text_clean:
        raise ValueError("Пустой текст работы — нечего анализировать.")

    if progress:
        await progress("signal1_start", {"message": "Считаю 20 статистических метрик..."})

    # --- Сигнал 1 ---
    signal1: Signal1Result = analyze_stat(text_clean)

    if progress:
        await progress("signal1_done", {
            "message": f"Сигнал 1 завершён: stat_score={signal1.stat_score}/100",
            "stat_score": signal1.stat_score,
            "burstiness": signal1.metrics.burstiness,
            "ttr": signal1.metrics.lexical_diversity_ttr,
            "cliche_density": signal1.metrics.synthetic_cliche_density,
            "interpretation": signal1.interpretation,
        })

    # --- Сигнал 2 ---
    if progress:
        ens_count = len(await storage.list_enabled_llm_providers(role="ensemble"))
        judge_count = len(await storage.list_enabled_llm_providers(role="judge"))
        if ens_count == 0 and judge_count == 0:
            await progress("ensemble_skipped", {
                "message": "LLM-провайдеров нет — работаю в degraded-режиме (только Сигнал 1)"
            })
        else:
            await progress("ensemble_start", {
                "message": f"Опрашиваю ансамбль из {ens_count} детекторов параллельно...",
                "ensemble_count": ens_count,
                "judge_count": judge_count,
            })

    try:
        judge: JudgeResult = await run_judge(text_clean, signal1, storage)
    except Exception as exc:
        log.exception("Сигнал 2 упал: %s", exc)
        from app.services.signal2_judge import _offline

        judge = _offline(signal1, note=f"внутренняя ошибка: {exc!s}")

    if progress:
        if judge.judge_status == "OFFLINE":
            await progress("ensemble_done", {
                "message": "Судья OFFLINE — вердикт на базе Сигнала 1",
                "judge_status": "OFFLINE",
            })
        elif judge.judge_status == "ENSEMBLE_ONLY":
            await progress("ensemble_done", {
                "message": f"Ансамбль завершён: median ai_score={judge.ai_score}/100 (метасудья не подключён)",
                "judge_status": "ENSEMBLE_ONLY",
                "ai_score": judge.ai_score,
            })
        else:
            await progress("ensemble_done", {
                "message": f"Ансамбль завершён, передаю вердикты метасудье...",
                "judge_status": "FULL",
                "ai_score": judge.ai_score,
            })

    if progress and judge.judge_status == "FULL":
        await progress("judge_start", {"message": "Метасудья оценивает вердикты ансамбля..."})

    veto_threshold = await storage.get_veto_threshold()

    # --- Финальный ai_score ---
    if judge.judge_status == "OFFLINE":
        final_ai_score = judge.ai_score  # = stat_score
    else:
        final_ai_score = judge.ai_score
    final_ai_score = max(0, min(100, final_ai_score))

    # --- Финальное ВЕТО ---
    if judge.judge_status == "OFFLINE":
        final_veto = False
        final_veto_reason = None
    else:
        final_veto = judge.veto_triggered or final_ai_score >= veto_threshold
        final_veto_reason = judge.veto_reason
        if final_veto and not final_veto_reason:
            final_veto_reason = (
                f"Итоговый ai_score ({final_ai_score}%) ≥ порогу VETO ({veto_threshold}%)."
            )

    # --- Финальная честность ---
    if judge.judge_status == "OFFLINE":
        integrity = "ПОДТВЕРЖДЕНА"
    elif final_veto or final_ai_score >= veto_threshold:
        integrity = "НАРУШЕНА"
    else:
        integrity = "ПОДТВЕРЖДЕНА"

    # --- Фрагменты ---
    fragments: list[str] = list(judge.suspicious_fragments)
    seen = {f.strip().lower() for f in fragments}
    for f in signal1.suspicious_stat_fragments:
        k = f.strip().lower()
        if k and k not in seen:
            fragments.append(f)
            seen.add(k)
        if len(fragments) >= 20:
            break

    report = ScanReportOut(
        id=None,
        source=source,
        title=title,
        ai_score=final_ai_score,
        stat_score=signal1.stat_score,
        veto=final_veto,
        veto_reason=final_veto_reason,
        academic_integrity=integrity,
        judge_status=judge.judge_status,
        metrics={
            "signal1": signal1.metrics.model_dump(),
            "signal1_interpretation": signal1.interpretation,
            "signal2_ensemble_panel": judge.raw_json or {},
            "judge_status": judge.judge_status,
        },
        suspicious_fragments=fragments,
        text_preview=_preview(text_clean),
        created_at=_now(),
    )

    report_id = await storage.add_scan_report(report)
    report.id = report_id  # type: ignore[misc]

    if progress:
        await progress("done", {
            "message": f"Готово! ai_score={final_ai_score}, честность={integrity}",
            "report_id": report_id,
        })

    return report
