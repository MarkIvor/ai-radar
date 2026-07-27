"""Сигнал 2: AI Semantic Judge — двухуровневая архитектура AI Radar.

Архитектура:
  Уровень 1 — **Ансамбль детекторов** (`role='ensemble'`).
    Любое число OpenAI-compatible LLM анализируют сам текст работы +
    матрицу метрик Сигнала 1. Каждая модель возвращает свой вердикт.
    Семейства моделей, которые мы ожидаем в ансамбле:
      - ChatGPT    (openai/gpt-*)
      - Claude     (anthropic/claude-*)
      - Gemini     (google/gemini-*)
      - DeepSeek   (deepseek/*)
      - Grok       (x-ai/grok-*)
      - Kimi       (moonshotai/kimi-*)
    Чем больше разных семейств — тем устойчивее вердикт.

    Уникальность AI Radar: каждая модель-детектор анализирует не только
    "синтетический синтаксис", но и **проверяет, не она ли сама (или её
    "семья" LLM) могла написать этот текст**. Модель знает свой
    собственный стиль аргументации, типичные паттерны своей генерации
    и может опознать "собственный почерк". Также оценивается наличие
    человеческих особенностей: личного опыта, креативности, эмоций,
    нестандартных мнений, живой аргументации.

  Уровень 2 — **Метасудья** (`role='judge'`).
    ОДНА сильная модель (рекомендация: Claude Sonnet/Opus, Gemini Pro),
    которая НЕ получает сам текст работы. Вместо этого ей передаётся:
      - сводка Сигнала 1 (метрики + интерпретация);
      - вердикты всех моделей ансамбля (ai_score, veto, fragments);
      - промпт для оценки.
    Метасудья выносит финальный вердикт и имеет право БЛОКИРУЮЩЕГО ВЕТО.

Статусы (`judge_status`):
  - "OFFLINE"        — нет ни ансамбля, ни судьи (degraded на Сигнале 1).
  - "ENSEMBLE_ONLY"  — есть ансамбль, нет судьи (финал = агрегат ансамбля).
  - "FULL"           — есть и ансамбль, и судья (финал = вердикт судьи).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

from app.config import get_settings
from app.services.signal1_stat import Signal1Result
from app.services.storage import LlmProvider, Storage


log = logging.getLogger("ai-radar.judge")


# --------------------------------------------------------------------------- #
#  Модели                                                                       #
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class DetectorVerdict:
    """Вердикт одной модели-детектора ансамбля."""

    provider_name: str
    model: str
    ai_score: int
    veto_triggered: bool
    veto_reason: str | None
    academic_integrity: str
    suspicious_fragments: list[str] = field(default_factory=list)
    human_signals: list[str] = field(default_factory=list)
    llm_signals: list[str] = field(default_factory=list)
    raw: str = ""
    error: str | None = None


@dataclass(slots=True)
class EnsembleResult:
    """Агрегированный вердикт ансамбля детекторов."""

    ai_score: int
    veto_triggered: bool
    veto_reason: str | None
    academic_integrity: str
    suspicious_fragments: list[str]
    panel: list[DetectorVerdict]
    available: bool  # есть ли хотя бы один успешный вердикт

    def to_summary(self) -> dict[str, Any]:
        """Краткая сводка для передачи метасудье."""
        # Объединить human/llm signals со всех детекторов
        all_human_signals: list[str] = []
        all_llm_signals: list[str] = []
        for v in self.panel:
            if v.error is None:
                all_human_signals.extend(v.human_signals[:2])
                all_llm_signals.extend(v.llm_signals[:2])
        return {
            "ai_score_median": self.ai_score,
            "veto_votes": sum(1 for v in self.panel if v.veto_triggered and v.error is None),
            "veto_triggered": self.veto_triggered,
            "academic_integrity": self.academic_integrity,
            "human_signals_summary": all_human_signals[:10],
            "llm_signals_summary": all_llm_signals[:10],
            "detectors": [
                {
                    "model": v.model,
                    "ai_score": v.ai_score,
                    "veto": v.veto_triggered,
                    "veto_reason": v.veto_reason,
                    "fragments_count": len(v.suspicious_fragments),
                    "fragments_sample": v.suspicious_fragments[:3],
                    "human_signals": v.human_signals[:3],
                    "llm_signals": v.llm_signals[:3],
                    "error": v.error,
                }
                for v in self.panel
            ],
        }


@dataclass(slots=True)
class JudgeResult:
    """Финальный вердикт (после метасудьи или ансамбля)."""

    ai_score: int
    veto_triggered: bool
    veto_reason: str | None
    academic_integrity: str
    suspicious_fragments: list[str]
    judge_status: str  # "OFFLINE" | "ENSEMBLE_ONLY" | "FULL"
    ensemble_panel: list[DetectorVerdict] = field(default_factory=list)
    judge_verdict: DetectorVerdict | None = None
    raw_json: dict[str, Any] | None = None


# --------------------------------------------------------------------------- #
#  Промпты                                                                      #
# --------------------------------------------------------------------------- #


DETECTOR_SYSTEM_PROMPT = """Ты — детектор ИИ-генерированного текста в ансамбле AI Radar.
Текущая дата: {current_date}.

Твоя задача — проанализировать текст работы и оценить вероятность его
генерации LLM (ChatGPT, Claude, DeepSeek, Gemini, Grok, Kimi и др.).

УНИКАЛЬНОСТЬ AI Radar: ты не просто ищешь "синтетический синтаксис" —
ты проводишь **саморефлексивный анализ**. Представь, что ты сам мог
написать этот текст. Насколько похоже на твой собственный стиль
аргументации? Это твой "почерк"? Типичные для LLM паттерны мысли?

Что искать (по приоритету):

1. **Следы генерации LLM (синтаксис и структура)**:
   - механическая однородность глубины абзацев, одинаковая плотность
     аргументации;
   - повторяющиеся клише-маркеры ("Таким образом...", "Важно отметить...",
     "Подводя итог...");
   - сверхправильная структура (введение → N одинаковых абзацев → вывод);
   - "универсальные" обобщения без конкретики.

2. **Семантический разбор — чья логика?**:
   - Насколько аргументация похожа на "LLM-мышление": сбалансированные
     "про и контра", нейтральность, избегание позиции, "правильные"
     но поверхностные выводы?
   - Есть ли галлюцинации (выдуманные ссылки, факты, цитаты)?
   - Нелогичные обобщения, не следующие из контекста?
   - "Безопасная" аргументация, уход от оценок?

3. **Человеческие особенности (их отсутствие = сигнал ИИ)**:
   - Личный опыт, конкретные истории, имена, даты, места?
   - Креативность, нестандартные метафоры, юмор, ирония?
   - Эмоциональная окраска, субъективные оценки?
   - "Шероховатости" — неравномерная глубина, отступления, живые
     рассуждения, сомнения?
   - Авторская позиция (а не "с одной стороны... с другой...")?

Если текст демонстрирует живую человеческую мысль, личный опыт,
креативность — СНИЖАЙ ai_score. Если текст "слишком правильный",
нейтральный, безличный — ПОВЫШАЙ ai_score.

Учитывай переданные тебе метрики Сигнала 1: если статистика указывает
на синтетику (низкий burstiness, низкая TTR, высокая плотность клише) —
усиливай подозрение. Но если статистика нейтральна, а смысл живой —
доверяй семантике.

Верни ТОЛЬКО валидный JSON (без markdown-обёртки, без пояснений):
{{
  "ai_score": <целое 0..100>,
  "veto_triggered": <true|false>,
  "veto_reason": <строка или null>,
  "academic_integrity": <"ПОДТВЕРЖДЕНА" | "НАРУШЕНА">,
  "suspicious_fragments": [<строка>, ...],
  "human_signals": [<строка — найденные признаки живой человеческой мысли>, ...],
  "llm_signals": [<строка — найденные признаки LLM-генерации>, ...]
}}

`suspicious_fragments` — короткие подстроки из текста (1-3 предложения),
которые иллюстрируют подозрения. Не более 10.
`human_signals` / `llm_signals` — списки коротких описаний (до 5 пунктов).
"""


DETECTOR_USER_TEMPLATE = """Проанализируй текст работы на признаки ИИ-генерации.

Текущая дата: {current_date}.

=== МАТРИЦА МЕТРИК СИГНАЛА 1 ===
{metrics_json}

=== ИНТЕРПРЕТАЦИЯ СИГНАЛА 1 ===
{interpretation}

=== ТЕКСТ РАБОТЫ ===
{text}

Верни строго JSON согласно схеме из системного промпта.
"""


JUDGE_SYSTEM_PROMPT = """Ты — Метасудья AI Radar: старший эксперт-филолог, выносящий финальный вердикт
по результатам работы ансамбля детекторов ИИ-контента.
Текущая дата: {current_date}.

ВАЖНО: ты НЕ получаешь исходный текст работы. Вместо этого тебе передаётся:
  1. Сводка метрик Сигнала 1 (статистический анализ текста).
  2. Вердикты всех моделей-детекторов ансамбля (их ai_score, veto-флаги,
     подозрительные фрагменты, признаки человека/ИИ).

Твоя задача — критически оценить согласованность ансамбля, выявить
возможные ошибки отдельных детекторов (слишком мягкие/жёсткие) и вынести
финальный вердикт.

Особое внимание удели:
  - Если несколько детекторов независимо нашли одни и те же признаки
    LLM-генерации (особенно отсутствие личного опыта, "нейтральность"
    аргументации, синтетические клише) — это сильный сигнал.
  - Если детекторы нашли человеческие признаки (личный опыт, креативность,
    живая аргументация) — это снижает вероятность ИИ.
  - Согласованность по ai_score: если разброс между детекторами большой,
    финальный скор должен быть ближе к медиане, но с учётом veto-голосов.

Ты обладаешь ПРАВОМ БЛОКИРУЮЩЕГО ВЕТО: если ансамбль или Сигнал 1
указывают на критические логико-структурные аномалии генерации, поставь
`veto_triggered = true` и обоснуй причину в `veto_reason`. При ВЕТО
академическая честность считается НАРУШЕННОЙ независимо от итогового
процента. И наоборот: если есть явные признаки живой человеческой работы
(личный опыт, креативность, шероховатости) — не активируй ВЕТО даже
при умеренном ai_score.

Верни ТОЛЬКО валидный JSON (без markdown-обёртки):
{{
  "ai_score": <целое 0..100 — финальный скор>,
  "veto_triggered": <true|false>,
  "veto_reason": <строка или null>,
  "academic_integrity": <"ПОДТВЕРЖДЕНА" | "НАРУШЕНА">,
  "suspicious_fragments": [<строка>, ...]
}}

`suspicious_fragments` — фрагменты, на которые опирается твой вердикт
(могут быть взяты из вердиктов ансамбля). Не более 10.
"""


JUDGE_USER_TEMPLATE = """Вынеси финальный вердикт по результатам работы ансамбля детекторов.

Текущая дата: {current_date}.

=== СВОДКА СИГНАЛА 1 (СТАТИСТИКА) ===
stat_score: {stat_score} / 100
interpretation: {interpretation}

=== ВЕРДИКТЫ АНСАМБЛЯ ДЕТЕКТОРОВ ===
{ensemble_summary_json}

=== ПОДОЗРИТЕЛЬНЫЕ ФРАГМЕНТЫ (объединённые из ансамбля) ===
{fragments}

Вынеси финальный вердикт. Верни строго JSON согласно схеме.
"""


# --------------------------------------------------------------------------- #
#  Утилиты                                                                     #
# --------------------------------------------------------------------------- #


_JSON_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def _find_balanced_json(raw: str, start_idx: int) -> str | None:
    """Найти сбалансированный JSON-объект начиная с позиции start_idx."""
    depth = 0
    in_string = False
    escape = False
    for i in range(start_idx, len(raw)):
        c = raw[i]
        if escape:
            escape = False
            continue
        if c == "\\" and in_string:
            escape = True
            continue
        if c == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return raw[start_idx : i + 1]
    return None


def _extract_json(raw: str) -> dict[str, Any]:
    """Извлечь JSON-объект из ответа LLM.

    Стратегии (по порядку):
      1. Markdown-обёртка ```json ... ```
      2. Обычный json.loads от первого { до последнего }
      3. Сбалансированный поиск скобок (для обрезанных ответов)
      4. Regex-извлечение полей (fallback)
    """
    if not raw:
        raise ValueError("empty LLM response")

    # 1. Markdown-обёртка
    m = _JSON_FENCE_RE.match(raw.strip())
    if m:
        raw = m.group(1)

    start = raw.find("{")
    if start == -1:
        raise ValueError(f"no JSON object found in LLM response: {raw[:200]!r}")

    end = raw.rfind("}")
    if end != -1 and end > start:
        # 2. Обычный парсинг
        candidate = raw[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # 3. Сбалансированный поиск (для обрезанных/вложенных)
    balanced = _find_balanced_json(raw, start)
    if balanced:
        try:
            return json.loads(balanced)
        except json.JSONDecodeError:
            pass

    # 4. Regex-fallback: извлечь поля по одному
    result = _extract_json_via_regex(raw[start:])
    if result is not None:
        return result

    raise ValueError(f"failed to parse JSON from LLM response: {raw[:300]!r}")


def _extract_json_via_regex(raw: str) -> dict[str, Any] | None:
    """Fallback: извлечь поля JSON через regex, если парсинг сломан."""
    result: dict[str, Any] = {}

    # ai_score
    m = re.search(r'"ai_score"\s*:\s*(\d+)', raw)
    if m:
        result["ai_score"] = int(m.group(1))

    # veto_triggered
    m = re.search(r'"veto_triggered"\s*:\s*(true|false)', raw, re.IGNORECASE)
    if m:
        result["veto_triggered"] = m.group(1).lower() == "true"

    # veto_reason
    m = re.search(r'"veto_reason"\s*:\s*(?:"((?:[^"\\]|\\.)*)"|null)', raw, re.DOTALL)
    if m:
        result["veto_reason"] = m.group(1) if m.group(1) else None

    # academic_integrity
    m = re.search(r'"academic_integrity"\s*:\s*"([^"]*)"', raw)
    if m:
        result["academic_integrity"] = m.group(1)

    # suspicious_fragments — извлечь все строки в массиве
    fragments = re.findall(r'"suspicious_fragments"\s*:\s*\[(.*?)\]', raw, re.DOTALL)
    if fragments:
        frags = re.findall(r'"((?:[^"\\]|\\.)*)"', fragments[0])
        result["suspicious_fragments"] = frags

    # human_signals
    human = re.findall(r'"human_signals"\s*:\s*\[(.*?)\]', raw, re.DOTALL)
    if human:
        result["human_signals"] = re.findall(r'"((?:[^"\\]|\\.)*)"', human[0])

    # llm_signals
    llm = re.findall(r'"llm_signals"\s*:\s*\[(.*?)\]', raw, re.DOTALL)
    if llm:
        result["llm_signals"] = re.findall(r'"((?:[^"\\]|\\.)*)"', llm[0])

    if "ai_score" in result:
        return result
    return None


def _truncate_for_llm(text: str, max_chars: int = 24000) -> str:
    """Обрезать текст для одного LLM-вызова (устаревшее — используется как fallback)."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n\n[...промежуточный текст опущен...]\n\n" + text[-half:]


def _chunk_text(text: str, max_chars: int = 20000) -> list[str]:
    """Разбить длинный текст на чанки по предложениям.

    Возвращает список чанков. Если текст короче max_chars — один чанк.
    Каждый чанк не превышает max_chars.
    """
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    sentences = re.split(r'(?<=[.!?…])\s+', text)
    current = ""
    for sent in sentences:
        if len(current) + len(sent) + 1 <= max_chars:
            current = (current + " " + sent).strip() if current else sent
        else:
            if current:
                chunks.append(current)
            # Если одно предложение длиннее max_chars — разрезаем по max_chars
            while len(sent) > max_chars:
                chunks.append(sent[:max_chars])
                sent = sent[max_chars:]
            current = sent
    if current:
        chunks.append(current)
    return chunks


def _format_chunks_for_llm(text: str, max_chars: int = 20000) -> str:
    """Отформатировать текст для LLM: если длинный — разбить на чанки с пометками."""
    chunks = _chunk_text(text, max_chars)
    if len(chunks) == 1:
        return chunks[0]

    parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"=== ЧАСТЬ {i} ИЗ {len(chunks)} ===\n{chunk}\n")
    parts.append(
        f"\n[ВНИМАНИЕ: Текст разбит на {len(chunks)} частей из-за длины. "
        "Проанализируй все части целиком и вынеси единый вердикт.]"
    )
    return "\n".join(parts)


def _current_date() -> str:
    """Текущая дата в человекочитаемом формате для передачи в LLM."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _build_verdict(provider_name: str, model: str, parsed: dict[str, Any], raw: str = "") -> DetectorVerdict:
    ai_score = int(parsed.get("ai_score", 0))
    ai_score = max(0, min(100, ai_score))
    veto = bool(parsed.get("veto_triggered", False))
    veto_reason = parsed.get("veto_reason")
    if isinstance(veto_reason, (list, dict)):
        veto_reason = json.dumps(veto_reason, ensure_ascii=False)
    integrity_raw = str(parsed.get("academic_integrity", "")).strip().upper()
    integrity = "НАРУШЕНА" if "НАРУШЕНА" in integrity_raw or veto else "ПОДТВЕРЖДЕНА"
    fragments = parsed.get("suspicious_fragments", [])
    if not isinstance(fragments, list):
        fragments = []
    fragments = [str(f) for f in fragments][:20]
    human_signals = parsed.get("human_signals", [])
    if not isinstance(human_signals, list):
        human_signals = []
    human_signals = [str(s) for s in human_signals][:5]
    llm_signals = parsed.get("llm_signals", [])
    if not isinstance(llm_signals, list):
        llm_signals = []
    llm_signals = [str(s) for s in llm_signals][:5]
    return DetectorVerdict(
        provider_name=provider_name,
        model=model,
        ai_score=ai_score,
        veto_triggered=veto,
        veto_reason=veto_reason,
        academic_integrity=integrity,
        suspicious_fragments=fragments,
        human_signals=human_signals,
        llm_signals=llm_signals,
        raw=raw,
    )


# --------------------------------------------------------------------------- #
#  Уровень 1: один детектор ансамбля                                            #
# --------------------------------------------------------------------------- #


async def _call_detector(
    provider: LlmProvider,
    text: str,
    signal1: Signal1Result,
    timeout_sec: int,
) -> DetectorVerdict:
    """Вызвать одну модель-детектор ансамбля."""
    user_message = DETECTOR_USER_TEMPLATE.format(
        current_date=_current_date(),
        metrics_json=json.dumps(signal1.metrics.model_dump(), ensure_ascii=False, indent=2),
        interpretation=signal1.interpretation,
        text=_format_chunks_for_llm(text),
    )
    return await _call_llm_json(
        provider=provider,
        system_prompt=DETECTOR_SYSTEM_PROMPT.format(current_date=_current_date()),
        user_message=user_message,
        timeout_sec=timeout_sec,
        provider_label=provider.name,
    )


async def _call_judge(
    provider: LlmProvider,
    signal1: Signal1Result,
    ensemble: EnsembleResult,
    timeout_sec: int,
) -> DetectorVerdict:
    """Вызвать метасудью с вердиктами ансамбля (без исходного текста)."""
    fragments = []
    seen: set[str] = set()
    for v in ensemble.panel:
        for f in v.suspicious_fragments:
            k = f.strip().lower()
            if k and k not in seen:
                fragments.append(f)
                seen.add(k)
            if len(fragments) >= 15:
                break
        if len(fragments) >= 15:
            break
    user_message = JUDGE_USER_TEMPLATE.format(
        current_date=_current_date(),
        stat_score=signal1.stat_score,
        interpretation=signal1.interpretation,
        ensemble_summary_json=json.dumps(ensemble.to_summary(), ensure_ascii=False, indent=2),
        fragments="\n".join(f"- {f}" for f in fragments) or "(фрагменты не предоставлены)",
    )
    return await _call_llm_json(
        provider=provider,
        system_prompt=JUDGE_SYSTEM_PROMPT.format(current_date=_current_date()),
        user_message=user_message,
        timeout_sec=timeout_sec,
        provider_label=f"judge:{provider.name}",
    )


async def _call_llm_json(
    *,
    provider: LlmProvider,
    system_prompt: str,
    user_message: str,
    timeout_sec: int,
    provider_label: str,
) -> DetectorVerdict:
    """Универсальный OpenAI-compatible вызов с JSON-ответом."""
    payload = {
        "model": provider.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.0,
        "max_tokens": 4000,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {provider.api_key}",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=timeout_sec) as client:
            url = provider.base_url.rstrip("/") + "/chat/completions"
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as exc:
        log.warning("LLM %s HTTP error: %s", provider_label, exc)
        return DetectorVerdict(
            provider_name=provider.name, model=provider.model,
            ai_score=0, veto_triggered=False, veto_reason=None,
            academic_integrity="ПОДТВЕРЖДЕНА", error=f"http: {exc!s}",
        )
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        return DetectorVerdict(
            provider_name=provider.name, model=provider.model,
            ai_score=0, veto_triggered=False, veto_reason=None,
            academic_integrity="ПОДТВЕРЖДЕНА",
            error=f"malformed response: {exc!s}", raw=str(data)[:500],
        )
    # content может быть None (некоторые провайдеры возвращают пустое message)
    if content is None:
        return DetectorVerdict(
            provider_name=provider.name, model=provider.model,
            ai_score=0, veto_triggered=False, veto_reason=None,
            academic_integrity="ПОДТВЕРЖДЕНА",
            error="empty content in LLM response",
            raw=str(data)[:500],
        )
    try:
        parsed = _extract_json(content)
    except (json.JSONDecodeError, ValueError) as exc:
        return DetectorVerdict(
            provider_name=provider.name, model=provider.model,
            ai_score=0, veto_triggered=False, veto_reason=None,
            academic_integrity="ПОДТВЕРЖДЕНА",
            error=f"json parse: {exc!s}", raw=content[:500],
        )
    return _build_verdict(provider.name, provider.model, parsed, raw=content)


# --------------------------------------------------------------------------- #
#  Агрегация ансамбля                                                          #
# --------------------------------------------------------------------------- #


def _aggregate_ensemble(verdicts: list[DetectorVerdict], veto_threshold: int) -> EnsembleResult:
    valid = [v for v in verdicts if v.error is None]
    if not valid:
        return EnsembleResult(
            ai_score=0, veto_triggered=False, veto_reason=None,
            academic_integrity="ПОДТВЕРЖДЕНА", suspicious_fragments=[],
            panel=verdicts, available=False,
        )
    scores = sorted(v.ai_score for v in valid)
    median_score = scores[len(scores) // 2]
    veto_votes = sum(1 for v in valid if v.veto_triggered)
    veto_triggered = veto_votes >= max(1, len(valid) // 2) or median_score >= veto_threshold
    veto_reasons = [v.veto_reason for v in valid if v.veto_triggered and v.veto_reason]
    veto_reason = " | ".join(veto_reasons) if veto_reasons else None
    if veto_triggered and not veto_reason:
        veto_reason = (
            f"Ансамбль: median ai_score={median_score}, порог={veto_threshold}, "
            f"veto-голосов={veto_votes}/{len(valid)}."
        )
    integrity = "НАРУШЕНА" if veto_triggered or median_score >= veto_threshold else "ПОДТВЕРЖДЕНА"
    fragments: list[str] = []
    seen: set[str] = set()
    for v in valid:
        for f in v.suspicious_fragments:
            k = f.strip().lower()
            if k and k not in seen:
                fragments.append(f)
                seen.add(k)
    return EnsembleResult(
        ai_score=median_score, veto_triggered=veto_triggered, veto_reason=veto_reason,
        academic_integrity=integrity, suspicious_fragments=fragments[:20],
        panel=verdicts, available=True,
    )


# --------------------------------------------------------------------------- #
#  Оркестратор                                                                  #
# --------------------------------------------------------------------------- #


async def run_judge(
    text: str,
    signal1: Signal1Result,
    storage: Storage,
) -> JudgeResult:
    """Полный конвейер Сигнала 2: ансамбль → метасудья."""
    s = get_settings()
    veto_threshold = await storage.get_veto_threshold()
    timeout = s.judge_http_timeout_sec
    max_retries = await storage.get_judge_max_retries()

    # --- Уровень 1: ансамбль ---
    ensemble_providers = await storage.list_enabled_llm_providers(role="ensemble")
    judge_providers = await storage.list_enabled_llm_providers(role="judge")

    if not ensemble_providers and not judge_providers:
        return _offline(signal1)

    if not ensemble_providers:
        # Есть судья, но нет ансамбля — судья не может работать без данных.
        # Деградируем на Сигнале 1, но помечаем что судья подключён.
        return _offline(signal1, note="судья подключён, но ансамбль пуст")

    # Параллельный опрос детекторов.
    # return_exceptions=True — один упавший детектор не должен ронять весь ансамбль.
    tasks = [_call_detector(p, text, signal1, timeout) for p in ensemble_providers]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    verdicts: list[DetectorVerdict] = []
    for i, res in enumerate(raw_results):
        if isinstance(res, Exception):
            log.warning("Детектор %s упал: %s", ensemble_providers[i].name, res)
            verdicts.append(DetectorVerdict(
                provider_name=ensemble_providers[i].name,
                model=ensemble_providers[i].model,
                ai_score=0, veto_triggered=False, veto_reason=None,
                academic_integrity="ПОДТВЕРЖДЕНА",
                error=f"exception: {res!s}",
            ))
        else:
            verdicts.append(res)

    # Retry невалидных
    if max_retries > 0:
        for _ in range(max_retries):
            bad = [(i, v) for i, v in enumerate(verdicts) if v.error is not None]
            if not bad:
                break
            retry_tasks = [_call_detector(ensemble_providers[i], text, signal1, timeout) for i, _ in bad]
            retry_raw = await asyncio.gather(*retry_tasks, return_exceptions=True)
            for (i, _), r in zip(bad, retry_raw):
                if isinstance(r, Exception):
                    continue
                if r.error is None:
                    verdicts[i] = r

    ensemble = _aggregate_ensemble(verdicts, veto_threshold)
    if not ensemble.available:
        # Все детекторы упали
        if judge_providers:
            return _offline(signal1, note="ансамбль упал, судья не запущен")
        return _offline(signal1, note="ансамбль упал")

    # --- Уровень 2: метасудья (опционально) ---
    if not judge_providers:
        # Судьи нет — финал = агрегат ансамбля
        return JudgeResult(
            ai_score=ensemble.ai_score,
            veto_triggered=ensemble.veto_triggered,
            veto_reason=ensemble.veto_reason,
            academic_integrity=ensemble.academic_integrity,
            suspicious_fragments=ensemble.suspicious_fragments,
            judge_status="ENSEMBLE_ONLY",
            ensemble_panel=ensemble.panel,
            judge_verdict=None,
            raw_json={
                "ensemble": ensemble.to_summary(),
                "judge": None,
            },
        )

    # Вызов метасудьи (берём первого активного).
    # Защита от исключений — судья не должен ронять весь конвейер.
    judge_provider = judge_providers[0]
    try:
        judge_v = await _call_judge(judge_provider, signal1, ensemble, timeout)
    except Exception as exc:
        log.exception("Метасудья упал с исключением: %s", exc)
        judge_v = DetectorVerdict(
            provider_name=judge_provider.name, model=judge_provider.model,
            ai_score=0, veto_triggered=False, veto_reason=None,
            academic_integrity="ПОДТВЕРЖДЕНА",
            error=f"exception: {exc!s}",
        )

    # Retry судьи
    if judge_v.error is not None and max_retries > 0:
        for _ in range(max_retries):
            try:
                rv = await _call_judge(judge_provider, signal1, ensemble, timeout)
            except Exception as exc:
                continue
            if rv.error is None:
                judge_v = rv
                break

    if judge_v.error is not None:
        # Судья упал — используем ансамбль
        return JudgeResult(
            ai_score=ensemble.ai_score,
            veto_triggered=ensemble.veto_triggered,
            veto_reason=ensemble.veto_reason,
            academic_integrity=ensemble.academic_integrity,
            suspicious_fragments=ensemble.suspicious_fragments,
            judge_status="ENSEMBLE_ONLY",
            ensemble_panel=ensemble.panel,
            judge_verdict=judge_v,
            raw_json={
                "ensemble": ensemble.to_summary(),
                "judge": {"error": judge_v.error, "raw": judge_v.raw[:500]},
            },
        )

    # Судья отработал — финальный вердикт от него
    # Фрагменты: приоритет у судьи, дополняем ансамблем
    fragments: list[str] = []
    seen: set[str] = set()
    for f in judge_v.suspicious_fragments + ensemble.suspicious_fragments:
        k = f.strip().lower()
        if k and k not in seen:
            fragments.append(f)
            seen.add(k)
        if len(fragments) >= 20:
            break

    # VETO: судья может наложить вето независимо; также учитываем ансамбль
    veto_triggered = judge_v.veto_triggered or ensemble.veto_triggered
    veto_reason = judge_v.veto_reason or ensemble.veto_reason
    if veto_triggered and not veto_reason:
        veto_reason = f"ВЕТО метасудьи (score={judge_v.ai_score})."

    integrity = "НАРУШЕНА" if veto_triggered or judge_v.ai_score >= veto_threshold else "ПОДТВЕРЖДЕНА"

    return JudgeResult(
        ai_score=judge_v.ai_score,
        veto_triggered=veto_triggered,
        veto_reason=veto_reason,
        academic_integrity=integrity,
        suspicious_fragments=fragments,
        judge_status="FULL",
        ensemble_panel=ensemble.panel,
        judge_verdict=judge_v,
        raw_json={
            "ensemble": ensemble.to_summary(),
            "judge": {
                "model": judge_v.model,
                "ai_score": judge_v.ai_score,
                "veto": judge_v.veto_triggered,
                "veto_reason": judge_v.veto_reason,
                "integrity": judge_v.academic_integrity,
                "fragments": judge_v.suspicious_fragments,
            },
        },
    )


def _offline(signal1: Signal1Result, note: str = "нет активных LLM") -> JudgeResult:
    """Degraded-режим: вердикт на базе Сигнала 1, без VETO."""
    return JudgeResult(
        ai_score=signal1.stat_score,
        veto_triggered=False,
        veto_reason=None,
        academic_integrity="ПОДТВЕРЖДЕНА",
        suspicious_fragments=signal1.suspicious_stat_fragments[:20],
        judge_status="OFFLINE",
        ensemble_panel=[],
        judge_verdict=None,
        raw_json={"offline_reason": note, "stat_score": signal1.stat_score},
    )


async def judge_status(storage: Storage) -> dict[str, Any]:
    """Статус системы Сигнала 2 для UI."""
    ensemble = await storage.list_enabled_llm_providers(role="ensemble")
    judges = await storage.list_enabled_llm_providers(role="judge")
    if ensemble and judges:
        return {
            "status": "FULL",
            "label": f"ANSAMBLE+JUDGE ({len(ensemble)}+{len(judges)})",
            "ensemble_count": len(ensemble),
            "judge_count": len(judges),
        }
    if ensemble:
        return {
            "status": "ENSEMBLE_ONLY",
            "label": f"ENSEMBLE ONLY ({len(ensemble)})",
            "ensemble_count": len(ensemble),
            "judge_count": 0,
        }
    return {
        "status": "OFFLINE",
        "label": "OFFLINE",
        "ensemble_count": 0,
        "judge_count": len(judges),
    }
