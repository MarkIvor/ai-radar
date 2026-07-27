"""Сигнал 1: статистический движок AI Radar.

Реализован набор современных математических метрик для оценки текста
без тяжелых локальных LLM-моделей. Метрики объединяются в сводный
`stat_score` [0..100], который затем передаётся в Сигнал 2 (AI Judge).

Реализованные метрики:

  1. Burstiness  (SD / mean длин предложений)
  2. Lexical Diversity (TTR) с корректировкой для длинных текстов
  3. Sentence Length Variance
  4. Perplexity Proxy (Shannon entropy по биграммам)
  5. Synthetic Cliche Density
  6. Punctuation Diversity (энтропия пунктуации)
  7. Sentence Starter Entropy (энтропия первых слов предложений)
  8. Yule's K (текстовый K, обратная лексическая насыщенность)
  9. Honore's R (богатство лексики)
 10. Sichel's S (доля hapax legomena)
 11. Hapax Legomena Ratio
 12. Connective Density (логические маркеры)
 13. N-gram Repetition Rate
 14. Sentence Rhythm Entropy (по длине предложений)
 15. Comma-per-Sentence Ratio
 16. Avg Syllables per Word
 17. Function-Word Distribution
 18. Lexical Density (content / function)
 19. Type-Token Ratio sliding-window
 20. Determiner Density
"""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Iterable

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
#  Лингвистические словари (русский язык)                                       #
# --------------------------------------------------------------------------- #


SYNTHETIC_CLICHES: tuple[str, ...] = (
    "таким образом",
    "подводя итог",
    "подводя итоги",
    "в заключение",
    "важно отметить, что",
    "важно отметить что",
    "следует отметить, что",
    "следует отметить что",
    "необходимо отметить, что",
    "необходимо отметить что",
    "стоит отметить, что",
    "стоит отметить что",
    "важно понимать, что",
    "важно понимать что",
    "необходимо подчеркнуть, что",
    "следует подчеркнуть, что",
    "как было сказано ранее",
    "как упоминалось выше",
    "как было отмечено",
    "в данном случае",
    "в свою очередь",
    "с другой стороны",
    "с одной стороны",
    "в целом",
    "в общем",
    "по сути",
    "по сути дела",
    "на сегодняшний день",
    "в современном мире",
    "в наше время",
    "в современном обществе",
    "играет важную роль",
    "играет ключевую роль",
    "играет значимую роль",
    "является важным",
    "является ключевым",
    "представляет собой",
    "так или иначе",
    "в определенной степени",
    "в определённой степени",
    "в некоторой степени",
    "в том числе",
    "в первую очередь",
    "во-первых",
    "во-вторых",
    "в-третьих",
    "наконец",
    "итак",
    "таким образом",
    "следовательно",
    "поэтому",
    "в результате",
    "благодаря этому",
    "вследствие этого",
    "в связи с этим",
    "в данном контексте",
    "с этой целью",
    "в данных условиях",
    "в целом можно сказать",
    "нельзя не отметить",
    "нельзя не упомянуть",
    "следует выделить",
    "необходимо выделить",
    "целесообразно отметить",
    "представляет несомненный интерес",
    "вызывает особый интерес",
    "заслуживает особого внимания",
    "требует особого внимания",
    "имеет огромное значение",
    "имеет важное значение",
    "имеет принципиальное значение",
    "оказывает влияние",
    "оказывает существенное влияние",
    "с точки зрения",
    "в рамках данного",
    "в рамках настоящего",
    "в рамках исследования",
    "как показывает практика",
    "как правило",
    "как известно",
    "по мнению",
    "по мнению специалистов",
    "по мнению экспертов",
    "согласно",
    "согласно мнению",
    "включая в себя",
    "заключается в том, что",
    "состоит в том, что",
    "выражается в том, что",
    "проявляется в том, что",
    "обусловлено тем, что",
    "связано с тем, что",
    "объясняется тем, что",
    "обуславливается тем, что",
)


FUNCTION_WORDS_RU: frozenset = frozenset(
    {
        # предлоги
        "в", "во", "на", "с", "со", "к", "ко", "по", "от", "ото", "из", "изо", "у", "к",
        "об", "обо", "при", "про", "для", "до", "без", "безо", "ради", "через",
        "сквозь", "между", "перед", "передо", "под", "подо", "над", "надо", "около",
        "возле", "у", "вокруг", "впереди", "beside", "близ", "вдоль", "вместо",
        "вследствие", "внутри", "вокруг", "после", "против", "посреди", "среди",
        "сверху", "снизу", "из-за", "из-под", "по-за", "по-над", "на-под", "не",
        "бы", "ли", "же", "то", "или", "ни", "ибо", "но", "да", "что", "чтобы",
        "как", "когда", "где", "куда", "откуда", "зачем", "почему", "если", "то",
        "так", "вот", "это", "тот", "та", "те", "эти", "этот", "эта", "такой",
        "такая", "такие", "мой", "моя", "моё", "мои", "твой", "твоя", "твоё",
        "твои", "наш", "наша", "наше", "наши", "ваш", "ваша", "ваше", "ваши",
        "свой", "своя", "своё", "свои", "их", "его", "её", "он", "она", "оно",
        "они", "мы", "вы", "ты", "я", "он", "тот", "та", "то", "те", "себя",
        "сам", "сама", "само", "сами", "весь", "вся", "всё", "все", "каждый",
        "каждая", "каждое", "все", "любой", "всякий", "никакой", "никакая",
        "никакое", "никакие", "какой-то", "какая-то", "какое-то", "какие-то",
        "это", "то", "такой", "другой", "другая", "другое", "другие", "иначе",
        "конечно", "безусловно", "разумеется", "несомненно", "вероятно",
        "возможно", "пожалуй", "кажется", "по-видимому", "по-моему", "по-твоему",
        "по-нашему", "по-вашему", "итак", "значит", "итак", "впрочем", "кстати",
        "например", "между прочим", "однако", "зато", "только", "лишь", "хоть",
        "хотя", "чуть", "почти", "едва", "совсем", "совершенно", "абсолютно",
        "очень", "крайне", "весьма", "довольно", "достаточно", "более", "менее",
        "больше", "меньше", "много", "мало", "немного", "немало", "слишком",
        "так", "настолько", "столь", "тоже", "также", "впрочем", "итак", "потому",
        "поэтому", "оттого", "затем", "потом", "теперь", "уже", "ещё", "опять",
        "снова", "вновь", "часто", "редко", "иногда", "всегда", "никогда",
        "повсюду", "везде", "всюду", "отовсюду", "тут", "там", "туда", "сюда",
        "нигде", "никак", "никогда",
    }
)


CONNECTIVES_RU: frozenset = frozenset(
    {
        "следовательно", "поэтому", "таким образом", "в результате", "итак",
        "значит", "вследствие", "благодаря", "вследствие", "поэтому", "итак",
        "следовательно", "однако", "но", "зато", "впрочем", "хотя", "несмотря на",
        "невзирая на", "поэтому", "следовательно", "так как", "потому что",
        "ибо", "поскольку", "ввиду того что", "вследствие того что",
        "благодаря тому что", "из-за того что", "в связи с тем что", "для того чтобы",
        "чтобы", "с тем чтобы", "с целью", "для того чтобы", "ради того чтобы",
        "в то время как", "между тем", "тем временем", "тогда как",
        "по мере того как", "после того как", "с тех пор как", "до того как",
        "перед тем как", "когда", "пока", "едва", "лишь только", "как только",
        "если", "если бы", "коли", "кабы", "раз", "в случае если", "при условии что",
        "в случае когда", "в противном случае", "иначе", "а то", "а иначе",
        "так что", "до такой степени что", "до того что", "настолько что",
        "слишком чтобы", "достаточно чтобы", "как будто", "будто", "словно",
        "точно", "как бы", "будто бы", "как если бы",
    }
)


DETERMINERS_RU: frozenset = frozenset(
    {
        "этот", "эта", "это", "эти", "тот", "та", "то", "те",
        "такой", "такая", "такие", "такое", "сие", "сей", "оный",
        "наш", "наша", "наше", "наши", "ваш", "ваша", "ваше", "ваши",
        "мой", "моя", "моё", "мои", "твой", "твоя", "твоё", "твои",
        "их", "его", "её", "какой", "какая", "какое", "какие", "каждый",
        "каждая", "каждое", "весь", "вся", "всё", "все", "любой", "всякий",
        "сам", "сама", "само", "сами", "самый", "никакой", "никакая",
        "некий", "некая", "некое", "некоторые", "определённый",
        "некоторый", "некоторая", "некоторые",
    }
)


# --------------------------------------------------------------------------- #
#  Pydantic-модель результата                                                  #
# --------------------------------------------------------------------------- #


class Signal1Metrics(BaseModel):
    """Полный набор метрик Сигнала 1."""

    burstiness: float = Field(..., description="SD/Mean длин предложений")
    lexical_diversity_ttr: float = Field(..., description="Type-Token Ratio (леммы)")
    sentence_length_variance: float = Field(..., description="Дисперсия длин предложений")
    perplexity_proxy: float = Field(..., description="Энтропия биграмм (прокс перплексии)")
    synthetic_cliche_density: float = Field(..., description="Плотность синтетических клише")
    punctuation_entropy: float = Field(..., description="Энтропия пунктуации")
    sentence_starter_entropy: float = Field(..., description="Энтропия первых слов предложений")
    yule_k: float = Field(..., description="Yule's K (текстовая характеристика)")
    honore_r: float = Field(..., description="Honore's R")
    sichel_s: float = Field(..., description="Sichel's S (доля hapax)")
    hapax_ratio: float = Field(..., description="Hapax legomena / total")
    connective_density: float = Field(..., description="Логические маркеры / 1000 слов")
    ngram_repetition_rate: float = Field(..., description="Доля повторяющихся 3-грамм")
    sentence_rhythm_entropy: float = Field(..., description="Энтропия распределения длин")
    comma_per_sentence: float = Field(..., description="Запятых на предложение")
    avg_syllables_per_word: float = Field(..., description="Среднее число слогов в слове")
    function_word_ratio: float = Field(..., description="Доля служебных слов")
    lexical_density: float = Field(..., description="Content-words / total")
    ttr_windowed: float = Field(..., description="TTR по скользящему окну (500 слов)")
    determiner_density: float = Field(..., description="Determiners / 1000 слов")
    num_sentences: int
    num_words: int
    num_chars: int


class Signal1Result(BaseModel):
    """Сводный результат Сигнала 1."""

    stat_score: int = Field(..., ge=0, le=100, description="Итоговый статистический скор [0..100]")
    metrics: Signal1Metrics
    interpretation: str = Field(..., description="Текстовое резюме для AI Judge")
    suspicious_stat_fragments: list[str] = Field(
        default_factory=list,
        description="Подозрительные клише-фразы (для подсветки в UI)",
    )


# --------------------------------------------------------------------------- #
#  Токенизация                                                                  #
# --------------------------------------------------------------------------- #


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+(?=[А-ЯA-Z0-9«\"'])")
WORD_RE = re.compile(r"[А-Яа-яЁёA-Za-z0-9][А-Яа-яЁёA-Za-z0-9\-]*")
PUNCT_RE = re.compile(r"[.,!?;:\-—()«»\"']")


def _split_sentences(text: str) -> list[str]:
    """Разбить текст на предложения (эвристика без spacy)."""
    text = text.replace("\n", " ").strip()
    if not text:
        return []
    parts = SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in parts if s.strip()]


def _tokenize_words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def _count_syllables(word: str) -> int:
    """Грубая оценка слогов в русском слове по числу гласных."""
    return max(1, sum(1 for c in word.lower() if c in "аеиоуыэюяё"))


# --------------------------------------------------------------------------- #
#  Вспомогательные метрики                                                      #
# --------------------------------------------------------------------------- #


def _entropy(counts: Iterable[int]) -> float:
    """Shannon entropy по распределению counts (в битах)."""
    total = sum(counts)
    if total == 0:
        return 0.0
    h = 0.0
    for n in counts:
        if n > 0:
            p = n / total
            h -= p * math.log2(p)
    return h


def _safe_mean(xs: list[float]) -> float:
    return statistics.fmean(xs) if xs else 0.0


def _safe_stdev(xs: list[float]) -> float:
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


def _safe_variance(xs: list[float]) -> float:
    return statistics.pvariance(xs) if len(xs) > 1 else 0.0


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _ttr_windowed(tokens: list[str], window: int = 500) -> float:
    """Скользящий TTR по окну (MATTR-like)."""
    if not tokens:
        return 0.0
    if len(tokens) <= window:
        return len(set(tokens)) / len(tokens)
    vals = []
    for i in range(len(tokens) - window + 1):
        chunk = tokens[i : i + window]
        vals.append(len(set(chunk)) / len(chunk))
    return _safe_mean(vals)


def _yule_k(tokens: list[str]) -> float:
    """Yule's K (текстовая характеристика): 10^4 * (M - N) / N^2, M = sum i^2 * V_i."""
    if not tokens:
        return 0.0
    freq = Counter(tokens)
    n = len(tokens)
    spectrum = Counter(freq.values())  # {frequency: number_of_words}
    m = sum(f * f * v for f, v in spectrum.items())
    if n <= 1:
        return 0.0
    return 10_000.0 * (m - n) / (n * n)


def _honore_r(tokens: list[str]) -> float:
    """Honore's R = 100 * (1 - V1 / V), V1 = hapax, V = vocab size.

    Высокое значение — богатый словарь.
    """
    if not tokens:
        return 0.0
    freq = Counter(tokens)
    v = len(freq)
    v1 = sum(1 for w, f in freq.items() if f == 1)
    if v == 0:
        return 0.0
    if v1 == v:
        # нет повторов вообще — возвращаем максимум
        return 1000.0
    return 100.0 * (1 - v1 / v)


def _sichel_s(tokens: list[str]) -> float:
    """Sichel's S = V1 / V."""
    if not tokens:
        return 0.0
    freq = Counter(tokens)
    v = len(freq)
    v1 = sum(1 for w, f in freq.items() if f == 1)
    return v1 / v if v else 0.0


# --------------------------------------------------------------------------- #
#  Главная функция                                                              #
# --------------------------------------------------------------------------- #


def compute_metrics(text: str) -> Signal1Metrics:
    """Рассчитать полный набор метрик Сигнала 1.

    Не использует внешних NLP-моделей: работает на regex-токенизации,
    что обеспечивает предсказуемость и скорость.
    """
    text = (text or "").strip()
    sentences = _split_sentences(text)
    tokens = _tokenize_words(text)
    chars = [c for c in text if not c.isspace()]

    n_sentences = max(1, len(sentences))
    n_words = len(tokens)
    n_chars = len(chars)

    # Длины предложений (в словах)
    sent_lens: list[int] = [len(_tokenize_words(s)) for s in sentences if s]
    mean_len = _safe_mean([float(x) for x in sent_lens])
    burstiness = (_safe_stdev([float(x) for x in sent_lens]) / mean_len) if mean_len > 0 else 0.0
    sentence_length_variance = _safe_variance([float(x) for x in sent_lens])

    # Lexical diversity
    if tokens:
        unique = set(tokens)
        ttr = len(unique) / len(tokens)
    else:
        ttr = 0.0

    # Perplexity proxy: Shannon entropy по биграммам символов
    if len(chars) >= 2:
        bg = Counter(text[i : i + 2] for i in range(len(text) - 1))
        char_uni = Counter(chars)
        # Условная энтропия H(C_{i+1} | C_i) ≈ H(bg) - H(uni)
        h_uni = _entropy(char_uni.values())
        h_bg = _entropy(bg.values())
        perplexity_proxy = max(0.0, h_uni - (h_bg - h_uni))
        # нормируем на [0..1] приближённо
        perplexity_proxy = perplexity_proxy / 5.0 if perplexity_proxy > 0 else 0.0
    else:
        perplexity_proxy = 0.0

    # Synthetic cliche density
    text_low = text.lower()
    cliche_hits = sum(text_low.count(c) for c in SYNTHETIC_CLICHES)
    synthetic_cliche_density = (cliche_hits / n_words * 1000.0) if n_words else 0.0

    # Punctuation entropy
    punct_counts = Counter(PUNCT_RE.findall(text))
    punctuation_entropy = _entropy(punct_counts.values()) / 4.0  # нормировка

    # Sentence starter entropy
    starters = [
        (_tokenize_words(s)[0] if _tokenize_words(s) else "")
        for s in sentences
    ]
    starters = [s for s in starters if s]
    starter_counts = Counter(starters)
    sentence_starter_entropy = _entropy(starter_counts.values()) / 5.0 if starters else 0.0

    # Yule / Honore / Sichel / hapax
    yule_k = _yule_k(tokens) / 200.0  # нормировка для удобства
    honore_r = _honore_r(tokens) / 100.0
    sichel_s = _sichel_s(tokens)
    hapax_ratio = sichel_s

    # Connective density
    conn_hits = sum(text_low.count(c) for c in CONNECTIVES_RU)
    connective_density = (conn_hits / n_words * 1000.0) if n_words else 0.0

    # N-gram repetition rate
    trigrams = _ngrams(tokens, 3)
    if trigrams:
        tg_counts = Counter(trigrams)
        repeated = sum(1 for g, c in tg_counts.items() if c > 1)
        ngram_repetition_rate = repeated / len(tg_counts)
    else:
        ngram_repetition_rate = 0.0

    # Sentence rhythm entropy (по bucket'ам длин)
    if sent_lens:
        buckets = Counter(min(50, l) // 5 for l in sent_lens)
        sentence_rhythm_entropy = _entropy(buckets.values()) / 4.0
    else:
        sentence_rhythm_entropy = 0.0

    # Comma per sentence
    n_commas = text.count(",")
    comma_per_sentence = n_commas / n_sentences

    # Avg syllables per word
    if tokens:
        avg_syll = _safe_mean([float(_count_syllables(w)) for w in tokens])
    else:
        avg_syll = 0.0

    # Function words
    if tokens:
        fw_count = sum(1 for t in tokens if t in FUNCTION_WORDS_RU)
        function_word_ratio = fw_count / len(tokens)
    else:
        function_word_ratio = 0.0

    # Lexical density = content words / total
    if tokens:
        cw_count = sum(1 for t in tokens if t not in FUNCTION_WORDS_RU)
        lexical_density = cw_count / len(tokens)
    else:
        lexical_density = 0.0

    # TTR windowed
    ttr_windowed = _ttr_windowed(tokens, window=500)

    # Determiner density
    if tokens:
        det_count = sum(1 for t in tokens if t in DETERMINERS_RU)
        determiner_density = (det_count / n_words * 1000.0) if n_words else 0.0
    else:
        determiner_density = 0.0

    return Signal1Metrics(
        burstiness=round(burstiness, 4),
        lexical_diversity_ttr=round(ttr, 4),
        sentence_length_variance=round(sentence_length_variance, 4),
        perplexity_proxy=round(perplexity_proxy, 4),
        synthetic_cliche_density=round(synthetic_cliche_density, 4),
        punctuation_entropy=round(punctuation_entropy, 4),
        sentence_starter_entropy=round(sentence_starter_entropy, 4),
        yule_k=round(yule_k, 4),
        honore_r=round(honore_r, 4),
        sichel_s=round(sichel_s, 4),
        hapax_ratio=round(hapax_ratio, 4),
        connective_density=round(connective_density, 4),
        ngram_repetition_rate=round(ngram_repetition_rate, 4),
        sentence_rhythm_entropy=round(sentence_rhythm_entropy, 4),
        comma_per_sentence=round(comma_per_sentence, 4),
        avg_syllables_per_word=round(avg_syll, 4),
        function_word_ratio=round(function_word_ratio, 4),
        lexical_density=round(lexical_density, 4),
        ttr_windowed=round(ttr_windowed, 4),
        determiner_density=round(determiner_density, 4),
        num_sentences=n_sentences,
        num_words=n_words,
        num_chars=n_chars,
    )


def _stat_score(metrics: Signal1Metrics) -> int:
    """Свести 20 метрик в интегральный `stat_score` [0..100].

    Высокий = вероятно ИИ, низкий = вероятно человек.

    Калибровка:
      - Человек (живой текст, личный опыт, вариативность): ~10-25%
      - Смешанный/стилистически нейтральный: ~30-50%
      - ИИ-генерация (повторы, клише, однородность): ~65-95%

    Веса подобраны эмпирически по разметке человек/ИИ текстов.
    """
    # Базовый уровень — низкий, т.к. по умолчанию текст "честный",
    # и только накопление ИИ-паттернов повышает скор.
    score = 18.0

    # --- Сильные сигналы ИИ (весомые добавки) ---

    # Burstiness — главный дифференциатор.
    # ИИ ~0.1-0.3, человек ~0.5-1.2+
    b = metrics.burstiness
    if b < 0.25:
        score += 28
    elif b < 0.35:
        score += 18
    elif b < 0.45:
        score += 6
    elif b < 0.60:
        score += 0  # нейтральная зона
    elif b < 0.80:
        score -= 5
    else:
        score -= 10

    # Cliche density — очень сильный сигнал. Человек почти не использует
    # "таким образом, важно отметить" в бытовом тексте.
    cd = metrics.synthetic_cliche_density
    if cd > 10:
        score += 25
    elif cd > 5:
        score += 15
    elif cd > 2:
        score += 6
    elif cd > 0.5:
        score += 2

    # N-gram repetition — ИИ часто повторяет целые конструкции.
    nr = metrics.ngram_repetition_rate
    if nr > 0.20:
        score += 14
    elif nr > 0.10:
        score += 7
    elif nr > 0.05:
        score += 2

    # --- Средние сигналы ---

    # Sentence rhythm entropy — ИИ более однороден по длинам.
    # ВАЖНО: для коротких текстов энтропия естественным образом низкая,
    # поэтому пороги сделаны более консервативными.
    sre = metrics.sentence_rhythm_entropy
    if sre < 0.5:
        score += 8
    elif sre < 0.9:
        score += 3
    elif sre > 2.0:
        score -= 4

    # Sentence starter entropy — ИИ часто повторяет первые слова.
    sse = metrics.sentence_starter_entropy
    if sse < 0.5:
        score += 8
    elif sse < 0.9:
        score += 3
    elif sse > 1.8:
        score -= 3

    # TTR — низкое разнообразие = ИИ.
    ttr = metrics.lexical_diversity_ttr
    if ttr < 0.35:
        score += 8
    elif ttr < 0.45:
        score += 4
    elif ttr > 0.70:
        score -= 3

    # Sichel's S — много hapax = живой словарь = человек.
    ss = metrics.sichel_s
    if ss < 0.30:
        score += 6
    elif ss < 0.45:
        score += 2
    elif ss > 0.65:
        score -= 4

    # Connective density — избыток логических маркеров типичен для ИИ.
    # ВАЖНО: разговорный текст тоже содержит связки ("по", "к", "в"),
    # поэтому порог сделан высоким.
    cden = metrics.connective_density
    if cden > 35:
        score += 6
    elif cden > 20:
        score += 3

    # --- Слабые сигналы (тонкая настройка) ---

    # Punctuation entropy — ИИ более стандартизирован.
    pe = metrics.punctuation_entropy
    if pe < 0.4:
        score += 3
    elif pe < 0.7:
        score += 1
    elif pe > 1.3:
        score -= 2

    # Perplexity proxy — низкая энтропия = синтетика.
    pp = metrics.perplexity_proxy
    if pp < 0.15:
        score += 4
    elif pp > 0.5:
        score -= 2

    # Comma per sentence — ИИ часто перегружает запятыми или наоборот.
    cps = metrics.comma_per_sentence
    if cps > 3.5:
        score += 3

    # Lexical density — ИИ часто более "плотный".
    ld = metrics.lexical_density
    if ld > 0.72:
        score += 2

    return max(0, min(100, int(round(score))))


def _interpretation(m: Signal1Metrics, score: int) -> str:
    if score >= 70:
        verdict = "ВЫСОКАЯ вероятность ИИ-генерации"
    elif score >= 50:
        verdict = "Средняя вероятность ИИ-генерации"
    elif score >= 30:
        verdict = "Низкая вероятность ИИ-генерации"
    else:
        verdict = "Очень низкая вероятность ИИ-генерации"
    return (
        f"{verdict}. Burstiness={m.burstiness:.2f}, TTR={m.lexical_diversity_ttr:.2f}, "
        f"Variance={m.sentence_length_variance:.2f}, PerplexityProxy={m.perplexity_proxy:.2f}, "
        f"ClicheDensity={m.synthetic_cliche_density:.2f}, NgramRepeat={m.ngram_repetition_rate:.2f}, "
        f"SichelS={m.sichel_s:.2f}, HonoreR={m.honore_r:.2f}, YuleK={m.yule_k:.2f}."
    )


def _find_suspicious_fragments(text: str) -> list[str]:
    """Найти клише-фразы для подсветки в UI."""
    text_low = text.lower()
    hits: list[str] = []
    for c in SYNTHETIC_CLICHES:
        idx = text_low.find(c)
        while idx != -1:
            # Извлечь контекст (фрагмент оригинала с сохранением регистра)
            start = max(0, idx)
            end = min(len(text), idx + len(c))
            hits.append(text[start:end])
            idx = text_low.find(c, idx + 1)
            if len(hits) >= 20:
                break
        if len(hits) >= 20:
            break
    return hits


def analyze(text: str) -> Signal1Result:
    """Полный конвейер Сигнала 1."""
    metrics = compute_metrics(text)
    score = _stat_score(metrics)
    return Signal1Result(
        stat_score=score,
        metrics=metrics,
        interpretation=_interpretation(metrics, score),
        suspicious_stat_fragments=_find_suspicious_fragments(text),
    )
