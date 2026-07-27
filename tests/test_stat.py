"""Тесты Сигнала 1 (статистический движок)."""

from __future__ import annotations

from app.services.signal1_stat import (
    Signal1Metrics,
    Signal1Result,
    analyze,
    compute_metrics,
)


HUMAN_TEXT = """
Сегодня утром я вышел в магазин за хлебом. Дорога заняла минут двадцать, не меньше.
По пути встретил соседку, она рассказывала про свою кошку, которая опять убежала
на улицу. Я кивнул и улыбнулся — что ещё оставалось делать? Хлеб в магазине,
к слову, подорожал на пять рублей. Обидно, но что поделаешь.
Дома меня ждал недовольный кот. Он орал так, будто я не ел его неделю, хотя
кормил его всего час назад. Пришлось дать добавки. Потом заварил чай и сел
у окна, глядя на прохожих. Жизнь идёт своим чередом, и иногда это даже хорошо.
"""


AI_TEXT = """
В современном мире развитие технологий играет важную роль в жизни общества.
Таким образом, необходимо отметить, что прогресс оказывает существенное
влияние на различные сферы деятельности человека. Важно отметить, что данная
тенденция проявляется в таких областях, как образование, медицина и промышленность.
Следует подчеркнуть, что внедрение инноваций представляет собой ключевой фактор
развития экономики. Подводя итог, можно сделать вывод о том, что технологический
прогресс имеет огромное значение для современного общества. Таким образом,
важность данных преобразований не вызывает сомнений. Кроме того, следует отметить,
что развитие технологий продолжается непрерывно. В заключение важно подчеркнуть,
что данная проблема требует комплексного подхода. Таким образом, подводя итог,
можно утверждать, что роль технологий будет только возрастать.
"""


def test_compute_metrics_returns_all_fields():
    m = compute_metrics(HUMAN_TEXT)
    assert isinstance(m, Signal1Metrics)
    assert m.num_words > 0
    assert m.num_sentences > 0
    assert m.num_chars > 0
    # Все 20 метрик должны быть числами
    for f in (
        m.burstiness,
        m.lexical_diversity_ttr,
        m.sentence_length_variance,
        m.perplexity_proxy,
        m.synthetic_cliche_density,
        m.punctuation_entropy,
        m.sentence_starter_entropy,
        m.yule_k,
        m.honore_r,
        m.sichel_s,
        m.hapax_ratio,
        m.connective_density,
        m.ngram_repetition_rate,
        m.sentence_rhythm_entropy,
        m.comma_per_sentence,
        m.avg_syllables_per_word,
        m.function_word_ratio,
        m.lexical_density,
        m.ttr_windowed,
        m.determiner_density,
    ):
        assert isinstance(f, (int, float))


def test_human_text_has_higher_burstiness_than_ai():
    human = compute_metrics(HUMAN_TEXT)
    ai = compute_metrics(AI_TEXT)
    assert human.burstiness > ai.burstiness, (
        f"human.burstiness={human.burstiness} должен быть > ai.burstiness={ai.burstiness}"
    )


def test_ai_text_has_higher_cliche_density():
    human = compute_metrics(HUMAN_TEXT)
    ai = compute_metrics(AI_TEXT)
    assert ai.synthetic_cliche_density > human.synthetic_cliche_density


def test_ai_text_score_higher_than_human():
    ai_result: Signal1Result = analyze(AI_TEXT)
    human_result: Signal1Result = analyze(HUMAN_TEXT)
    assert 0 <= ai_result.stat_score <= 100
    assert 0 <= human_result.stat_score <= 100
    assert ai_result.stat_score > human_result.stat_score, (
        f"AI stat_score={ai_result.stat_score} должен быть > human stat_score={human_result.stat_score}"
    )


def test_analyze_returns_interpretation_and_fragments():
    result = analyze(AI_TEXT)
    assert isinstance(result, Signal1Result)
    assert "ai" in result.interpretation.lower() or "вероятность" in result.interpretation.lower()
    # AI-текст должен содержать клише
    assert len(result.suspicious_stat_fragments) > 0


def test_empty_text_does_not_crash():
    result = analyze("")
    assert result.stat_score >= 0
    assert result.metrics.num_words == 0


def test_short_text_does_not_crash():
    result = analyze("Мяу.")
    assert result.metrics.num_words >= 1


def test_ttr_in_valid_range():
    for text in (HUMAN_TEXT, AI_TEXT):
        m = compute_metrics(text)
        assert 0.0 <= m.lexical_diversity_ttr <= 1.0


def test_ngram_repetition_in_valid_range():
    for text in (HUMAN_TEXT, AI_TEXT):
        m = compute_metrics(text)
        assert 0.0 <= m.ngram_repetition_rate <= 1.0
