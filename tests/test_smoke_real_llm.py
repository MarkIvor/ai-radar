"""Дымовой (smoke) тест с реальным LLM API через ансамбль.

НЕ входит в pytest-сьют по умолчанию (вызывается вручную):
    set RUN_REAL_LLM_SMOKE=1
    set SMOKE_LLM_BASE_URL=https://your-llm-endpoint/v1
    set SMOKE_LLM_MODEL=openai/gpt-5.1
    set SMOKE_LLM_API_KEY=sk-...
    python -m pytest tests/test_smoke_real_llm.py -s

Используется для проверки корректности интеграции с реальным API.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_REAL_LLM_SMOKE"),
    reason="Smoke-тест реального LLM запускается только с RUN_REAL_LLM_SMOKE=1",
)


SAMPLE_TEXT = (
    "В современном мире развитие технологий играет важную роль в жизни общества. "
    "Таким образом, необходимо отметить, что прогресс оказывает существенное влияние "
    "на различные сферы деятельности человека. Важно отметить, что данная тенденция "
    "проявляется в образовании, медицине и промышленности. Подводя итог, можно сделать "
    "вывод о том, что технологический прогресс имеет огромное значение."
)


def test_real_llm_quick_scan(fresh_db):
    base_url = os.environ.get("SMOKE_LLM_BASE_URL")
    model = os.environ.get("SMOKE_LLM_MODEL", "openai/gpt-5.1")
    api_key = os.environ.get("SMOKE_LLM_API_KEY")
    assert base_url and api_key, "Задай SMOKE_LLM_BASE_URL и SMOKE_LLM_API_KEY"

    app = create_app()
    with TestClient(app) as client:
        # Логин
        r = client.post("/api/auth/login", json={"password": "airadar"})
        assert r.status_code == 200
        token = r.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Добавить провайдер (ансамбль-детектор)
        r = client.post(
            "/api/admin/llm-providers",
            json={
                "name": "smoke",
                "base_url": base_url,
                "model": model,
                "api_key": api_key,
                "role": "ensemble",
                "enabled": True,
            },
            headers=headers,
        )
        assert r.status_code == 201, r.text

        # Проверить, что провайдер активен (ENSEMBLE_ONLY — судья не назначен)
        r = client.get("/api/admin/judge-status", headers=headers)
        assert r.json()["status"] in ("ENSEMBLE_ONLY", "FULL"), r.text

        # Запустить реальный quick scan
        r = client.post(
            "/api/scan/quick/json",
            json={"text": SAMPLE_TEXT, "title": "smoke.txt"},
            headers=headers,
        )
        assert r.status_code == 200, r.text
        data = r.json()
        assert 0 <= data["ai_score"] <= 100
        assert data["judge_status"] in ("ENSEMBLE_ONLY", "FULL")
        print(
            f"\n[SMOKE] model={model} ai_score={data['ai_score']} "
            f"veto={data['veto']} integrity={data['academic_integrity']} "
            f"status={data['judge_status']}"
        )
