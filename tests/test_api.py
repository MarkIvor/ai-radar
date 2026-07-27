"""Тесты API AI Radar.

Используют TestClient FastAPI. LLM-вызовы полностью мокаются, чтобы
тесты были детерминированными и не требовали внешних API.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.services.signal2_judge import JudgeResult, DetectorVerdict
from app.services.storage import close_storage, reset_storage_for_tests


SAMPLE_TEXT = (
    "В современном мире развитие технологий играет важную роль. "
    "Таким образом, необходимо отметить, что прогресс оказывает существенное влияние. "
    "Важно отметить, что данная тенденция проявляется во многих сферах. "
    "Подводя итог, можно сделать вывод о том, что значимость велика."
)


SAMPLE_AI_VERDICT = DetectorVerdict(
    provider_name="mock-claude",
    model="anthropic/claude-sonnet-5",
    ai_score=82,
    veto_triggered=True,
    veto_reason="Синтетический синтаксис, повторы клише «таким образом», «важно отметить».",
    academic_integrity="НАРУШЕНА",
    suspicious_fragments=["Таким образом, необходимо отметить", "Важно отметить, что"],
)


@pytest.fixture
def client(fresh_db):
    app = create_app()
    with TestClient(app) as c:
        yield c


def _login(client: TestClient, login: str = "admin", password: str = "airadar") -> str:
    r = client.post("/api/auth/login", json={"login": login, "password": password})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def _auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# --------------------------------------------------------------------------- #
#  Health                                                                      #
# --------------------------------------------------------------------------- #


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# --------------------------------------------------------------------------- #
#  Auth                                                                        #
# --------------------------------------------------------------------------- #


def test_login_success(client):
    token = _login(client)
    assert token
    assert token.count(".") == 2  # JWT-формат


def test_login_wrong_password(client):
    r = client.post("/api/auth/login", json={"login": "admin", "password": "wrong"})
    assert r.status_code == 401


def test_protected_endpoint_requires_auth(client):
    r = client.get("/api/admin/settings")
    assert r.status_code == 401


def test_whoami_with_session(client):
    token = _login(client)
    r = client.get("/api/auth/whoami", headers=_auth_headers(token))
    assert r.status_code == 200
    assert r.json()["type"] == "session"


# --------------------------------------------------------------------------- #
#  Quick scan (json)                                                           #
# --------------------------------------------------------------------------- #


def test_quick_scan_json_with_mocked_judge(client):
    """Успешный quick scan: LLM-судья замокан, отчёт возвращает ai_score."""
    token = _login(client)

    async def fake_run_judge(text, signal1, storage):
        return JudgeResult(
            ai_score=SAMPLE_AI_VERDICT.ai_score,
            veto_triggered=SAMPLE_AI_VERDICT.veto_triggered,
            veto_reason=SAMPLE_AI_VERDICT.veto_reason,
            academic_integrity=SAMPLE_AI_VERDICT.academic_integrity,
            suspicious_fragments=SAMPLE_AI_VERDICT.suspicious_fragments,
            judge_status="FULL",
            ensemble_panel=[SAMPLE_AI_VERDICT],
            judge_verdict=SAMPLE_AI_VERDICT,
            raw_json={"ensemble": {"ai_score_median": 82}, "judge": {"ai_score": 82}},
        )

    with patch("app.services.pipeline.run_judge", new=AsyncMock(side_effect=fake_run_judge)):
        r = client.post(
            "/api/scan/quick/json",
            json={"text": SAMPLE_TEXT, "title": "test.txt"},
            headers=_auth_headers(token),
        )
    assert r.status_code == 200, r.text
    data = r.json()
    assert 0 <= data["ai_score"] <= 100
    assert data["veto"] is True
    assert data["academic_integrity"] == "НАРУШЕНА"
    assert data["judge_status"] == "FULL"
    assert len(data["suspicious_fragments"]) > 0
    assert "signal1" in data["metrics"]


def test_quick_scan_degraded_when_no_llm(client):
    """Без LLM-провайдеров — degraded mode, VETO не активируется."""
    token = _login(client)
    r = client.post(
        "/api/scan/quick/json",
        json={"text": SAMPLE_TEXT, "title": "test.txt"},
        headers=_auth_headers(token),
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["judge_status"] == "OFFLINE"
    assert data["veto"] is False
    assert data["ai_score"] == data["stat_score"]


def test_quick_scan_empty_text_rejected(client):
    token = _login(client)
    r = client.post(
        "/api/scan/quick/json",
        json={"text": "x"},
        headers=_auth_headers(token),
    )
    assert r.status_code in (400, 422)


# --------------------------------------------------------------------------- #
#  Quick scan (multipart form)                                                #
# --------------------------------------------------------------------------- #


def test_quick_scan_with_file_upload(client):
    token = _login(client)
    # Создадим текстовый файл в памяти
    files = {"file": ("test.txt", SAMPLE_TEXT.encode("utf-8"), "text/plain")}
    r = client.post(
        "/api/scan/quick",
        files=files,
        data={"title": "test.txt"},
        headers=_auth_headers(token),
    )
    assert r.status_code == 200, r.text
    assert r.json()["judge_status"] == "OFFLINE"


# --------------------------------------------------------------------------- #
#  Folders CRUD                                                                #
# --------------------------------------------------------------------------- #


def test_folder_crud(client):
    token = _login(client)
    # Создать
    r = client.post(
        "/api/folders",
        json={"name": "Диссертации 2026"},
        headers=_auth_headers(token),
    )
    assert r.status_code == 201
    folder = r.json()
    fid = folder["id"]
    # Список
    r = client.get("/api/folders", headers=_auth_headers(token))
    assert r.status_code == 200
    assert any(f["id"] == fid for f in r.json())
    # Подпапка
    r = client.post(
        "/api/folders",
        json={"name": "Бакалавриат ПИ", "parent_id": fid},
        headers=_auth_headers(token),
    )
    assert r.status_code == 201
    # Удалить
    r = client.delete(f"/api/folders/{fid}", headers=_auth_headers(token))
    assert r.status_code == 204


def test_file_upload_and_deep_scan(client):
    token = _login(client)
    # Создать папку
    r = client.post("/api/folders", json={"name": "Works"}, headers=_auth_headers(token))
    fid = r.json()["id"]
    # Загрузить файл
    files = {"file": ("work.txt", SAMPLE_TEXT.encode("utf-8"), "text/plain")}
    r = client.post(
        f"/api/folders/{fid}/files",
        files=files,
        headers=_auth_headers(token),
    )
    assert r.status_code == 201
    file_id = r.json()["id"]
    # Глубокая проверка
    r = client.post(f"/api/scan/deep/{file_id}", headers=_auth_headers(token))
    assert r.status_code == 200
    data = r.json()
    assert data["source"] == "deep"


# --------------------------------------------------------------------------- #
#  Admin settings                                                              #
# --------------------------------------------------------------------------- #


def test_admin_settings_get_and_update(client):
    token = _login(client)
    # GET
    r = client.get("/api/admin/settings", headers=_auth_headers(token))
    assert r.status_code == 200
    assert r.json()["veto_threshold"] == 75
    # PUT
    r = client.put(
        "/api/admin/settings",
        json={"veto_threshold": 80, "judge_mode": "single"},
        headers=_auth_headers(token),
    )
    assert r.status_code == 200
    assert r.json()["veto_threshold"] == 80
    assert r.json()["judge_mode"] == "single"
    # GET снова
    r = client.get("/api/admin/settings", headers=_auth_headers(token))
    assert r.json()["veto_threshold"] == 80


def test_admin_password_change(client):
    token = _login(client)
    # Смена пароля через me/password
    r = client.put(
        "/api/admin/users/me/password",
        json={"new_password": "newpass123"},
        headers=_auth_headers(token),
    )
    assert r.status_code == 204
    # Старый пароль больше не работает
    r = client.post("/api/auth/login", json={"login": "admin", "password": "airadar"})
    assert r.status_code == 401
    # Новый работает
    r = client.post("/api/auth/login", json={"login": "admin", "password": "newpass123"})
    assert r.status_code == 200


# --------------------------------------------------------------------------- #
#  LLM providers CRUD                                                          #
# --------------------------------------------------------------------------- #


def test_llm_provider_crud(client):
    token = _login(client)
    payload = {
        "name": "test-claude",
        "base_url": "https://api.example.com/v1",
        "model": "anthropic/claude-sonnet-5",
        "api_key": "sk-test-123456",
        "role": "ensemble",
        "enabled": True,
    }
    r = client.post("/api/admin/llm-providers", json=payload, headers=_auth_headers(token))
    assert r.status_code == 201
    p = r.json()
    assert p["api_key_masked"] != "sk-test-123456"  # ключ замаскирован
    assert p["role"] == "ensemble"
    pid = p["id"]

    # List
    r = client.get("/api/admin/llm-providers", headers=_auth_headers(token))
    assert any(x["id"] == pid for x in r.json())

    # Delete
    r = client.delete(f"/api/admin/llm-providers/{pid}", headers=_auth_headers(token))
    assert r.status_code == 204


# --------------------------------------------------------------------------- #
#  API keys                                                                    #
# --------------------------------------------------------------------------- #


def test_api_key_create_verify_revoke(client):
    token = _login(client)
    # Create
    r = client.post("/api/admin/api-keys", json={"name": "ci"}, headers=_auth_headers(token))
    assert r.status_code == 201
    data = r.json()
    key = data["key"]
    assert key.startswith("air_")
    kid = data["id"]

    # Использовать ключ для доступа к защищённому эндпоинту
    r = client.get("/api/auth/whoami", headers={"Authorization": f"Bearer {key}"})
    assert r.status_code == 200
    assert r.json()["type"] == "api_key"
    assert r.json()["name"] == "ci"

    # X-API-Key header тоже работает
    r = client.get("/api/auth/whoami", headers={"X-API-Key": key})
    assert r.status_code == 200

    # Revoke
    r = client.delete(f"/api/admin/api-keys/{kid}", headers=_auth_headers(token))
    assert r.status_code == 204

    # Отозванный ключ не работает
    r = client.get("/api/auth/whoami", headers={"Authorization": f"Bearer {key}"})
    assert r.status_code == 401


def test_api_key_rotate(client):
    token = _login(client)
    r = client.post("/api/admin/api-keys", json={"name": "ci2"}, headers=_auth_headers(token))
    kid = r.json()["id"]
    old_key = r.json()["key"]

    r = client.post(f"/api/admin/api-keys/{kid}/rotate", headers=_auth_headers(token))
    assert r.status_code == 200
    new_key = r.json()["key"]
    assert new_key != old_key

    # Старый отозван
    r = client.get("/api/auth/whoami", headers={"Authorization": f"Bearer {old_key}"})
    assert r.status_code == 401
    # Новый работает
    r = client.get("/api/auth/whoami", headers={"Authorization": f"Bearer {new_key}"})
    assert r.status_code == 200


# --------------------------------------------------------------------------- #
#  Recent reports                                                              #
# --------------------------------------------------------------------------- #


def test_recent_reports(client):
    token = _login(client)
    # Создаём пару отчётов
    for i in range(3):
        r = client.post(
            "/api/scan/quick/json",
            json={"text": SAMPLE_TEXT + f" проверка номер {i}", "title": f"t{i}.txt"},
            headers=_auth_headers(token),
        )
        assert r.status_code == 200
    r = client.get("/api/scan/reports/recent?limit=10", headers=_auth_headers(token))
    assert r.status_code == 200
    assert len(r.json()) >= 3


# --------------------------------------------------------------------------- #
#  Judge status                                                                #
# --------------------------------------------------------------------------- #


def test_judge_status_offline_when_no_providers(client):
    token = _login(client)
    r = client.get("/api/admin/judge-status", headers=_auth_headers(token))
    assert r.status_code == 200
    assert r.json()["status"] == "OFFLINE"


def test_judge_status_online_with_provider(client):
    token = _login(client)
    r = client.post(
        "/api/admin/llm-providers",
        json={
            "name": "test",
            "base_url": "https://api.example.com/v1",
            "model": "test/model",
            "api_key": "sk-xxx",
            "role": "ensemble",
            "enabled": True,
        },
        headers=_auth_headers(token),
    )
    assert r.status_code == 201
    r = client.get("/api/admin/judge-status", headers=_auth_headers(token))
    assert r.status_code == 200
    # С одним ансамблевым провайдером и без судьи — ENSEMBLE_ONLY
    assert r.json()["status"] == "ENSEMBLE_ONLY"


# --------------------------------------------------------------------------- #
#  Refresh token                                                               #
# --------------------------------------------------------------------------- #


def test_refresh_token_flow(client):
    r = client.post("/api/auth/login", json={"login": "admin", "password": "airadar"})
    tokens = r.json()
    refresh = tokens["refresh_token"]
    r = client.post("/api/auth/refresh", json={"refresh_token": refresh})
    assert r.status_code == 200
    assert r.json()["access_token"]


# --------------------------------------------------------------------------- #
#  Roles & users management                                                    #
# --------------------------------------------------------------------------- #


def test_create_teacher_and_login(client):
    token = _login(client)
    r = client.post(
        "/api/admin/users",
        json={"login": "teacher1", "password": "teach123", "role": "teacher",
              "display_name": "Иван Иванов"},
        headers=_auth_headers(token),
    )
    assert r.status_code == 201
    # Логин как teacher
    r = client.post("/api/auth/login", json={"login": "teacher1", "password": "teach123"})
    assert r.status_code == 200
    assert r.json()["role"] == "teacher"


def test_teacher_cannot_see_admin_settings(client):
    token = _login(client)
    client.post(
        "/api/admin/users",
        json={"login": "teacher2", "password": "teach123", "role": "teacher"},
        headers=_auth_headers(token),
    )
    r = client.post("/api/auth/login", json={"login": "teacher2", "password": "teach123"})
    teacher_token = r.json()["access_token"]
    # teacher не видит настройки системы
    r = client.get("/api/admin/settings", headers=_auth_headers(teacher_token))
    assert r.status_code == 403
    r = client.get("/api/admin/llm-providers", headers=_auth_headers(teacher_token))
    assert r.status_code == 403
    r = client.get("/api/admin/api-keys", headers=_auth_headers(teacher_token))
    assert r.status_code == 403
    # но может сменить свой пароль
    r = client.put(
        "/api/admin/users/me/password",
        json={"new_password": "newteach123"},
        headers=_auth_headers(teacher_token),
    )
    assert r.status_code == 204


def test_teacher_sees_only_own_folders(client):
    admin_token = _login(client)
    # Создать teacher
    client.post(
        "/api/admin/users",
        json={"login": "teacher3", "password": "teach123", "role": "teacher"},
        headers=_auth_headers(admin_token),
    )
    r = client.post("/api/auth/login", json={"login": "teacher3", "password": "teach123"})
    teacher_token = r.json()["access_token"]
    # Admin создаёт shared папку
    r = client.post("/api/folders", json={"name": "Shared"},
                    headers=_auth_headers(admin_token))
    # Teacher создаёт свою папку
    r = client.post("/api/folders", json={"name": "My Folder"},
                    headers=_auth_headers(teacher_token))
    assert r.status_code == 201
    my_id = r.json()["id"]
    # Teacher видит обе (свою + shared)
    r = client.get("/api/folders", headers=_auth_headers(teacher_token))
    names = [f["name"] for f in r.json()]
    assert "My Folder" in names
    assert "Shared" in names


def test_admin_can_reset_teacher_password(client):
    admin_token = _login(client)
    client.post(
        "/api/admin/users",
        json={"login": "teacher4", "password": "teach123", "role": "teacher"},
        headers=_auth_headers(admin_token),
    )
    users = client.get("/api/admin/users", headers=_auth_headers(admin_token)).json()
    uid = next(u["id"] for u in users if u["login"] == "teacher4")
    r = client.post(
        f"/api/admin/users/{uid}/reset-password",
        json={"new_password": "reset123"},
        headers=_auth_headers(admin_token),
    )
    assert r.status_code == 204
    # Старый пароль не работает
    r = client.post("/api/auth/login", json={"login": "teacher4", "password": "teach123"})
    assert r.status_code == 401
    # Новый работает
    r = client.post("/api/auth/login", json={"login": "teacher4", "password": "reset123"})
    assert r.status_code == 200
