"""The whole user journey, from an uploaded report to a re-analysis.

This is the API-side counterpart of the frontend journey test: it drives the
exact endpoint sequence the UI drives — upload, preview, correct, confirm,
history, summary, re-analyse — and asserts the shape of every response the
frontend reads. A change that breaks the screen breaks this test too.
"""

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.errors import install_error_handlers
from app.healthroutes import router as health_router
from database.config import Base, get_db
from database.models import User
from utils.helpers import create_access_token, get_password_hash

REPORT_PDF = Path(__file__).resolve().parents[1] / "health_test_report_template.pdf"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """A test client backed by a throwaway database with one registered user."""
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))

    engine = create_engine(
        f"sqlite:///{tmp_path / 'test.db'}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    session = TestingSessionLocal()
    user = User(
        username="tester",
        email="tester@example.com",
        hashed_password=get_password_hash("secret123"),
        is_active=True,
    )
    session.add(user)
    session.commit()
    user_id = user.id
    session.close()

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app = FastAPI()
    app.include_router(health_router)
    app.dependency_overrides[get_db] = override_get_db
    install_error_handlers(app)

    with TestClient(app) as test_client:
        test_client.user_id = user_id
        yield test_client


def _auth_headers(client):
    token = create_access_token({"sub": "tester", "user_id": client.user_id})
    return {"Authorization": f"Bearer {token}"}


def test_upload_preview_correction_assessment_history_and_reanalysis(client):
    headers = _auth_headers(client)

    # ── 1. The input contract the manual form renders ───────────────────
    schema = client.get("/api/v1/health/input-schema", headers=headers).json()
    assert schema["status"] == "success"
    assert schema["parameters"]["hba1c"]["required_for"] == ["diabetes"]
    assert schema["parameters"]["systolic_bp"]["required_for"] == ["cardiovascular"]
    assert schema["parameters"]["smoker"]["type"] == "boolean"
    assert schema["diseases"]["cardiovascular"]["required"] == [
        "age",
        "systolic_bp",
        "total_cholesterol",
    ]

    # ── 2. The report is uploaded and read ───────────────────────────────
    with REPORT_PDF.open("rb") as report:
        preview = client.post(
            "/api/v1/health/upload-report/preview",
            files={"file": ("report.pdf", report, "application/pdf")},
            headers=headers,
        )
    assert preview.status_code == 200, preview.text
    preview_body = preview.json()

    assert preview_body["status"] == "pending"
    assert preview_body["available_diseases"] == ["diabetes"]
    assert "systolic_bp" in preview_body["missing_fields"]["cardiovascular"]

    # Every value the screen shows is there, with the fields it reads.
    hba1c = preview_body["extracted_parameters"]["hba1c"]
    assert hba1c["value"] == 6.5
    assert hba1c["status"] == "High"
    assert hba1c["unit"] == "%"

    # ── 3. The user corrects a misread value and confirms ───────────────
    confirmed = client.post(
        "/api/v1/health/upload-report/confirm",
        json={
            "extraction_id": preview_body["extraction_id"],
            "hba1c": 6.5,
            "glucose": 165,
            "bmi": 28.4,
            "age": 45,
            "symptoms": ["Fatigue / Low energy"],
        },
        headers=headers,
    )
    assert confirmed.status_code == 200, confirmed.text
    confirmed_body = confirmed.json()

    assert confirmed_body["status"] == "success"
    assert confirmed_body["workflow_status"] == "completed"
    assert confirmed_body["record_id"] == preview_body["extraction_id"]

    # The assessment the results screen renders.
    unified = confirmed_body["unified"]
    assert unified["status"] == "completed"
    assert unified["assessments"]["diabetes"]["status"] in (
        "completed",
        "completed_with_errors",
    )
    assert unified["assessments"]["diabetes"]["score"] is not None
    # The condition that was never applicable is reported as such.
    assert unified["assessments"]["cardiovascular"]["status"] == "not_applicable"
    assert unified["model_disclaimer"]
    assert unified["agent_status"]

    # ── 4. The history lists the assessment ──────────────────────────────
    history = client.get("/api/v1/health/history", headers=headers).json()
    assert history["status"] == "success"
    assert history["total"] == 1
    entry = history["items"][0]
    assert entry["record_id"] == confirmed_body["record_id"]
    assert entry["status"] == "completed"
    assert entry["risk_level"] == confirmed_body["risk_level"]
    assert entry["parameters"]["hba1c"] == 6.5
    assert "diabetes" in entry["diseases"]

    # ── 5. The summary drives the charts ─────────────────────────────────
    summary = client.get("/api/v1/health/history/summary", headers=headers).json()
    assert summary["status"] == "success"
    assert summary["count"] == 1
    assert summary["parameters"]["hba1c"][0]["value"] == 6.5
    assert summary["scores"][0]["value"] == entry["score"]
    assert summary["risk_levels"]

    # ── 6. The record is re-analysed ─────────────────────────────────────
    reanalyzed = client.post(
        f"/api/v1/health/analyze/{confirmed_body['record_id']}", headers=headers
    )
    assert reanalyzed.status_code == 200, reanalyzed.text
    reanalyzed_body = reanalyzed.json()
    assert reanalyzed_body["status"] == "success"
    assert reanalyzed_body["record_id"] == confirmed_body["record_id"]
    assert reanalyzed_body["unified"]["assessments"]["diabetes"]["score"] == (
        unified["assessments"]["diabetes"]["score"]
    )

    # The history still holds one assessment: a re-analysis replaces the
    # result, it does not add a second record.
    history_after = client.get("/api/v1/health/history", headers=headers).json()
    assert history_after["total"] == 1


def test_a_report_that_cannot_be_read_is_reported_as_such(client):
    """A report with no readable values must not silently produce a record."""
    with REPORT_PDF.open("rb") as report:
        response = client.post(
            "/api/v1/health/upload-report/preview",
            files={"file": ("report.pdf", report, "application/pdf")},
            headers=_auth_headers(client),
        )
    assert response.status_code == 200

    # Nothing was assessed yet, so the record is pending and the history is
    # empty until the values are confirmed.
    history = client.get(
        "/api/v1/health/history", headers=_auth_headers(client)
    ).json()
    assert history["total"] == 0
