"""API contract, persistence and history tests.

Covers the input contract exposed to the frontend, the normalized response
schema, the persistence of extracted parameters and assessments, the
consistent parsing of historical results, the assessment history endpoints
and the report upload → preview → confirm flow.
"""

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.errors import install_error_handlers
from app.healthroutes import router as health_router
from database.config import Base, get_db
from database.models import HealthRecord, User
from utils.helpers import create_access_token, get_password_hash

REPORT_PDF = Path(__file__).resolve().parents[1] / "health_test_report_template.pdf"

DIABETES_PAYLOAD = {
    "hba1c": 6.8,
    "glucose": 148,
    "bmi": 29,
    "age": 45,
    "symptoms": ["Fatigue / Low energy"],
}

CARDIO_PAYLOAD = {
    "age": 55,
    "systolic_bp": 165,
    "diastolic_bp": 95,
    "total_cholesterol": 250,
    "ldl": 140,
    "hdl": 38,
    "smoker": True,
}


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
        test_client.session_factory = TestingSessionLocal
        yield test_client


def _auth_headers(client):
    token = create_access_token({"sub": "tester", "user_id": client.user_id})
    return {"Authorization": f"Bearer {token}"}


def _submit(client, payload):
    response = client.post(
        "/api/v1/health/health-data", json=payload, headers=_auth_headers(client)
    )
    assert response.status_code == 200, response.text
    return response.json()


# ═════════════════════════════════════════════════════════════════════════════
#  INPUT CONTRACT
# ═════════════════════════════════════════════════════════════════════════════

def test_input_schema_exposes_every_supported_parameter(client):
    response = client.get("/api/v1/health/input-schema", headers=_auth_headers(client))
    assert response.status_code == 200, response.text
    body = response.json()

    parameters = body["parameters"]
    for name in (
        "hba1c", "glucose", "bmi", "age", "systolic_bp", "diastolic_bp",
        "total_cholesterol", "ldl", "hdl", "smoker",
    ):
        assert name in parameters, f"{name} is missing from the input contract"

    assert parameters["hba1c"]["unit"] == "%"
    assert parameters["systolic_bp"]["min"] == 50.0
    assert parameters["systolic_bp"]["max"] == 260.0
    assert parameters["smoker"]["type"] == "boolean"


def test_input_schema_reports_the_disease_requirements(client):
    body = client.get(
        "/api/v1/health/input-schema", headers=_auth_headers(client)
    ).json()

    diseases = body["diseases"]
    assert diseases["diabetes"]["required"] == ["hba1c", "glucose", "bmi", "age"]
    # Smoking status is optional: an absent value means "not supplied", which
    # is not the same as "does not smoke".
    assert diseases["cardiovascular"]["required"] == [
        "age", "systolic_bp", "total_cholesterol",
    ]
    assert "smoker" in diseases["cardiovascular"]["optional"]
    # A parameter says which conditions need it, so the client knows what to ask for.
    assert body["parameters"]["systolic_bp"]["required_for"] == ["cardiovascular"]
    assert body["parameters"]["hba1c"]["required_for"] == ["diabetes"]
    assert body["parameters"]["smoker"]["required_for"] == []


def test_input_schema_lists_the_symptom_catalogue(client):
    body = client.get(
        "/api/v1/health/input-schema", headers=_auth_headers(client)
    ).json()
    assert "Fatigue / Low energy" in body["symptoms"]


# ═════════════════════════════════════════════════════════════════════════════
#  NORMALIZED RESPONSE
# ═════════════════════════════════════════════════════════════════════════════

def test_submission_returns_the_normalized_unified_response(client):
    body = _submit(client, DIABETES_PAYLOAD)

    unified = body["unified"]
    assert unified["status"] == "completed"
    assert unified["model_disclaimer"]
    assert set(unified["assessments"]) == {"diabetes", "cardiovascular"}
    assert unified["assessments"]["diabetes"]["status"] == "completed"
    assert unified["assessments"]["cardiovascular"]["status"] == "not_applicable"
    assert unified["assessments"]["diabetes"]["risk_level"]
    assert unified["assessments"]["diabetes"]["score"] is not None
    assert unified["agent_status"]["analysis"]["status"] == "completed"
    assert unified["differential"]["status"]
    assert unified["trend"]["status"]


def test_submission_returns_the_assessment_snapshot(client):
    body = _submit(client, DIABETES_PAYLOAD)

    assessment = body["assessment"]
    assert assessment["status"] == "completed"
    assert assessment["risk_level"]
    assert assessment["score"] is not None
    assert assessment["disease"] == "diabetes"
    assert "diabetes" in assessment["diseases"]
    assert assessment["diseases"]["diabetes"]["score"] is not None


def test_cardiovascular_submission_is_assessed(client):
    body = _submit(client, CARDIO_PAYLOAD)

    unified = body["unified"]
    assert unified["assessments"]["cardiovascular"]["status"] == "completed"
    assert unified["assessments"]["diabetes"]["status"] == "not_applicable"
    assert body["assessment"]["disease"] == "cardiovascular"


# ═════════════════════════════════════════════════════════════════════════════
#  PERSISTENCE
# ═════════════════════════════════════════════════════════════════════════════

def test_submission_persists_the_parameters_and_the_assessment(client):
    body = _submit(client, DIABETES_PAYLOAD)
    record_id = body["record_id"]

    session = client.session_factory()
    record = session.query(HealthRecord).filter(HealthRecord.id == record_id).first()
    assert record is not None
    assert record.source == "manual"
    assert record.analyzed is True

    # The captured parameters are stored, with their clinical status.
    captured = json.loads(record.extracted_parameters)
    assert captured["hba1c"]["value"] == 6.8
    assert captured["hba1c"]["status"] == "High"
    assert captured["hba1c"]["source"] == "submitted"

    # The assessment is stored as a normalized snapshot.
    snapshot = json.loads(record.assessment)
    assert snapshot["risk_level"] == body["final_assessment"]["risk_level"]
    assert snapshot["score"] == body["final_assessment"]["score"]
    session.close()


def test_stored_assessment_is_returned_with_the_analysis(client):
    body = _submit(client, DIABETES_PAYLOAD)

    response = client.get(
        f"/api/v1/health/health-data/{body['record_id']}/analysis",
        headers=_auth_headers(client),
    )
    assert response.status_code == 200, response.text
    payload = response.json()

    assert payload["source"] == "manual"
    assert payload["analysis"]["assessment"]["risk_level"]
    assert payload["analysis"]["extracted_parameters"]["hba1c"]["value"] == 6.8
    assert payload["analysis"]["unified"]["assessments"]["diabetes"]["status"] == "completed"


# ═════════════════════════════════════════════════════════════════════════════
#  ASSESSMENT HISTORY
# ═════════════════════════════════════════════════════════════════════════════

def test_history_lists_the_assessments_newest_first(client):
    first = _submit(client, DIABETES_PAYLOAD)
    second = _submit(client, CARDIO_PAYLOAD)

    response = client.get("/api/v1/health/history", headers=_auth_headers(client))
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["status"] == "success"
    assert body["count"] == 2
    assert body["total"] == 2

    record_ids = [item["record_id"] for item in body["items"]]
    assert record_ids[0] == second["record_id"]
    assert record_ids[1] == first["record_id"]

    newest = body["items"][0]
    assert newest["risk_level"]
    assert newest["score"] is not None
    assert newest["disease"] == "cardiovascular"
    assert newest["status"] == "completed"
    assert newest["source"] == "manual"
    assert newest["parameters"]["systolic_bp"] == 165


def test_history_paginates(client):
    for _ in range(3):
        _submit(client, DIABETES_PAYLOAD)

    body = client.get(
        "/api/v1/health/history", params={"skip": 1, "limit": 1}, headers=_auth_headers(client)
    ).json()
    assert body["count"] == 1
    assert body["total"] == 3
    assert body["skip"] == 1
    assert body["limit"] == 1


def test_history_summary_returns_the_parameter_and_score_series(client):
    _submit(client, DIABETES_PAYLOAD)
    _submit(client, DIABETES_PAYLOAD)

    response = client.get(
        "/api/v1/health/history/summary", headers=_auth_headers(client)
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["count"] == 2
    assert len(body["parameters"]["hba1c"]) == 2
    assert body["parameters"]["hba1c"][0]["value"] == 6.8
    assert len(body["scores"]) == 2
    assert body["scores"][0]["risk_level"]
    assert sum(body["risk_levels"].values()) == 2


def test_history_is_scoped_to_the_user(client):
    _submit(client, DIABETES_PAYLOAD)

    # A second user must not see the first user's assessments.
    other = client.session_factory()
    intruder = User(
        username="intruder",
        email="intruder@example.com",
        hashed_password=get_password_hash("secret123"),
        is_active=True,
    )
    other.add(intruder)
    other.commit()
    intruder_id = intruder.id
    other.close()

    token = create_access_token({"sub": "intruder", "user_id": intruder_id})
    response = client.get(
        "/api/v1/health/history", headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert response.json()["total"] == 0


def test_legacy_history_entries_are_normalized(client):
    """A record written by an older build still answers the same fields."""
    session = client.session_factory()
    record = HealthRecord(
        user_id=client.user_id,
        hba1c=5.4,
        age=40,
        analyzed=True,
        source="manual",
        # A legacy analysis result: no final_assessment, no workflow_status.
        analysis_result=json.dumps({
            "risk_level": "Moderate",
            "risk_probability": 0.42,
            "abnormal_parameters": ["hba1c"],
            "abnormal_count": 1,
        }),
    )
    session.add(record)
    session.commit()
    record_id = record.id
    session.close()

    body = client.get(
        "/api/v1/health/history", headers=_auth_headers(client)
    ).json()
    item = body["items"][0]

    assert item["record_id"] == record_id
    assert item["risk_level"] == "Moderate"
    assert item["status"] == "completed"
    assert item["parameters"]["hba1c"] == 5.4


def test_unparsable_history_entry_does_not_break_the_endpoint(client):
    session = client.session_factory()
    record = HealthRecord(
        user_id=client.user_id,
        hba1c=5.4,
        age=40,
        analyzed=True,
        analysis_result="{not valid json",
    )
    session.add(record)
    session.commit()
    record_id = record.id
    session.close()

    response = client.get("/api/v1/health/history", headers=_auth_headers(client))
    assert response.status_code == 200
    item = response.json()["items"][0]
    assert item["record_id"] == record_id
    assert item["status"] == "unreadable"


# ═════════════════════════════════════════════════════════════════════════════
#  VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

def test_missing_parameters_are_rejected_with_field_details(client):
    response = client.post(
        "/api/v1/health/health-data",
        json={"age": 30},
        headers=_auth_headers(client),
    )
    assert response.status_code == 422
    body = response.json()

    assert body["status"] == "error"
    assert body["error"] == "validation_error"
    assert body["details"]

    message = " ".join(detail["message"] for detail in body["details"])
    assert "hba1c" in message
    assert "systolic_bp" in message


def test_the_input_schema_exposes_the_category_codes(client):
    body = client.get(
        "/api/v1/health/input-schema", headers=_auth_headers(client)
    ).json()

    for name in ("chest_pain_type", "st_slope"):
        assert name in body["parameters"], f"{name} is missing from the input contract"
        assert body["parameters"][name]["min"] == 0
        assert body["parameters"][name]["max"] == 3
        assert body["parameters"][name]["unit"] == "category"
    # They are optional, so no condition requires them.
    assert body["parameters"]["chest_pain_type"]["required_for"] == []
    assert "chest_pain_type" in body["diseases"]["cardiovascular"]["optional"]


def test_the_category_codes_are_stored_and_used(client):
    """A submission with both categories is scored by the extended model."""
    response = client.post(
        "/api/v1/health/health-data",
        json={**CARDIO_PAYLOAD, "chest_pain_type": 2, "st_slope": 3},
        headers=_auth_headers(client),
    )
    assert response.status_code == 200, response.text
    body = response.json()

    record_id = body["record_id"]
    session = client.session_factory()
    record = session.get(HealthRecord, record_id)
    assert record.chest_pain_type == 2
    assert record.st_slope == 3
    session.close()

    assessment = body.get("assessment") or {}
    assert assessment.get("status") == "completed"


def test_a_category_code_outside_the_range_is_rejected(client):
    response = client.post(
        "/api/v1/health/health-data",
        json={**CARDIO_PAYLOAD, "chest_pain_type": 9},
        headers=_auth_headers(client),
    )
    assert response.status_code == 422
    body = response.json()
    assert any(detail["field"] == "chest_pain_type" for detail in body["details"])


def test_out_of_range_values_are_rejected(client):
    response = client.post(
        "/api/v1/health/health-data",
        json={**DIABETES_PAYLOAD, "glucose": 5000},
        headers=_auth_headers(client),
    )
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "validation_error"
    assert any(detail["field"] == "glucose" for detail in body["details"])


def test_invalid_values_are_rejected(client):
    response = client.post(
        "/api/v1/health/health-data",
        json={**DIABETES_PAYLOAD, "hba1c": "not-a-number"},
        headers=_auth_headers(client),
    )
    assert response.status_code == 422
    body = response.json()
    assert any(detail["field"] == "hba1c" for detail in body["details"])


def test_inconsistent_blood_pressure_is_rejected(client):
    response = client.post(
        "/api/v1/health/health-data",
        json={**CARDIO_PAYLOAD, "systolic_bp": 120, "diastolic_bp": 130},
        headers=_auth_headers(client),
    )
    assert response.status_code == 422
    body = response.json()
    message = " ".join(detail["message"] for detail in body["details"])
    assert "diastolic_bp" in message


def test_impossible_lipid_values_are_rejected(client):
    response = client.post(
        "/api/v1/health/health-data",
        json={**CARDIO_PAYLOAD, "total_cholesterol": 100, "ldl": 180},
        headers=_auth_headers(client),
    )
    assert response.status_code == 422
    message = " ".join(detail["message"] for detail in response.json()["details"])
    assert "ldl" in message


def test_a_partial_cardiovascular_submission_is_accepted(client):
    """Only the required cardiovascular parameters are needed."""
    body = _submit(client, {
        "age": 55,
        "systolic_bp": 165,
        "total_cholesterol": 250,
        "smoker": True,
    })
    assert body["unified"]["assessments"]["cardiovascular"]["status"] == "completed"


# ═════════════════════════════════════════════════════════════════════════════
#  REPORT UPLOAD, PREVIEW AND CORRECTED SUBMISSION
# ═════════════════════════════════════════════════════════════════════════════

def test_report_preview_returns_the_extracted_values(client):
    with REPORT_PDF.open("rb") as report:
        response = client.post(
            "/api/v1/health/upload-report/preview",
            files={"file": ("report.pdf", report, "application/pdf")},
            headers=_auth_headers(client),
        )
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["status"] == "pending"
    assert body["extraction_id"] > 0
    assert body["extracted_parameters"]["hba1c"]["value"] == 6.5
    assert body["extracted_parameters"]["hba1c"]["status"] == "High"
    assert body["extracted_parameters"]["bmi"]["value"] == 28.4
    assert body["available_diseases"] == ["diabetes"]
    # The cardiovascular condition is still missing its parameters.
    assert "systolic_bp" in body["missing_fields"]["cardiovascular"]


def test_report_preview_creates_a_pending_record(client):
    with REPORT_PDF.open("rb") as report:
        body = client.post(
            "/api/v1/health/upload-report/preview",
            files={"file": ("report.pdf", report, "application/pdf")},
            headers=_auth_headers(client),
        ).json()

    session = client.session_factory()
    record = session.query(HealthRecord).filter(
        HealthRecord.id == body["extraction_id"]
    ).first()
    assert record is not None
    assert record.source == "report"
    assert record.analyzed is False
    assert json.loads(record.extracted_parameters)["hba1c"]["value"] == 6.5
    session.close()

    # A pending record is not part of the assessment history yet.
    history = client.get(
        "/api/v1/health/history", headers=_auth_headers(client)
    ).json()
    assert history["total"] == 0

    pending = client.get(
        "/api/v1/health/history",
        params={"include_pending": "true"},
        headers=_auth_headers(client),
    ).json()
    assert pending["total"] == 1
    assert pending["items"][0]["status"] == "pending"


def test_confirming_corrected_values_runs_the_assessment(client):
    with REPORT_PDF.open("rb") as report:
        preview = client.post(
            "/api/v1/health/upload-report/preview",
            files={"file": ("report.pdf", report, "application/pdf")},
            headers=_auth_headers(client),
        ).json()

    response = client.post(
        "/api/v1/health/upload-report/confirm",
        json={
            "extraction_id": preview["extraction_id"],
            # The glucose reading in the report was misread by the parser.
            "hba1c": 6.5,
            "glucose": 165,
            "bmi": 28.4,
            "age": 45,
            "symptoms": ["Fatigue / Low energy"],
        },
        headers=_auth_headers(client),
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["status"] == "success"
    assert body["workflow_status"] == "completed"
    assert body["final_assessment"]["disease"] == "diabetes"
    assert body["assessment"]["status"] == "completed"

    # The raw extraction is preserved next to the corrected values.
    session = client.session_factory()
    record = session.query(HealthRecord).filter(
        HealthRecord.id == preview["extraction_id"]
    ).first()
    assert record.analyzed is True
    assert record.glucose == 165
    assert json.loads(record.extracted_parameters)["glucose"]["value"] == 95
    assert json.loads(record.assessment)["score"] == body["final_assessment"]["score"]
    session.close()


def test_confirming_with_invalid_values_is_rejected(client):
    with REPORT_PDF.open("rb") as report:
        preview = client.post(
            "/api/v1/health/upload-report/preview",
            files={"file": ("report.pdf", report, "application/pdf")},
            headers=_auth_headers(client),
        ).json()

    response = client.post(
        "/api/v1/health/upload-report/confirm",
        json={
            "extraction_id": preview["extraction_id"],
            "hba1c": 6.5,
            "glucose": 9000,
            "bmi": 28.4,
            "age": 45,
        },
        headers=_auth_headers(client),
    )
    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"


def test_confirming_an_unknown_extraction_is_rejected(client):
    response = client.post(
        "/api/v1/health/upload-report/confirm",
        json={
            "extraction_id": 999999,
            "hba1c": 6.5,
            "glucose": 120,
            "bmi": 28.4,
            "age": 45,
        },
        headers=_auth_headers(client),
    )
    assert response.status_code == 404
    assert response.json()["status"] == "error"


def test_a_pending_record_has_no_analysis(client):
    with REPORT_PDF.open("rb") as report:
        preview = client.post(
            "/api/v1/health/upload-report/preview",
            files={"file": ("report.pdf", report, "application/pdf")},
            headers=_auth_headers(client),
        ).json()

    response = client.get(
        f"/api/v1/health/health-data/{preview['extraction_id']}/analysis",
        headers=_auth_headers(client),
    )
    assert response.status_code == 404


def test_uploading_a_non_pdf_is_rejected(client):
    response = client.post(
        "/api/v1/health/upload-report/preview",
        files={"file": ("notes.txt", b"plain text", "text/plain")},
        headers=_auth_headers(client),
    )
    assert response.status_code == 400
    assert response.json()["status"] == "error"
