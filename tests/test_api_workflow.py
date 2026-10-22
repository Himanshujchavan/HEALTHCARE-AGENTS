"""End-to-end API test for the orchestrated workflow.

Exercises the full chain — JWT auth, persistence, the LangGraph workflow and
the unified response — against a throwaway database.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.healthroutes import router as health_router
from database.config import Base, get_db
from database.models import User
from utils.helpers import create_access_token, get_password_hash


@pytest.fixture()
def client(tmp_path):
    """A test client backed by a throwaway database with one registered user."""
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

    with TestClient(app) as test_client:
        test_client.user_id = user_id
        yield test_client


def _auth_headers(client):
    token = create_access_token({"sub": "tester", "user_id": client.user_id})
    return {"Authorization": f"Bearer {token}"}


def test_health_data_submission_runs_the_full_workflow(client):
    response = client.post(
        "/api/v1/health/health-data",
        json={
            "hba1c": 6.8,
            "glucose": 148,
            "bmi": 29,
            "age": 45,
            "symptoms": ["Fatigue / Low energy"],
        },
        headers=_auth_headers(client),
    )

    assert response.status_code == 200, response.text
    body = response.json()

    assert body["status"] == "success"
    assert body["workflow_status"] == "completed"
    assert body["record_id"] > 0
    assert body["final_assessment"]["disease"] == "diabetes"
    assert body["final_assessment"]["score"] is not None
    assert body["explanation"]
    assert body["disease_results"]["diabetes"]["agent_status"]["analysis"]["status"] == "completed"


def test_unified_response_is_returned(client):
    response = client.post(
        "/api/v1/health/health-data",
        json={"hba1c": 5.2, "glucose": 90, "bmi": 22, "age": 30},
        headers=_auth_headers(client),
    )
    assert response.status_code == 200, response.text
    unified = response.json()["unified"]

    assert unified["status"] == "completed"
    assert unified["assessments"]["diabetes"]["status"] in (
        "completed",
        "completed_with_errors",
    )
    assert unified["assessments"]["cardiovascular"]["status"] == "not_applicable"
    assert unified["agent_status"]
    assert unified["model_disclaimer"]


def test_two_disease_submission_returns_both_results(client):
    response = client.post(
        "/api/v1/health/health-data",
        json={
            "hba1c": 6.8,
            "glucose": 148,
            "bmi": 29,
            "age": 55,
            "systolic_bp": 165,
            "diastolic_bp": 95,
            "total_cholesterol": 250,
            "ldl": 140,
            "hdl": 38,
            "smoker": True,
            "symptoms": ["Fatigue / Low energy", "chest pain"],
        },
        headers=_auth_headers(client),
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert set(body["disease_results"]) == {"diabetes", "cardiovascular"}
    assert body["differential_result"]["primary"] in ("diabetes", "cardiovascular")
    assert set(body["final_assessment"]["all_diseases"]) == {"diabetes", "cardiovascular"}


def test_stored_analysis_can_be_read_back(client):
    created = client.post(
        "/api/v1/health/health-data",
        json={"hba1c": 6.8, "glucose": 148, "bmi": 29, "age": 45},
        headers=_auth_headers(client),
    )
    record_id = created.json()["record_id"]

    response = client.get(
        f"/api/v1/health/health-data/{record_id}/analysis",
        headers=_auth_headers(client),
    )
    assert response.status_code == 200, response.text
    analysis = response.json()["analysis"]
    assert analysis["final_assessment"]["disease"] == "diabetes"
    assert analysis["unified"]["assessments"]["diabetes"]["score"] is not None


def test_reanalysis_reproduces_the_original_score(client):
    """Re-analysing a stored record must give the same assessment."""
    created = client.post(
        "/api/v1/health/health-data",
        json={
            "hba1c": 6.8,
            "glucose": 148,
            "bmi": 29,
            "age": 45,
            "symptoms": ["Fatigue / Low energy"],
        },
        headers=_auth_headers(client),
    )
    record_id = created.json()["record_id"]
    original_score = created.json()["final_assessment"]["score"]

    reanalyzed = client.post(
        f"/api/v1/health/analyze/{record_id}",
        headers=_auth_headers(client),
    )
    assert reanalyzed.status_code == 200, reanalyzed.text

    stored = client.get(
        f"/api/v1/health/health-data/{record_id}/analysis",
        headers=_auth_headers(client),
    ).json()["analysis"]
    assert stored["final_assessment"]["score"] == original_score


def test_incomplete_submission_is_rejected(client):
    response = client.post(
        "/api/v1/health/health-data",
        json={"age": 30},
        headers=_auth_headers(client),
    )
    assert response.status_code == 422


def test_unauthenticated_request_is_rejected(client):
    response = client.post(
        "/api/v1/health/health-data",
        json={"hba1c": 6.8, "glucose": 148, "bmi": 29, "age": 45},
    )
    assert response.status_code == 401
