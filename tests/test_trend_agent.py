"""Tests for the Trend Agent."""

import pytest

from Agents.masterhealth import MasterHealthAgent
from Agents.plugins.diabetes import DiabetesPlugin
from Agents.trendagent import TrendAgent

CURRENT = {"hba1c": 5.9, "glucose": 95, "bmi": 22, "age": 40}


def _history(*hba1c_values):
    return [
        {"created_at": f"2024-0{index + 1}-01", "params": {"hba1c": value}}
        for index, value in enumerate(hba1c_values)
    ]


def test_single_prior_record_is_not_enough_for_a_trend():
    result = TrendAgent.run_trend(_history(5.4), DiabetesPlugin(), CURRENT)
    assert result["status"] == "insufficient_history"
    assert result["trend_score"] == 0.0
    assert "at least" in result["reason"]


def test_no_history_is_reported_as_insufficient():
    result = TrendAgent.run_trend([], DiabetesPlugin(), CURRENT)
    assert result["status"] == "insufficient_history"
    assert result["visits_used"] == 1


def test_rising_hba1c_produces_a_rising_trend():
    result = TrendAgent.run_trend(_history(5.4, 5.6, 5.9), DiabetesPlugin(), CURRENT)
    assert result["status"] == "completed"
    assert result["direction"] == "rising"
    assert result["trend_score"] == pytest.approx(1.0)
    assert result["visits_used"] == 4
    # The window delta (current minus oldest) drives the trend score ...
    assert result["changes"]["hba1c"]["window_delta"] == pytest.approx(0.5)
    # ... while "delta" reports the change since the previous visit.
    assert result["changes"]["hba1c"]["delta"] == pytest.approx(0.0)
    assert result["changes"]["hba1c"]["oldest"] == pytest.approx(5.4)


def test_stable_history_produces_no_trend_signal():
    result = TrendAgent.run_trend(_history(5.9, 5.9, 5.9), DiabetesPlugin(), CURRENT)
    assert result["status"] == "completed"
    assert result["trend_score"] == 0.0
    assert result["direction"] == "stable"


def test_history_raises_the_assessment_score():
    """A rising history must produce a higher score than the same visit alone."""
    agent = MasterHealthAgent()

    with_history = agent.process_health_data(
        CURRENT, history=_history(5.4, 5.6, 5.9)
    )
    without_history = agent.process_health_data(CURRENT)

    with_score = with_history["final_assessment"]["score"]
    without_score = without_history["final_assessment"]["score"]

    assert with_score > without_score
    assert with_history["final_assessment"]["formula"].startswith("0.50*risk")
    assert without_history["final_assessment"]["formula"].startswith("0.60*risk")
    assert with_history["final_assessment"]["trend_score"] > 0
    assert without_history["final_assessment"]["trend_score"] == 0


def test_trend_is_reported_per_disease():
    agent = MasterHealthAgent()
    result = agent.process_health_data(
        {
            "hba1c": 5.9, "glucose": 95, "bmi": 22, "age": 40,
            "systolic_bp": 150, "total_cholesterol": 220, "smoker": False,
        },
        history=[
            {"params": {"hba1c": 5.4, "systolic_bp": 140, "total_cholesterol": 210}},
            {"params": {"hba1c": 5.6, "systolic_bp": 145, "total_cholesterol": 215}},
        ],
    )
    trends = result["steps"]["5b_trend_analysis"]["all_trends"]
    assert set(trends) == {"diabetes", "cardiovascular"}
    assert trends["diabetes"]["status"] == "completed"
    assert trends["cardiovascular"]["status"] == "completed"
    # Each disease keeps its own trend.
    assert result["disease_results"]["diabetes"]["trend"] is trends["diabetes"]
    assert result["disease_results"]["cardiovascular"]["trend"] is trends["cardiovascular"]
