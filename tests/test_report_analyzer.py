import pytest
from Agents.reportanalyzer import analyze_health_data


def test_report_analyzer_detects_abnormal():
    data = {
        "hba1c": 6.8,
        "glucose": 148,
        "bmi": 29,
        "age": 45,
    }

    result = analyze_health_data(data)

    assert result["abnormal_count"] >= 2
    assert "parameters" in result
    assert "hba1c" in result["parameters"]


def test_report_analyzer_normal_case():
    data = {
        "hba1c": 5.2,
        "glucose": 90,
        "bmi": 22,
        "age": 25,
    }

    result = analyze_health_data(data)

    assert result["abnormal_count"] == 0
