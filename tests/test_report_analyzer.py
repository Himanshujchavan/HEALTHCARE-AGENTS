import json
from pathlib import Path

from Agents.reportanalyzer import ReportAnalyzerAgent


def test_report_analyzer_pdf_detects_abnormal():
    agent = ReportAnalyzerAgent()
    pdf_path = Path(__file__).resolve().parents[1] / "health_test_report_template.pdf"
    output = agent.analyze_pdf_report(str(pdf_path), use_llm=False)
    result = json.loads(output)

    # From the dummy PDF, expected extracted params are:
    # hba1c=6.5 (High), glucose=95 (Normal), bmi=28.4 (High), age=45 (Normal)
    assert result["abnormal_count"] == 2
    assert "parameters" in result
    assert "hba1c" in result["parameters"]
    assert result["parameters"]["hba1c"]["status"] == "High"
    assert result["parameters"]["bmi"]["status"] == "High"


def test_report_analyzer_pdf_extracts_expected_values():
    from tools.labparse import extract_health_data_from_pdf

    pdf_path = Path(__file__).resolve().parents[1] / "health_test_report_template.pdf"
    data = extract_health_data_from_pdf(str(pdf_path), use_llm=False)

    assert data["hba1c"] == 6.5
    assert data["glucose"] == 95.0
    assert data["bmi"] == 28.4
    assert data["age"] == 45.0


def test_report_analyzer_pdf_normal_case(monkeypatch, tmp_path):
    # Keep a small deterministic unit test for the normal-case branch
    # without depending on a specific PDF file/text extraction.
    from tools import labparse

    dummy_pdf = tmp_path / "lab_report.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

    def fake_extract(_pdf_path: str, use_llm: bool = True):
        return {"hba1c": 5.2, "glucose": 90.0, "bmi": 22.0, "age": 25.0}

    monkeypatch.setattr(labparse, "extract_health_data_from_pdf", fake_extract)

    agent = ReportAnalyzerAgent()
    output = agent.analyze_pdf_report(str(dummy_pdf), use_llm=False)
    result = json.loads(output)

    assert result["abnormal_count"] == 0


def test_report_analyzer_pdf_no_extractable_params(monkeypatch, tmp_path):
    from tools import labparse

    dummy_pdf = tmp_path / "lab_report.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

    def fake_extract(_pdf_path: str, use_llm: bool = True):
        return {"hba1c": None, "glucose": None, "bmi": None, "age": None}

    monkeypatch.setattr(labparse, "extract_health_data_from_pdf", fake_extract)

    agent = ReportAnalyzerAgent()
    output = agent.analyze_pdf_report(str(dummy_pdf), use_llm=False)
    result = json.loads(output)

    assert result["status"] == "failed"
    assert "No health parameters" in result["error"]
