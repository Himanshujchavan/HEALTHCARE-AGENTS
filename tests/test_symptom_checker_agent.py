from Agents.symptomchecker import analyze_symptoms


def test_symptom_checker_alignment():
    symptoms = [
        "Fatigue / Low energy",
        "Polyuria (frequent urination)",
    ]

    result = analyze_symptoms(
        symptoms=symptoms,
        report_context={},
        risk_context={},
        manual_text="",
        use_llm=False,
    )

    assert "input_summary" in result


def test_symptom_checker_no_symptoms():
    result = analyze_symptoms(
        symptoms=[],
        report_context={},
        risk_context={},
        manual_text="",
        use_llm=False,
    )

    assert result is not None
