from agent.tools import assess_risk, triage_route
import json


def test_assess_risk_critical():
    data = json.dumps({
        'symptoms': ['chest_pain'],
        'age': 70,
        'severity': 9,
        'duration_days': 1,
        'comorbidities': ['heart_disease']
    })
    result = assess_risk.invoke(data)
    assert 'CRITICAL' in result


def test_assess_risk_low():
    data = json.dumps({
        'symptoms': ['runny_nose'],
        'age': 25,
        'severity': 2,
        'duration_days': 1,
        'comorbidities': []
    })
    result = assess_risk.invoke(data)
    assert 'LOW' in result


def test_assess_risk_high():
    data = json.dumps({
        'symptoms': ['headache', 'dizziness'],
        'age': 58,
        'severity': 7,
        'duration_days': 2,
        'comorbidities': ['hypertension']
    })
    result = assess_risk.invoke(data)
    assert 'HIGH' in result


def test_triage_route_emergency():
    result = triage_route.invoke('CRITICAL')
    assert '911' in result or 'Emergency' in result


def test_triage_route_self_care():
    result = triage_route.invoke('LOW')
    assert 'SELF-CARE' in result