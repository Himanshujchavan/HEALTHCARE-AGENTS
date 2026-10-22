"""Tests for the Disease Registry.

The registry is the bridge between the Master Health Agent and the
disease-specific plugins: it decides which plugins apply to a submission and
makes them reachable regardless of module import order.
"""

from Agents.plugins import registry
from Agents.plugins.base import DiseasePlugin
from Agents.plugins.boot import init_plugins
from Agents.plugins.cardiovascular import CardiovascularPlugin
from Agents.plugins.diabetes import DiabetesPlugin


DIABETES_DATA = {"hba1c": 6.8, "glucose": 148, "bmi": 29, "age": 45}
CARDIO_DATA = {
    "age": 55,
    "systolic_bp": 150,
    "total_cholesterol": 220,
    "smoker": True,
}


def test_builtin_plugins_are_registered():
    names = registry.list_all()
    assert "diabetes" in names
    assert "cardiovascular" in names


def test_registry_is_self_initializing():
    """The registry works without the master agent having been imported first."""
    # A fresh registry instance must still resolve the built-in plugins.
    plugin = registry.get("diabetes")
    assert isinstance(plugin, DiabetesPlugin)


def test_get_returns_plugin_instances():
    assert isinstance(registry.get("diabetes"), DiseasePlugin)
    assert isinstance(registry.get("cardiovascular"), CardiovascularPlugin)
    assert registry.get("does-not-exist") is None


def test_applicable_selects_diabetes_only():
    plugins = registry.applicable(DIABETES_DATA)
    assert [p.name for p in plugins] == ["diabetes"]


def test_applicable_selects_cardiovascular_only():
    plugins = registry.applicable(CARDIO_DATA)
    assert [p.name for p in plugins] == ["cardiovascular"]


def test_applicable_selects_both_for_a_two_disease_submission():
    combined = {**DIABETES_DATA, **CARDIO_DATA}
    plugins = registry.applicable(combined)
    assert {p.name for p in plugins} == {"diabetes", "cardiovascular"}


def test_applicable_ignores_none_valued_fields():
    """A field that was submitted as null does not make a plugin applicable."""
    data = {**DIABETES_DATA, "hba1c": None}
    plugins = registry.applicable(data)
    assert "diabetes" not in [p.name for p in plugins]


def test_missing_fields_reports_what_is_absent():
    plugin = registry.get("cardiovascular")
    missing = registry.missing_fields({"age": 55}, plugin)
    assert set(missing) == {"systolic_bp", "total_cholesterol"}


def test_init_plugins_is_idempotent():
    before = registry.list_all()
    init_plugins()
    assert registry.list_all() == before


def test_plugins_declare_their_contract():
    for name in ("diabetes", "cardiovascular"):
        plugin = registry.get(name)
        assert plugin.required_fields
        assert plugin.reference_ranges
        assert plugin.trend_scales
        assert plugin.symptom_profiles
        assert plugin.alert_thresholds
        # Every plugin speaks the same method set.
        for method in ("flags", "notes", "rule_risk", "predict_risk", "score_symptoms"):
            assert callable(getattr(plugin, method))
