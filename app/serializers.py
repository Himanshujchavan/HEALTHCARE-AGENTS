"""Consistent parsing and normalization of stored health records.

Health records store three JSON payloads: the workflow result, the extracted
report parameters and the assessment snapshot. They were written by different
versions of the code, so reading them back has to be defensive: a legacy
record may have no ``final_assessment``, a truncated string, or a shape the
current workflow no longer produces.

Every read path in the API goes through this module so a record always
answers the same questions the same way.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

#: Parameters copied from a health record into a history entry.
HISTORY_PARAMETERS = [
    "hba1c", "glucose", "bmi", "homa_ir", "age",
    "systolic_bp", "diastolic_bp", "total_cholesterol",
    "ldl", "hdl", "smoker", "chest_pain_type", "st_slope",
]


def parse_json_object(raw: Any) -> Dict[str, Any]:
    """Parse a JSON payload into a dict without ever raising.

    Accepts an already-decoded dict, a JSON string or ``None``.
    """
    if isinstance(raw, dict):
        return dict(raw)
    if not raw or not isinstance(raw, str):
        return {}
    try:
        data = json.loads(raw)
    except (TypeError, ValueError) as exc:
        logger.warning("Could not parse stored JSON payload: %s", exc)
        return {}
    return data if isinstance(data, dict) else {}


def parse_analysis_result(raw: Any) -> Dict[str, Any]:
    """Parse a stored analysis result, tolerating older result shapes.

    Records written before the orchestrated workflow have no
    ``workflow_status``; they are reported as ``completed`` when they carry
    a usable assessment and ``unknown`` otherwise.
    """
    data = parse_json_object(raw)
    if not data:
        return {}
    if "workflow_status" not in data:
        data["workflow_status"] = (
            "completed" if data.get("final_assessment") or data.get("risk_level")
            else "unknown"
        )
    return data


def parse_extracted_parameters(raw: Any) -> Dict[str, Dict[str, Any]]:
    """Parse the stored report-extraction payload.

    Returns a mapping of parameter name to
    ``{"value", "status", "unit", "note"}``.
    """
    data = parse_json_object(raw)
    if not data:
        return {}

    parsed: Dict[str, Dict[str, Any]] = {}
    for name, entry in data.items():
        if isinstance(entry, dict):
            parsed[name] = {
                "value": entry.get("value"),
                "status": entry.get("status", "Unknown"),
                "unit": entry.get("unit", ""),
                "note": entry.get("note"),
            }
        else:
            parsed[name] = {
                "value": entry,
                "status": "Extracted",
                "unit": "",
                "note": None,
            }
    return parsed


def serialize_extracted_parameters(parameters: Dict[str, Any]) -> Optional[str]:
    """Serialize extracted parameters for storage, or ``None`` when empty."""
    if not parameters:
        return None
    return json.dumps(parameters, default=str)


def normalize_assessment(analysis: Any) -> Dict[str, Any]:
    """Return the normalized assessment snapshot for a stored analysis.

    The snapshot is the stable, comparable view of an assessment: the same
    keys are present whether the record was produced by the current
    orchestrated workflow or by an earlier analysis format.
    """
    data = parse_analysis_result(analysis)
    final = data.get("final_assessment") or {}
    if not isinstance(final, dict):
        final = {}

    risk_level = final.get("risk_level") or data.get("risk_level")
    score = final.get("score")
    if score is None:
        score = final.get("risk_score")
    if score is None:
        score = data.get("score")

    diseases: Dict[str, Dict[str, Any]] = {}
    all_diseases = final.get("all_diseases") or {}
    if isinstance(all_diseases, dict):
        for name, entry in all_diseases.items():
            if not isinstance(entry, dict):
                continue
            diseases[name] = {
                "risk_level": entry.get("risk_level"),
                "score": entry.get("score"),
                "status": entry.get("status"),
                "formula": entry.get("formula"),
            }

    status = data.get("workflow_status") or "unknown"
    if status == "completed" and final.get("status") in {
        "completed_with_errors", "failed"
    }:
        status = final["status"]

    return {
        "status": status,
        "risk_level": risk_level,
        "score": score,
        "disease": final.get("disease") or data.get("disease"),
        "formula": final.get("formula"),
        "components": final.get("components") or {},
        "component_status": final.get("component_status") or {},
        "coverage": final.get("coverage"),
        "missing_components": final.get("missing_components") or [],
        "skipped_components": final.get("skipped_components") or [],
        "abnormal_parameters": data.get("abnormal_parameters") or [],
        "diseases": diseases,
        "agent_status": data.get("agent_status") or {},
        "trend": data.get("trend_result") or {},
        "differential": data.get("differential_result") or {},
        "explanation": data.get("explanation"),
    }


def serialize_assessment(assessment: Dict[str, Any]) -> Optional[str]:
    """Serialize an assessment snapshot for storage."""
    if not assessment:
        return None
    return json.dumps(assessment, default=str)


def record_parameters(record: Any) -> Dict[str, Any]:
    """Extract the clinical parameters stored on a health record."""
    values: Dict[str, Any] = {}
    for field in HISTORY_PARAMETERS:
        value = getattr(record, field, None)
        if value is not None:
            values[field] = value
    return values


def record_assessment_item(record: Any) -> Dict[str, Any]:
    """Build one history entry from a health record.

    The record's stored assessment snapshot is preferred; when it is missing
    (a record analysed by an older build) the stored analysis result is
    normalized on the fly, so history never shows a blank entry.
    """
    raw_result = getattr(record, "analysis_result", None)
    analyzed = bool(getattr(record, "analyzed", False))
    analysis = parse_analysis_result(raw_result)
    snapshot = parse_json_object(getattr(record, "assessment", None))
    if not snapshot and analysis:
        snapshot = normalize_assessment(analysis)

    if not analysis and not snapshot:
        # Nothing has been assessed yet: the record is still pending. A
        # record that claims to be analysed but cannot be read is reported
        # as unreadable rather than silently treated as pending.
        status = "unreadable" if analyzed else "pending"
    else:
        status = snapshot.get("status", "unknown")

    return {
        "record_id": record.id,
        "created_at": record.created_at,
        "source": getattr(record, "source", None) or "manual",
        "status": status,
        "disease": snapshot.get("disease"),
        "risk_level": snapshot.get("risk_level"),
        "score": snapshot.get("score"),
        "diseases": snapshot.get("diseases") or {},
        "agent_status": snapshot.get("agent_status") or {},
        "parameters": record_parameters(record),
    }


def assessment_series(records: List[Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Build per-parameter and score time series from health records."""
    parameters: Dict[str, List[Dict[str, Any]]] = {}
    scores: List[Dict[str, Any]] = []

    for record in records:
        values = record_parameters(record)
        for name, value in values.items():
            parameters.setdefault(name, []).append({
                "record_id": record.id,
                "created_at": record.created_at,
                "value": value,
            })

        item = record_assessment_item(record)
        scores.append({
            "record_id": record.id,
            "created_at": record.created_at,
            "value": item.get("score"),
            "risk_level": item.get("risk_level"),
            "disease": item.get("disease"),
        })

    return {"parameters": parameters, "scores": scores}
