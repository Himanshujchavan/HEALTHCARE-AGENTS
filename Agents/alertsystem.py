"""
Alert Agent - LangChain Implementation

Purpose:
	Final communication layer that transforms health-analysis outputs into:
	1) alert decision,
	2) user-friendly report,
	3) actionable notification message.

Input sources it can consume:
	- Master workflow output (analysis_result, risk_result, health_data)
	- Flat payloads with direct keys (diabetes_probability, hba1c, abnormal_parameters)

Default alert rule:
	Trigger alert if ANY condition is true:
		diabetes_probability > 0.7
		OR HbA1c > 7
		OR abnormal_parameters >= 3
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate

try:
	from langchain_community.llms import Ollama
except Exception:  # pragma: no cover - graceful import fallback
	Ollama = None

logger = logging.getLogger(__name__)


@dataclass
class AlertThresholds:
	diabetes_probability: float = 0.70
	hba1c: float = 7.0
	abnormal_parameters: int = 3


class AlertAgent:
	"""Evaluate risk and generate patient-friendly alerts/reports."""

	def __init__(
		self,
		thresholds: Optional[AlertThresholds] = None,
		use_llm: bool = True,
		llm_model: str = "qwen2.5:3b",
	):
		self.thresholds = thresholds or AlertThresholds()
		self.use_llm = use_llm
		self.llm_model = llm_model

	def process(self, result: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Build the final alert payload from aggregated health analysis output.

		Args:
			result: dict from master pipeline or flat health-risk payload.

		Returns:
			{
				"alert": bool,
				"risk_level": "Low|Moderate|High|Critical",
				"report": "...simple explanation...",
				"notification": "...warning or reassurance...",
				"triggers": ["..."]
			}
		"""
		evaluation = self._evaluate_risk(result)
		report = self._generate_report(result, evaluation)
		notification = self._build_notification(evaluation)

		return {
			"alert": evaluation["alert"],
			"risk_level": evaluation["risk_level"],
			"report": report,
			"notification": notification,
			"triggers": evaluation["triggers"],
			"metrics": evaluation["metrics"],
			"timestamp": datetime.now(timezone.utc).isoformat(),
		}

	def _evaluate_risk(self, result: Dict[str, Any]) -> Dict[str, Any]:
		"""Apply deterministic thresholds to decide whether to trigger alert."""
		metrics = self._extract_metrics(result)

		probability = metrics.get("diabetes_probability")
		hba1c = metrics.get("hba1c")
		abnormal_count = metrics.get("abnormal_parameters")

		triggers: List[str] = []

		if probability is not None and probability > self.thresholds.diabetes_probability:
			triggers.append(
				f"diabetes_probability {probability:.2f} > {self.thresholds.diabetes_probability:.2f}"
			)

		if hba1c is not None and hba1c > self.thresholds.hba1c:
			triggers.append(f"hba1c {hba1c:.2f} > {self.thresholds.hba1c:.2f}")

		if (
			abnormal_count is not None
			and abnormal_count >= self.thresholds.abnormal_parameters
		):
			triggers.append(
				"abnormal_parameters "
				f"{abnormal_count} >= {self.thresholds.abnormal_parameters}"
			)

		alert = len(triggers) > 0
		risk_level = self._derive_risk_level(probability, alert, len(triggers))

		return {
			"alert": alert,
			"risk_level": risk_level,
			"triggers": triggers,
			"metrics": metrics,
		}

	def _extract_metrics(self, result: Dict[str, Any]) -> Dict[str, Optional[float]]:
		"""Normalize risk metrics from multiple payload shapes."""
		risk_result = result.get("risk_result") or {}
		analysis_result = result.get("analysis_result") or {}
		health_data = result.get("health_data") or {}

		probability = (
			result.get("diabetes_probability")
			or result.get("risk_probability")
			or risk_result.get("risk_probability")
			or risk_result.get("risk_percentage")
		)

		hba1c = (
			result.get("hba1c")
			or health_data.get("hba1c")
			or health_data.get("HbA1c")
			or analysis_result.get("hba1c")
			or analysis_result.get("HbA1c")
		)

		abnormal_count = result.get("abnormal_parameters")
		if isinstance(abnormal_count, list):
			abnormal_count = len(abnormal_count)

		if abnormal_count is None:
			abnormal_count = analysis_result.get("abnormal_count")

		if abnormal_count is None:
			abnormal_parameters = analysis_result.get("abnormal_parameters") or []
			if isinstance(abnormal_parameters, list):
				abnormal_count = len(abnormal_parameters)

		glucose = (
			result.get("glucose")
			or health_data.get("glucose")
			or analysis_result.get("glucose")
		)
		bmi = result.get("bmi") or health_data.get("bmi") or analysis_result.get("bmi")

		return {
			"diabetes_probability": self._parse_probability(probability),
			"hba1c": self._to_float(hba1c),
			"abnormal_parameters": self._to_int(abnormal_count),
			"glucose": self._to_float(glucose),
			"bmi": self._to_float(bmi),
		}

	@staticmethod
	def _parse_probability(value: Any) -> Optional[float]:
		"""
		Convert probability representations to float in [0, 1].
		Supports: 0.82, "0.82", "82%", 82.
		"""
		if value is None:
			return None

		if isinstance(value, str):
			clean = value.strip()
			if clean.endswith("%"):
				clean = clean[:-1].strip()
				try:
					return float(clean) / 100.0
				except ValueError:
					return None
			try:
				numeric = float(clean)
			except ValueError:
				return None
		else:
			try:
				numeric = float(value)
			except (TypeError, ValueError):
				return None

		if numeric > 1:
			return numeric / 100.0
		if numeric < 0:
			return 0.0
		return numeric

	@staticmethod
	def _to_float(value: Any) -> Optional[float]:
		if value is None:
			return None
		try:
			return float(value)
		except (TypeError, ValueError):
			return None

	@staticmethod
	def _to_int(value: Any) -> Optional[int]:
		if value is None:
			return None
		try:
			return int(value)
		except (TypeError, ValueError):
			return None

	@staticmethod
	def _derive_risk_level(
		probability: Optional[float],
		alert_triggered: bool,
		trigger_count: int,
	) -> str:
		if probability is not None:
			if probability >= 0.85:
				return "Critical"
			if probability >= 0.70:
				return "High"
			if probability >= 0.40:
				return "Moderate"
			return "Low"

		if alert_triggered and trigger_count >= 2:
			return "High"
		if alert_triggered:
			return "Moderate"
		return "Low"

	def _generate_report(
		self,
		result: Dict[str, Any],
		evaluation: Dict[str, Any],
	) -> str:
		"""Generate plain-language report; uses LLM when available."""
		if self.use_llm:
			report = self._generate_report_with_llm(result, evaluation)
			if report:
				return report

		return self._generate_report_fallback(result, evaluation)

	def _generate_report_with_llm(
		self,
		result: Dict[str, Any],
		evaluation: Dict[str, Any],
	) -> Optional[str]:
		if Ollama is None:
			return None

		try:
			llm = Ollama(model=self.llm_model)
			context = self._build_compact_llm_context(result, evaluation)
			prompt = ChatPromptTemplate.from_messages(
				[
					(
						"system",
						"You are a healthcare communication assistant. "
						"Write a concise, patient-friendly report in simple English. "
						"Use 8-10 short sentences. Avoid medical jargon and do not diagnose. "
						"Explain only: risk level and probability, key parameters, "
						"and the symptom checker findings (selected symptoms + manual notes). "
						"Expand on what each parameter suggests in context, and explain how the "
						"reported symptoms align with the top hypothesis. "
						"Do not add treatment advice or generic warnings.",
					),
					(
						"user",
						"Context (key fields only): {context}\n"
						"Generate the final report text only.",
					),
				]
			)

			chain = prompt | llm
			response = chain.invoke(
				{
					"context": json.dumps(context, default=str),
				}
			)

			content = response.strip() if isinstance(response, str) else response.content.strip()
			if not content:
				return None
			return content
		except Exception as exc:
			logger.warning(f"Alert report LLM generation failed, using fallback: {exc}")
			return None

	def _generate_report_fallback(self, result: Dict[str, Any], evaluation: Dict[str, Any]) -> str:
		metrics = evaluation["metrics"]
		level = evaluation["risk_level"]

		lines: List[str] = []

		if metrics.get("diabetes_probability") is not None:
			percentage = int(round(metrics["diabetes_probability"] * 100))
			lines.append(f"Risk Level: {level}. Estimated diabetes risk is about {percentage}%.")
		else:
			lines.append(f"Risk Level: {level}.")

		param_parts: List[str] = []
		if metrics.get("hba1c") is not None:
			param_parts.append(f"HbA1c {metrics['hba1c']:.1f}%")
		if metrics.get("glucose") is not None:
			param_parts.append(f"Glucose {metrics['glucose']:.0f}")
		if metrics.get("bmi") is not None:
			param_parts.append(f"BMI {metrics['bmi']:.1f}")
		if metrics.get("abnormal_parameters") is not None:
			param_parts.append(f"Abnormal parameters {metrics['abnormal_parameters']}")
		if param_parts:
			lines.append("Key parameters: " + ", ".join(param_parts) + ".")

		symptom_context = self._build_symptom_context(result)
		if symptom_context:
			lines.append(symptom_context)

		return " ".join(lines).strip()

	def _build_compact_llm_context(
		self,
		result: Dict[str, Any],
		evaluation: Dict[str, Any],
	) -> Dict[str, Any]:
		metrics = evaluation.get("metrics", {})
		symptom_info = self._extract_symptom_info(result)
		symptom_result = result.get("symptom_result") or {}

		return {
			"risk_level": evaluation.get("risk_level"),
			"alert": evaluation.get("alert"),
			"triggers": evaluation.get("triggers") or [],
			"metrics": {
				"diabetes_probability": metrics.get("diabetes_probability"),
				"hba1c": metrics.get("hba1c"),
				"glucose": metrics.get("glucose"),
				"bmi": metrics.get("bmi"),
				"abnormal_parameters": metrics.get("abnormal_parameters"),
			},
			"abnormal_labs": self._collect_abnormal_labs(result, limit=5),
			"symptoms": (symptom_info.get("symptoms") or [])[:6],
			"matched_symptoms": (symptom_info.get("matched_symptoms") or [])[:5],
			"unmatched_symptoms": (symptom_info.get("unmatched_symptoms") or [])[:5],
			"symptom_alignment": symptom_result.get("symptom_alignment"),
			"severity_score": symptom_result.get("severity_score"),
			"top_hypothesis": symptom_info.get("top_hypothesis"),
			"manual_text": result.get("manual_text"),
		}

	def _build_lab_summary(self, result: Dict[str, Any]) -> str:
		analysis_result = result.get("analysis_result") or {}
		params = analysis_result.get("parameters") or {}
		if not isinstance(params, dict) or not params:
			return ""

		abnormal_items = []
		normal_items = []
		for key, details in params.items():
			if not isinstance(details, dict):
				continue
			status = details.get("status")
			value = details.get("value")
			unit = details.get("unit") or ""
			note = details.get("note")
			label = f"{key.upper()} {value} {unit}".strip()
			if note:
				label = f"{label} ({note})"
			if status and status != "Normal":
				abnormal_items.append(label)
			else:
				normal_items.append(label)

		parts = []
		if abnormal_items:
			parts.append("Abnormal: " + ", ".join(abnormal_items))
		if normal_items:
			parts.append("Normal: " + ", ".join(normal_items[:3]))
		return "; ".join(parts)

	def _collect_abnormal_labs(
		self,
		result: Dict[str, Any],
		limit: int = 5,
	) -> List[Dict[str, Any]]:
		analysis_result = result.get("analysis_result") or {}
		params = analysis_result.get("parameters") or {}
		if not isinstance(params, dict):
			return []

		abnormal_items: List[Dict[str, Any]] = []
		for key, details in params.items():
			if not isinstance(details, dict):
				continue
			status = details.get("status")
			if status and status != "Normal":
				item = {
					"name": key,
					"value": details.get("value"),
					"unit": details.get("unit") or "",
					"status": status,
				}
				note = details.get("note")
				if note:
					item["note"] = note
				abnormal_items.append(item)

		return abnormal_items[:limit]

	def _build_symptom_context(self, result: Dict[str, Any]) -> str:
		info = self._extract_symptom_info(result)
		manual_text = result.get("manual_text")
		if not info["symptoms"] and not info["unmatched_symptoms"]:
			return "" if not manual_text else f"Manual notes: {manual_text.strip()}."

		lines: List[str] = []
		if info["matched_symptoms"] and info["top_hypothesis"]:
			lines.append(
				"Symptoms matching diabetes patterns ("
				f"{info['top_hypothesis']}): "
				+ ", ".join(info["matched_symptoms"][:5])
				+ ".")
		elif info["symptoms"]:
			lines.append("Symptoms reported: " + ", ".join(info["symptoms"][:6]) + ".")

		if info["unmatched_symptoms"]:
			lines.append(
				"Symptoms not typical for diabetes patterns were noted: "
				+ ", ".join(info["unmatched_symptoms"][:6])
				+ "."
			)

		if manual_text:
			cleaned = manual_text.strip()
			if cleaned:
				lines.append(f"Manual notes: {cleaned}.")

		return " ".join(lines).strip()

	def _extract_symptom_info(self, result: Dict[str, Any]) -> Dict[str, Any]:
		symptoms = result.get("symptoms") or []
		if isinstance(symptoms, str):
			symptoms = [s.strip() for s in symptoms.split(",") if s.strip()]
		if not isinstance(symptoms, list):
			symptoms = []

		symptom_result = result.get("symptom_result") or {}
		mapping = symptom_result.get("symptom_mapping") or {}
		matched_symptoms: List[str] = []
		if mapping.get("condition_hypotheses"):
			first = mapping["condition_hypotheses"][0]
			matched_symptoms = first.get("matched_symptoms") or []
		unmatched = mapping.get("unmatched_symptoms") or []

		return {
			"symptoms": symptoms,
			"top_hypothesis": mapping.get("top_hypothesis"),
			"matched_symptoms": matched_symptoms,
			"unmatched_symptoms": unmatched,
		}

	def _build_non_diabetes_notes(self, symptoms: List[str], unmatched: List[str]) -> List[str]:
		all_symptoms = [s.lower() for s in (symptoms or [])]
		all_symptoms += [s.lower() for s in (unmatched or [])]
		text = " | ".join(all_symptoms)

		notes = []
		if "chest" in text or "chest discomfort" in text:
			notes.append(
				"Chest discomfort can be related to heart or lung issues; seek urgent care if severe, worsening, or with shortness of breath."
			)
		if "shortness of breath" in text:
			notes.append(
				"Shortness of breath can have heart, lung, anemia, or infection causes; urgent evaluation is needed if it is sudden or severe."
			)
		if "dizziness" in text or "lightheaded" in text:
			notes.append(
				"Dizziness can be linked to dehydration, blood pressure changes, inner ear issues, or medication effects; fainting is a red flag."
			)
		if "tingling" in text or "numb" in text:
			notes.append(
				"Tingling can also come from nerve compression or vitamin deficiencies, not only diabetes-related neuropathy."
			)
		if "blurred vision" in text or "vision" in text:
			notes.append(
				"Blurred vision can also be related to eye conditions or blood pressure changes."
			)
		if "weight loss" in text:
			notes.append(
				"Unexplained weight loss can have multiple causes, including thyroid or other systemic conditions."
			)
		if "fatigue" in text or "low energy" in text:
			notes.append(
				"Fatigue can be related to sleep issues, anemia, thyroid problems, or infection."
			)
		return notes

	def _build_notification(self, evaluation: Dict[str, Any]) -> str:
		"""Create short warning/reassurance notification text."""
		if evaluation["alert"]:
			return (
				"High Diabetes Risk Detected. "
				"Your values suggest increased risk. "
				"Please consult a healthcare professional for further evaluation."
			)

		return (
			"No immediate diabetes alert. "
			"Your current values appear within acceptable limits. "
			"Maintain healthy habits and monitor regularly."
		)


def run_alert_assessment(result: Dict[str, Any]) -> Dict[str, Any]:
	"""Convenience wrapper for one-shot alert generation."""
	agent = AlertAgent()
	return agent.process(result)


if __name__ == "__main__":
	sample = {
		"health_data": {"hba1c": 6.8, "glucose": 148, "bmi": 29},
		"risk_result": {"risk_probability": 0.82},
		"analysis_result": {"abnormal_count": 3},
	}
	print(json.dumps(run_alert_assessment(sample), indent=2))
