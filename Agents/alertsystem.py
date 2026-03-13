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
		llm_model: str = "mistral",
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

		return self._generate_report_fallback(evaluation)

	def _generate_report_with_llm(
		self,
		result: Dict[str, Any],
		evaluation: Dict[str, Any],
	) -> Optional[str]:
		if Ollama is None:
			return None

		try:
			llm = Ollama(model=self.llm_model)
			prompt = ChatPromptTemplate.from_messages(
				[
					(
						"system",
						"You are a healthcare communication assistant. "
						"Write a short, clear, patient-friendly report in simple English. "
						"Avoid medical jargon. Keep to 3-5 sentences. "
						"Include: what is abnormal, risk level, and what to do next.",
					),
					(
						"user",
						"Health system output: {result}\n"
						"Evaluation summary: {evaluation}\n"
						"Generate the final report text only.",
					),
				]
			)

			chain = prompt | llm
			response = chain.invoke(
				{
					"result": json.dumps(result, default=str),
					"evaluation": json.dumps(evaluation, default=str),
				}
			)

			content = response.strip() if isinstance(response, str) else response.content.strip()
			if not content:
				return None
			return content
		except Exception as exc:
			logger.warning(f"Alert report LLM generation failed, using fallback: {exc}")
			return None

	def _generate_report_fallback(self, evaluation: Dict[str, Any]) -> str:
		metrics = evaluation["metrics"]
		level = evaluation["risk_level"]
		alert = evaluation["alert"]

		notes: List[str] = []

		if metrics.get("hba1c") is not None:
			if metrics["hba1c"] > self.thresholds.hba1c:
				notes.append("Your HbA1c level is above the normal range.")
			else:
				notes.append("Your HbA1c level is currently within acceptable range.")

		if metrics.get("glucose") is not None:
			if metrics["glucose"] >= 126:
				notes.append("Your glucose value is elevated and needs attention.")
			elif metrics["glucose"] >= 100:
				notes.append("Your glucose is slightly high and should be monitored.")
			else:
				notes.append("Your glucose is in the expected range.")

		if metrics.get("bmi") is not None:
			if metrics["bmi"] >= 30:
				notes.append("Your BMI suggests obesity, which can increase diabetes risk.")
			elif metrics["bmi"] >= 25:
				notes.append("Your BMI is in the overweight range.")

		if metrics.get("diabetes_probability") is not None:
			percentage = int(round(metrics["diabetes_probability"] * 100))
			notes.append(f"Estimated diabetes risk probability is about {percentage}%.")

		recommendation = (
			"Please consult a healthcare professional soon for a complete evaluation."
			if alert
			else "Continue a healthy lifestyle and schedule routine monitoring."
		)

		base = " ".join(notes).strip()
		if not base:
			base = "Your recent health data was reviewed."

		return f"{base} Risk Level: {level}. {recommendation}"

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
