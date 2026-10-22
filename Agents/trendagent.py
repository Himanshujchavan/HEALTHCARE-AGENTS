from typing import List, Dict, Any, Optional
from .plugins.base import DiseasePlugin

def _extract_param_value(h: Dict, param: str) -> Optional[float]:
    if not isinstance(h, dict):
        return None
    if "params" in h and isinstance(h["params"], dict) and param in h["params"]:
        val = h["params"][param]
        if isinstance(val, dict):
            val = val.get("value")
        try:
            return float(val) if val is not None else None
        except (ValueError, TypeError):
            return None
    analysis = h.get("analysis")
    if isinstance(analysis, dict):
        parameters = analysis.get("parameters")
        if isinstance(parameters, dict) and param in parameters:
            p_obj = parameters[param]
            val = p_obj.get("value") if isinstance(p_obj, dict) else p_obj
            try:
                return float(val) if val is not None else None
            except (ValueError, TypeError):
                return None
    if param in h:
        try:
            return float(h[param]) if h[param] is not None else None
        except (ValueError, TypeError):
            return None
    return None

class TrendAgent:
    """Analyzes health trends over time for a specific disease."""

    #: A trend needs at least this many prior records to be meaningful.
    MIN_HISTORY_RECORDS = 2

    @staticmethod
    def run_trend(history: List[Dict], plugin: DiseasePlugin, current: Dict) -> Dict[str, Any]:
        # Window includes the historical records plus the current record
        if not history or len(history) < TrendAgent.MIN_HISTORY_RECORDS:
            return {
                "trend_score": 0.0,
                "direction": "stable",
                "rising_but_normal": [],
                "visits_used": (len(history) + 1) if current else len(history),
                "history_records": len(history or []),
                "status": "insufficient_history",
                "reason": (
                    f"Trend requires at least {TrendAgent.MIN_HISTORY_RECORDS} prior "
                    f"records; {len(history or [])} available."
                ),
            }

        max_delta_score = 0.0
        overall_direction = "stable"
        rising_but_normal = []

        # Analyze parameters defined in plugin.trend_scales
        for param, scale in plugin.trend_scales.items():
            current_raw = current.get(param)
            if current_raw is None:
                continue
            try:
                current_val = float(current_raw)
            except (ValueError, TypeError):
                continue

            # Extract past values across history (oldest first)
            past_values = []
            for h in history:
                val = _extract_param_value(h, param)
                if val is not None:
                    past_values.append(val)

            if not past_values:
                continue

            # Oldest in history is first_val
            first_val = past_values[0]
            delta = current_val - first_val

            # Normalized score based on scale:
            # For positive scale: delta > 0 is deteriorating (rising is bad)
            # For negative scale: delta < 0 is deteriorating (falling is bad)
            raw_score = delta / scale
            param_score = max(0.0, min(1.0, raw_score))

            param_direction = "rising" if delta > 0 else ("falling" if delta < 0 else "stable")

            # Check rising_but_normal
            if delta > 0 and hasattr(plugin, "reference_ranges") and param in plugin.reference_ranges:
                r = plugin.reference_ranges[param]
                low = r.get("low", float("-inf"))
                high = r.get("high", float("inf"))
                if low <= current_val <= high:
                    rising_but_normal.append(param)

            if param_score > max_delta_score:
                max_delta_score = param_score
                overall_direction = param_direction

        total_visits = len(history) + 1
        changes = {}
        for param in plugin.trend_scales:
            current_value = _extract_param_value({"params": current}, param)
            previous_value = next(
                (_extract_param_value(h, param) for h in reversed(history)
                 if _extract_param_value(h, param) is not None),
                None,
            )
            oldest_value = next(
                (_extract_param_value(h, param) for h in history
                 if _extract_param_value(h, param) is not None),
                None,
            )
            if current_value is None or previous_value is None:
                continue
            delta = round(current_value - previous_value, 4)
            changes[param] = {
                "previous": previous_value,
                "current": current_value,
                "delta": delta,
                "direction": "increased" if delta > 0 else "decreased" if delta < 0 else "stable",
                # The window delta is the one the trend score is built on
                # (current minus the oldest record in the window).
                "oldest": oldest_value,
                "window_delta": (
                    round(current_value - oldest_value, 4)
                    if oldest_value is not None else None
                ),
            }

        return {
            "trend_score": round(max_delta_score, 4),
            "direction": overall_direction,
            "rising_but_normal": rising_but_normal,
            "visits_used": total_visits,
            "history_records": len(history),
            "status": "completed",
            "changes": changes,
        }
