// src/models/assessment.ts
export interface AssessmentFactor {
  name: string;
  value: string | number | null;
}

export interface DiseaseAssessment {
  risk_level: string; // e.g. "moderate", "critical"
  score: number | null; // 0‑1 or null if not applicable
  factors: AssessmentFactor[]; // human‑readable reasons
}

export interface DifferentialResult {
  status: string; // "completed" | "failed" | "skipped"
  results: any[]; // raw differential output
}

export interface TrendResult {
  status: string; // "completed" | "insufficient_history"
  results: any[]; // each entry contains trend_score, direction, etc.
}

export interface AlertInfo {
  severity: string; // "high" | "medium" | "low"
  explanation: string;
  recommended_next_steps: string[];
}

export interface AssessmentResponse {
  assessment_id: string;
  status: string; // "completed", "completed_with_errors", "failed"
  created_at: string; // ISO timestamp
  assessments: {
    diabetes: DiseaseAssessment;
    cardiovascular: DiseaseAssessment;
  };
  differential: DifferentialResult;
  trend: TrendResult;
  alerts: AlertInfo[];
  explanation: Record<string, any>;
}
