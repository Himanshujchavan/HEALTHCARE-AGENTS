import {
  AssessmentHistoryItem,
  AssessmentHistoryResponse,
  AssessmentHistorySummary,
  ExtractionPreview,
  HealthDataSubmitResponse,
  InputSchema,
  UnifiedAssessment,
} from "../models/health";

/** The parameters the API accepts, as `/input-schema` returns them. */
export const inputSchema: InputSchema = {
  status: "success",
  parameters: {
    hba1c: { min: 3, max: 20, unit: "%", label: "HbA1c", required_for: ["diabetes"], type: "number" },
    glucose: { min: 40, max: 600, unit: "mg/dL", label: "Fasting glucose", required_for: ["diabetes"], type: "number" },
    bmi: { min: 10, max: 70, unit: "kg/m²", label: "Body mass index", required_for: ["diabetes"], type: "number" },
    homa_ir: { min: 0, max: 20, unit: "index", label: "HOMA-IR", required_for: [], type: "number" },
    age: { min: 1, max: 120, unit: "years", label: "Age", required_for: ["diabetes", "cardiovascular"], type: "number" },
    systolic_bp: { min: 50, max: 260, unit: "mmHg", label: "Systolic blood pressure", required_for: ["cardiovascular"], type: "number" },
    diastolic_bp: { min: 20, max: 160, unit: "mmHg", label: "Diastolic blood pressure", required_for: [], type: "number" },
    total_cholesterol: { min: 50, max: 500, unit: "mg/dL", label: "Total cholesterol", required_for: ["cardiovascular"], type: "number" },
    ldl: { min: 0, max: 400, unit: "mg/dL", label: "LDL cholesterol", required_for: [], type: "number" },
    hdl: { min: 10, max: 200, unit: "mg/dL", label: "HDL cholesterol", required_for: [], type: "number" },
    chest_pain_type: { min: 0, max: 3, unit: "category", label: "Chest pain category", required_for: [], type: "integer" },
    st_slope: { min: 0, max: 3, unit: "category", label: "ST segment slope category", required_for: [], type: "integer" },
    smoker: { min: null, max: null, unit: "", label: "Smoker", required_for: [], type: "boolean" },
  },
  diseases: {
    diabetes: {
      required: ["hba1c", "glucose", "bmi", "age"],
      optional: ["homa_ir"],
      reference_ranges: {},
      alert_thresholds: {},
    },
    cardiovascular: {
      required: ["age", "systolic_bp", "total_cholesterol"],
      optional: ["smoker", "diastolic_bp", "ldl", "hdl", "glucose", "bmi", "chest_pain_type", "st_slope"],
      reference_ranges: {},
      alert_thresholds: {},
    },
  },
  symptoms: ["Fatigue / Low energy"],
};

/** The preview the API returns after reading a report. */
export const extractionPreview: ExtractionPreview = {
  status: "pending",
  message: "Report parsed. Confirm the values to run the assessment.",
  extraction_id: 42,
  filename: "lab-report.pdf",
  source: "report",
  extracted_parameters: {
    hba1c: { value: 6.8, status: "High", unit: "%", note: null, source: "extracted" },
    glucose: { value: 148, status: "High", unit: "mg/dL", note: null, source: "extracted" },
    bmi: { value: 29, status: "Normal", unit: "kg/m²", note: null, source: "extracted" },
    age: { value: 45, status: "Reported", unit: "years", note: null, source: "extracted" },
  },
  missing_fields: { cardiovascular: ["systolic_bp", "total_cholesterol"] },
  available_diseases: ["diabetes"],
  created_at: "2026-10-22T10:00:00Z",
};

/** The unified assessment the workflow produces for the confirmed values. */
export const unifiedAssessment: UnifiedAssessment = {
  assessment_id: 42,
  status: "completed",
  assessments: {
    diabetes: {
      status: "completed",
      risk_level: "Moderate",
      score: 0.42,
      factors: ["HbA1c 6.8%", "Fasting glucose 148 mg/dL"],
      agent_status: { analysis: { status: "completed" } },
      model_confidence: "moderate",
      model_variant: "base",
    },
    cardiovascular: {
      status: "not_applicable",
      risk_level: "Unavailable",
      score: null,
      factors: [],
      reason: "No required parameters were submitted for this condition.",
    },
  },
  differential: { status: "completed", results: { primary: "diabetes" } },
  trend: { status: "insufficient_history" },
  alerts: [],
  explanation: "The submitted values indicate an elevated diabetes risk.",
  agent_status: {
    analysis: { status: "completed" },
    risk: { status: "completed" },
    symptom: { status: "skipped", reason: "No symptoms were reported." },
    alert: { status: "completed" },
  },
  created_at: "2026-10-22T10:05:00Z",
  model_disclaimer:
    "These are model predictions and risk signals, not confirmed medical diagnoses.",
};

/** The response the confirm endpoint returns. */
export const confirmResponse: HealthDataSubmitResponse = {
  status: "success",
  message: "Health data stored and processed through the master health workflow",
  record_id: 42,
  timestamp: "2026-10-22T10:05:00Z",
  workflow_status: "completed",
  risk_level: "Moderate",
  alert: false,
  report: "Moderate diabetes risk.",
  final_assessment: {
    disease: "diabetes",
    risk_level: "Moderate",
    score: 0.42,
    status: "completed",
  },
  unified: unifiedAssessment,
};

/** One entry of the assessment history. */
export const historyItem: AssessmentHistoryItem = {
  record_id: 42,
  created_at: "2026-10-22T10:05:00Z",
  source: "report",
  status: "completed",
  disease: "diabetes",
  risk_level: "Moderate",
  score: 0.42,
  diseases: {
    diabetes: { risk_level: "Moderate", score: 0.42, status: "completed" },
    cardiovascular: { risk_level: "Unavailable", score: null, status: "not_applicable" },
  },
  agent_status: { analysis: { status: "completed" } },
  parameters: { hba1c: 6.8, glucose: 148, bmi: 29, age: 45 },
};

export const historyResponse: AssessmentHistoryResponse = {
  status: "success",
  count: 1,
  total: 1,
  skip: 0,
  limit: 50,
  items: [historyItem],
};

export const historySummary: AssessmentHistorySummary = {
  status: "success",
  count: 2,
  parameters: {
    hba1c: [
      { record_id: 1, created_at: "2026-09-01T09:00:00Z", value: 6.1 },
      { record_id: 42, created_at: "2026-10-22T10:05:00Z", value: 6.8 },
    ],
    glucose: [
      { record_id: 1, created_at: "2026-09-01T09:00:00Z", value: 120 },
      { record_id: 42, created_at: "2026-10-22T10:05:00Z", value: 148 },
    ],
  },
  scores: [
    { record_id: 1, created_at: "2026-09-01T09:00:00Z", value: 0.31, risk_level: "Low", disease: "diabetes" },
    { record_id: 42, created_at: "2026-10-22T10:05:00Z", value: 0.42, risk_level: "Moderate", disease: "diabetes" },
  ],
  risk_levels: { Low: 1, Moderate: 1 },
};
