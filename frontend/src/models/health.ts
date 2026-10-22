import { JsonValue } from "./common";

export type RiskLevel = "Low" | "Moderate" | "High" | "Critical" | "Unknown" | string;

/**
 * The status vocabulary the backend uses, for a workflow, a single agent, or
 * a single disease assessment.
 *
 * The distinction matters: `completed_with_errors` means a usable score was
 * produced from fewer components than the documented formula, `failed` means
 * no score could be produced, and `unavailable` means the condition was never
 * applicable to the submitted values. They must not be collapsed into one
 * another in the UI.
 */
export type WorkflowStatus =
  | "completed"
  | "completed_with_errors"
  | "partial"
  | "failed"
  | "unavailable"
  | "not_applicable"
  | "skipped"
  | "pending"
  | "unreadable"
  | "unknown";

/** Status of a single agent inside the workflow. */
export interface AgentStatus {
  status: string;
  /** Why the agent is in this state, when it said so. */
  reason?: string | null;
  /** The raw workflow payload carries the failure detail under this key. */
  error?: string | null;
}

export interface DiseaseAssessment {
  status: WorkflowStatus | string;
  risk_level: string;
  score?: number | null;
  factors: string[];
  agent_status?: Record<string, AgentStatus>;
  analysis?: Record<string, JsonValue>;
  risk?: Record<string, JsonValue>;
  symptoms?: Record<string, JsonValue> | null;
  trend?: Record<string, JsonValue> | null;
  final_assessment?: Record<string, JsonValue>;
  /** How far the score should be trusted, when the disease was model-scored. */
  model_confidence?: string;
  /** Which model variant scored the case. */
  model_variant?: string;
  /** Why the assessment is in this state, when the backend says. */
  reason?: string | null;
}

export interface UnifiedAssessment {
  assessment_id?: number | null;
  status: string;
  assessments: {
    diabetes: DiseaseAssessment;
    cardiovascular: DiseaseAssessment;
  };
  differential: { status: string; results: Record<string, JsonValue> | JsonValue[] | null };
  trend: Record<string, JsonValue>;
  alerts: Array<{
    severity: string;
    alert: boolean;
    explanation?: string | null;
    recommended_next_steps?: string | null;
    triggers?: string[];
  }>;
  explanation: string | Record<string, JsonValue>;
  agent_status?: Record<string, AgentStatus>;
  created_at?: string | null;
  model_disclaimer: string;
}

/** The normalized assessment snapshot stored with every record. */
export interface AssessmentSnapshot {
  status: string;
  risk_level?: string | null;
  score?: number | null;
  disease?: string | null;
  formula?: string | null;
  components?: Record<string, JsonValue>;
  component_status?: Record<string, string>;
  coverage?: number | null;
  missing_components?: string[];
  skipped_components?: string[];
  abnormal_parameters?: string[];
  diseases?: Record<string, Record<string, JsonValue>>;
  agent_status?: Record<string, AgentStatus>;
  trend?: Record<string, JsonValue>;
  differential?: Record<string, JsonValue>;
  explanation?: string | null;
}

export interface HealthInput {
  hba1c?: number;
  glucose?: number;
  bmi?: number;
  homa_ir?: number;
  age?: number;
  systolic_bp?: number;
  diastolic_bp?: number;
  total_cholesterol?: number;
  ldl?: number;
  hdl?: number;
  smoker?: boolean;
  chest_pain_type?: number;
  st_slope?: number;
  symptoms: string[];
  manual_text?: string | null;
  use_llm?: boolean;
}

export interface HealthRecordResponse {
  id: number;
  user_id: number;
  hba1c?: number | null;
  glucose?: number | null;
  bmi?: number | null;
  age?: number | null;
  systolic_bp?: number | null;
  diastolic_bp?: number | null;
  total_cholesterol?: number | null;
  ldl?: number | null;
  hdl?: number | null;
  smoker?: boolean | null;
  chest_pain_type?: number | null;
  st_slope?: number | null;
  disease?: string;
  source?: "manual" | "report" | string;
  created_at: string;
  analyzed: boolean;
  analysis_result?: string | null;
  unified?: UnifiedAssessment | null;
}

export interface HealthDataSubmitResponse {
  status: string;
  message: string;
  record_id: number;
  timestamp: string;
  workflow_status?: string | null;
  risk_level?: string | null;
  alert?: boolean | null;
  report?: string | null;
  final_assessment?: Record<string, JsonValue> | null;
  disease?: Record<string, JsonValue> | null;
  explanation?: string | null;
  trend?: Record<string, JsonValue> | null;
  differential?: Record<string, JsonValue> | null;
  trend_result?: Record<string, JsonValue> | null;
  differential_result?: Record<string, JsonValue> | null;
  disease_results?: Record<string, JsonValue> | null;
  agent_status?: Record<string, AgentStatus> | null;
  assessment?: AssessmentSnapshot | null;
  unified?: UnifiedAssessment | null;
}

/** One parameter read from a report, with its clinical status. */
export interface ExtractedParameter {
  value?: number | null;
  status: string;
  unit?: string;
  note?: string | null;
  source?: "extracted" | "submitted" | string;
  /** Set when the user changed the extracted value. */
  corrected?: boolean;
}

/** Preview of the values extracted from an uploaded report. */
export interface ExtractionPreview {
  status: string;
  message: string;
  extraction_id: number;
  filename: string;
  source: string;
  extracted_parameters: Record<string, ExtractedParameter>;
  missing_fields: Record<string, string[]>;
  available_diseases: string[];
  created_at?: string | null;
}

/** Corrected values submitted after an extraction preview. */
export interface ExtractionConfirmRequest extends HealthInput {
  extraction_id: number;
}

/** What the API accepts for one parameter. */
export interface ParameterSpec {
  min?: number | null;
  max?: number | null;
  unit: string;
  label: string;
  /** The conditions that require this parameter. */
  required_for: string[];
  type: "number" | "integer" | "boolean";
}

/** What the API requires and accepts for one condition. */
export interface DiseaseSchema {
  required: string[];
  optional: string[];
  reference_ranges: Record<string, JsonValue>;
  alert_thresholds: Record<string, number>;
}

/** Describes the parameters the API accepts. */
export interface InputSchema {
  status: string;
  parameters: Record<string, ParameterSpec>;
  diseases: Record<string, DiseaseSchema>;
  symptoms: string[];
}

/** One entry of the assessment history. */
export interface AssessmentHistoryItem {
  record_id: number;
  created_at: string;
  source: string;
  status: string;
  disease?: string | null;
  risk_level?: string | null;
  score?: number | null;
  diseases: Record<string, Record<string, JsonValue>>;
  agent_status: Record<string, AgentStatus>;
  parameters: Record<string, JsonValue>;
}

export interface AssessmentHistoryResponse {
  status: string;
  count: number;
  total: number;
  skip: number;
  limit: number;
  items: AssessmentHistoryItem[];
}

export interface AssessmentSeriesPoint {
  record_id: number;
  created_at?: string | null;
  value?: number | null;
  risk_level?: string | null;
  disease?: string | null;
}

export interface AssessmentHistorySummary {
  status: string;
  count: number;
  parameters: Record<string, AssessmentSeriesPoint[]>;
  scores: AssessmentSeriesPoint[];
  risk_levels: Record<string, number>;
}

/**
 * The stored analysis payload.
 *
 * Its shape depends on which build wrote the record, so the fields the
 * interface does not know about stay reachable instead of being dropped.
 */
export interface AnalysisResult {
  [key: string]: unknown;
}

export interface HealthRecordAnalysisResponse {
  record_id: number;
  created_at: string;
  analysis: AnalysisResult;
}

export interface LatestRecordResponse {
  record: {
    id: number;
    hba1c?: number | null;
    glucose?: number | null;
    bmi?: number | null;
    age?: number | null;
    created_at: string;
  };
  analysis?: AnalysisResult;
  summary?: string;
  unified?: UnifiedAssessment;
}

export interface UploadReportResponse {
  status: string;
  message: string;
  filename: string;
  stored_path: string;
  extracted_parameters: string[];
  analysis: AnalysisResult;
  workflow_status?: string;
  final_assessment?: Record<string, JsonValue> | null;
  alert_result?: Record<string, JsonValue> | null;
  risk_level?: string | null;
  alert?: boolean | null;
  report?: string | null;
  error?: string | null;
  record_id?: number;
  assessment?: AssessmentSnapshot | null;
  unified?: UnifiedAssessment;
  steps?: Record<string, JsonValue>;
  [key: string]:
    | JsonValue
    | string[]
    | AnalysisResult
    | Record<string, JsonValue>
    | UnifiedAssessment
    | AssessmentSnapshot
    | null
    | undefined;
}

export interface ReanalyzeResponse {
  status: string;
  message: string;
  record_id: number;
  workflow_status?: string | null;
  risk_level?: string | null;
  alert?: boolean | null;
  report?: string | null;
}

export interface DeleteRecordResponse {
  status: string;
  message: string;
  record_id: number;
}
