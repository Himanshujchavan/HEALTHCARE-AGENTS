import { JsonValue } from "./common";

export type RiskLevel = "Low" | "Moderate" | "High" | "Critical" | "Unknown" | string;

export interface HealthInput {
  hba1c?: number;
  glucose?: number;
  bmi?: number;
  age?: number;
  systolic_bp?: number;
  diastolic_bp?: number;
  total_cholesterol?: number;
  ldl?: number;
  hdl?: number;
  smoker?: boolean;
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
  disease?: string;
  created_at: string;
  analyzed: boolean;
  analysis_result?: string | null;
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
  disease_results?: Record<string, JsonValue> | null;
}

export interface AnalysisResult {
  [key: string]: JsonValue;
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
  steps?: Record<string, JsonValue>;
  [key: string]: JsonValue | string[] | AnalysisResult | Record<string, JsonValue> | undefined;
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
