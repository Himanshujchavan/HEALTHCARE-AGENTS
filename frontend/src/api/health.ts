import { http } from "./http";
import {
  HealthDataSubmitResponse,
  HealthInput,
  HealthRecordAnalysisResponse,
  HealthRecordResponse,
  LatestRecordResponse,
  ReanalyzeResponse,
  DeleteRecordResponse,
  UploadReportResponse,
  UnifiedAssessment,
  AssessmentHistoryResponse,
  AssessmentHistorySummary,
  ExtractionPreview,
  ExtractionConfirmRequest,
  InputSchema,
} from "../models/health";

export async function uploadHealthReport(
  file: File,
  useLLM: boolean = false,
  symptoms: string[] = [],
  manualText: string = "",
  disease: "diabetes" | "cardiovascular" = "diabetes",
): Promise<UploadReportResponse> {
  const formData = new FormData();
  formData.append("file", file);
  symptoms.forEach((symptom) => formData.append("symptoms", symptom));
  if (manualText.trim()) {
    formData.append("manual_text", manualText.trim());
  }
  formData.append("disease", disease);

  const response = await http.post<UploadReportResponse>(
    "/api/v1/health/upload-report",
    formData,
    {
      params: { use_llm: useLLM },
      headers: { "Content-Type": "multipart/form-data" },
    },
  );

  return response.data;
}

export async function submitHealthData(
  payload: HealthInput,
): Promise<HealthDataSubmitResponse> {
  const response = await http.post<HealthDataSubmitResponse>(
    "/api/v1/health/health-data",
    payload,
  );
  return response.data;
}

export async function getHealthRecord(recordId: number): Promise<HealthRecordResponse> {
  const response = await http.get<HealthRecordResponse>(
    `/api/v1/health/health-data/${recordId}`,
  );
  return response.data;
}

export async function getHealthRecordAnalysis(
  recordId: number,
): Promise<HealthRecordAnalysisResponse> {
  const response = await http.get<HealthRecordAnalysisResponse>(
    `/api/v1/health/health-data/${recordId}/analysis`,
  );
  return response.data;
}

export async function getHealthRecords(
  skip: number = 0,
  limit: number = 100,
): Promise<HealthRecordResponse[]> {
  const response = await http.get<HealthRecordResponse[]>(
    "/api/v1/health/health-data",
    { params: { skip, limit } },
  );
  return response.data;
}

export async function getLatestRecord(): Promise<LatestRecordResponse> {
  const response = await http.get<LatestRecordResponse>("/api/v1/health/latest");
  return response.data;
}

export async function getUnifiedAssessment(recordId: number): Promise<UnifiedAssessment> {
  const response = await http.get<{ unified: UnifiedAssessment }>(
    `/api/v1/health/health-data/${recordId}/analysis`,
  );
  return response.data.unified;
}

export async function reanalyzeRecord(recordId: number): Promise<ReanalyzeResponse> {
  const response = await http.post<ReanalyzeResponse>(
    `/api/v1/health/analyze/${recordId}`,
  );
  return response.data;
}

export async function deleteHealthRecord(
  recordId: number,
): Promise<DeleteRecordResponse> {
  const response = await http.delete<DeleteRecordResponse>(
    `/api/v1/health/health-data/${recordId}`,
  );
  return response.data;
}

/** The parameters and conditions the API accepts. */
export async function getInputSchema(): Promise<InputSchema> {
  const response = await http.get<InputSchema>("/api/v1/health/input-schema");
  return response.data;
}

/** Upload a report and preview the values extracted from it. */
export async function previewHealthReport(
  file: File,
  useLLM: boolean = false,
): Promise<ExtractionPreview> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await http.post<ExtractionPreview>(
    "/api/v1/health/upload-report/preview",
    formData,
    {
      params: { use_llm: useLLM },
      headers: { "Content-Type": "multipart/form-data" },
    },
  );

  return response.data;
}

/** Confirm (and optionally correct) extracted values, then assess them. */
export async function confirmExtraction(
  payload: ExtractionConfirmRequest,
): Promise<HealthDataSubmitResponse> {
  const response = await http.post<HealthDataSubmitResponse>(
    "/api/v1/health/upload-report/confirm",
    payload,
  );
  return response.data;
}

/** The user's assessments, newest first. */
export async function getAssessmentHistory(
  skip: number = 0,
  limit: number = 50,
  includePending: boolean = false,
): Promise<AssessmentHistoryResponse> {
  const response = await http.get<AssessmentHistoryResponse>(
    "/api/v1/health/history",
    { params: { skip, limit, include_pending: includePending } },
  );
  return response.data;
}

/** Parameter and score time series for charting. */
export async function getAssessmentHistorySummary(
  limit: number = 100,
): Promise<AssessmentHistorySummary> {
  const response = await http.get<AssessmentHistorySummary>(
    "/api/v1/health/history/summary",
    { params: { limit } },
  );
  return response.data;
}
