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
} from "../models/health";

export async function uploadHealthReport(
  file: File,
  useLLM: boolean = false,
  symptoms: string[] = [],
  manualText: string = "",
): Promise<UploadReportResponse> {
  const formData = new FormData();
  formData.append("file", file);
  symptoms.forEach((symptom) => formData.append("symptoms", symptom));
  if (manualText.trim()) {
    formData.append("manual_text", manualText.trim());
  }

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
