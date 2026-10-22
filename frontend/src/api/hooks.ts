import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { UserCreate, UserLogin } from "../models/auth";
import {
  HealthInput,
  HealthRecordResponse,
  HealthRecordAnalysisResponse,
  HealthDataSubmitResponse,
  LatestRecordResponse,
  AssessmentHistoryResponse,
  AssessmentHistorySummary,
  ExtractionPreview,
  ExtractionConfirmRequest,
  InputSchema,
} from "../models/health";
import * as authApi from "./auth";
import * as healthApi from "./health";

export const queryKeys = {
  latest: ["health", "latest"] as const,
  records: (skip: number, limit: number) =>
    ["health", "records", skip, limit] as const,
  record: (recordId: number) => ["health", "record", recordId] as const,
  analysis: (recordId: number) => ["health", "analysis", recordId] as const,
  history: (skip: number, limit: number, includePending: boolean) =>
    ["health", "history", skip, limit, includePending] as const,
  historySummary: (limit: number) =>
    ["health", "history", "summary", limit] as const,
  inputSchema: ["health", "input-schema"] as const,
};

export const useRegister = () =>
  useMutation({ mutationFn: (payload: UserCreate) => authApi.registerUser(payload) });

export const useLogin = () =>
  useMutation({ mutationFn: (payload: UserLogin) => authApi.loginUser(payload) });

export const useLogout = () =>
  useMutation({
    mutationFn: async () => {
      await authApi.logoutUser();
    },
  });

export const useLatestRecord = () =>
  useQuery<LatestRecordResponse>({
    queryKey: queryKeys.latest,
    queryFn: () => healthApi.getLatestRecord(),
  });

export const useHealthRecords = (skip = 0, limit = 100) =>
  useQuery<HealthRecordResponse[]>({
    queryKey: queryKeys.records(skip, limit),
    queryFn: () => healthApi.getHealthRecords(skip, limit),
  });

export const useHealthRecord = (recordId: number) =>
  useQuery<HealthRecordResponse>({
    queryKey: queryKeys.record(recordId),
    queryFn: () => healthApi.getHealthRecord(recordId),
    enabled: Number.isFinite(recordId),
  });

export const useHealthAnalysis = (recordId: number) =>
  useQuery<HealthRecordAnalysisResponse>({
    queryKey: queryKeys.analysis(recordId),
    queryFn: () => healthApi.getHealthRecordAnalysis(recordId),
    enabled: Number.isFinite(recordId),
  });

export const useSubmitHealthData = () => {
  const client = useQueryClient();
  return useMutation<HealthDataSubmitResponse, Error, HealthInput>({
    mutationFn: (payload) => healthApi.submitHealthData(payload),
    onSuccess: () => {
      client.invalidateQueries({ queryKey: queryKeys.latest });
      client.invalidateQueries({ queryKey: ["health", "records"] });
    },
  });
};

export const useReanalyzeRecord = () => {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (recordId: number) => healthApi.reanalyzeRecord(recordId),
    onSuccess: () => {
      client.invalidateQueries({ queryKey: ["health", "records"] });
      // The history and its summary both change when a record is re-analysed.
      client.invalidateQueries({ queryKey: ["health", "history"] });
    },
  });
};

export const useDeleteHealthRecord = () => {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (recordId: number) => healthApi.deleteHealthRecord(recordId),
    onSuccess: () => {
      client.invalidateQueries({ queryKey: ["health", "records"] });
      client.invalidateQueries({ queryKey: queryKeys.latest });
    },
  });
};

export const useAssessmentHistory = (
  skip = 0,
  limit = 50,
  includePending = false,
) =>
  useQuery<AssessmentHistoryResponse>({
    queryKey: queryKeys.history(skip, limit, includePending),
    queryFn: () => healthApi.getAssessmentHistory(skip, limit, includePending),
  });

export const useAssessmentHistorySummary = (limit = 100) =>
  useQuery<AssessmentHistorySummary>({
    queryKey: queryKeys.historySummary(limit),
    queryFn: () => healthApi.getAssessmentHistorySummary(limit),
  });

export const useInputSchema = () =>
  useQuery<InputSchema>({
    queryKey: queryKeys.inputSchema,
    queryFn: () => healthApi.getInputSchema(),
    staleTime: 5 * 60 * 1000,
  });

export const usePreviewReport = () => {
  const client = useQueryClient();
  return useMutation<ExtractionPreview, Error, { file: File; useLLM?: boolean }>({
    mutationFn: ({ file, useLLM }) => healthApi.previewHealthReport(file, useLLM),
    onSuccess: () => {
      client.invalidateQueries({ queryKey: ["health", "history"] });
    },
  });
};

export const useConfirmExtraction = () => {
  const client = useQueryClient();
  return useMutation<HealthDataSubmitResponse, Error, ExtractionConfirmRequest>({
    mutationFn: (payload) => healthApi.confirmExtraction(payload),
    onSuccess: () => {
      client.invalidateQueries({ queryKey: queryKeys.latest });
      client.invalidateQueries({ queryKey: ["health", "records"] });
      client.invalidateQueries({ queryKey: ["health", "history"] });
    },
  });
};
