import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { UserCreate, UserLogin } from "../models/auth";
import {
  HealthInput,
  HealthRecordResponse,
  HealthRecordAnalysisResponse,
  HealthDataSubmitResponse,
  LatestRecordResponse,
  UploadReportResponse,
} from "../models/health";
import * as authApi from "./auth";
import * as healthApi from "./health";

export const queryKeys = {
  latest: ["health", "latest"] as const,
  records: (skip: number, limit: number) =>
    ["health", "records", skip, limit] as const,
  record: (recordId: number) => ["health", "record", recordId] as const,
  analysis: (recordId: number) => ["health", "analysis", recordId] as const,
};

export const useRegister = () =>
  useMutation({ mutationFn: (payload: UserCreate) => authApi.registerUser(payload) });

export const useLogin = () =>
  useMutation({ mutationFn: (payload: UserLogin) => authApi.loginUser(payload) });

export const useLogout = () => useMutation({ mutationFn: authApi.logoutUser });

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

export const useUploadReport = () => {
  const client = useQueryClient();
  return useMutation<
    UploadReportResponse,
    Error,
    { file: File; useLLM?: boolean; symptoms?: string[]; manualText?: string }
  >({
    mutationFn: ({ file, useLLM, symptoms, manualText }) =>
      healthApi.uploadHealthReport(file, useLLM, symptoms, manualText ?? ""),
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
