import { useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { PdfUploadPanel } from "../components/analyzer/PdfUploadPanel";
import { ManualEntryForm } from "../components/analyzer/ManualEntryForm";
import { ResultsOverview } from "../components/results/ResultsOverview";
import { useSubmitHealthData, useUploadReport } from "../api/hooks";
import { AnalysisResult, HealthRecordResponse } from "../models/health";
import { SectionHeader } from "../components/shared/SectionHeader";
import { SectionCard } from "../components/shared/SectionCard";

export function AnalyzerPage() {
  const [params] = useSearchParams();
  const [analysis, setAnalysis] = useState<AnalysisResult | undefined>();
  const [record, setRecord] = useState<Partial<HealthRecordResponse> | undefined>();
  const tab = params.get("tab") || "pdf";

  const uploadMutation = useUploadReport();
  const manualMutation = useSubmitHealthData();
  const isLoading = uploadMutation.isPending || manualMutation.isPending;
  const loaderRef = useRef<HTMLDivElement | null>(null);
  const resultsRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (isLoading && loaderRef.current) {
      loaderRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [isLoading]);

  useEffect(() => {
    if (analysis && resultsRef.current) {
      resultsRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [analysis]);

  useEffect(() => {
    setAnalysis(undefined);
    setRecord(undefined);
  }, [tab]);

  const resetResults = () => {
    setAnalysis(undefined);
    setRecord(undefined);
  };

  return (
    <div className="space-y-6">
      <SectionHeader
        eyebrow="Health Analyzer"
        title="Run a new health analysis"
        subtitle="Upload a report or enter key metrics to activate the AI workflow."
      />

      <div className="max-w-3xl">
        {tab === "manual" ? (
          <ManualEntryForm
            onSubmit={async (payload) => {
              resetResults();
              const response = await manualMutation.mutateAsync(payload);
              setAnalysis({
                ...(response.final_assessment ?? {}),
                risk_level: response.risk_level ?? response.final_assessment?.risk_level,
                alert: response.alert ?? null,
                report: response.report ?? null,
              });
              setRecord({
                id: response.record_id,
                user_id: 0,
                hba1c: payload.hba1c,
                glucose: payload.glucose,
                bmi: payload.bmi,
                age: payload.age,
                created_at: response.timestamp,
                analyzed: true,
                analysis_result: null,
              });
            }}
            isLoading={manualMutation.isPending}
          />
        ) : (
          <PdfUploadPanel
            onSubmit={async (file, useLLM, symptoms, manualText) => {
              resetResults();
              const response = await uploadMutation.mutateAsync({
                file,
                useLLM,
                symptoms,
                manualText,
              });
              const extractedParams = (response.analysis?.parameters ?? {}) as Record<
                string,
                { value?: number }
              >;
              setAnalysis({
                ...(response.analysis ?? {}),
                ...(response.final_assessment ?? {}),
                ...(response.alert_result ?? {}),
                risk_level:
                  response.risk_level ??
                  response.final_assessment?.risk_level ??
                  response.alert_result?.risk_level,
                alert: response.alert ?? response.alert_result?.alert ?? null,
                report: response.report ?? response.alert_result?.report ?? null,
                workflow_status: response.workflow_status,
                error: response.error,
              });
              setRecord({
                id: 0,
                user_id: 0,
                hba1c: extractedParams.hba1c?.value,
                glucose: extractedParams.glucose?.value,
                bmi: extractedParams.bmi?.value,
                age: extractedParams.age?.value,
                created_at: response.timestamp ?? new Date().toISOString(),
                analyzed: true,
                analysis_result: null,
              });
            }}
            isLoading={uploadMutation.isPending}
          />
        )}
      </div>

      {isLoading && (
        <div ref={loaderRef} className="max-w-3xl">
          <SectionCard title="Analyzing report">
            <div className="flex items-center gap-3 text-sm text-slate-600">
              <span className="h-5 w-5 animate-spin rounded-full border-2 border-emerald-500 border-t-transparent" />
              Your health report is being processed. This may take a moment.
            </div>
          </SectionCard>
        </div>
      )}

      {analysis && (
        <div ref={resultsRef}>
          <ResultsOverview
            analysis={analysis}
            record={record}
            showSymptomSection={false}
          />
        </div>
      )}
    </div>
  );
}
