import { useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { PdfUploadPanel, ReportUploadPayload } from "../components/analyzer/PdfUploadPanel";
import { ExtractionReview } from "../components/analyzer/ExtractionReview";
import { ManualEntryForm } from "../components/analyzer/ManualEntryForm";
import { ResultsOverview } from "../components/results/ResultsOverview";
import {
  useConfirmExtraction,
  useInputSchema,
  usePreviewReport,
  useSubmitHealthData,
} from "../api/hooks";
import {
  AnalysisResult,
  ExtractionPreview,
  HealthInput,
  HealthRecordResponse,
} from "../models/health";
import { SectionHeader } from "../components/shared/SectionHeader";
import { SectionCard } from "../components/shared/SectionCard";

/** The values a submission carries, as the results screen shows them. */
function toRecord(
  values: Partial<HealthInput>,
  recordId: number,
  createdAt: string,
): Partial<HealthRecordResponse> {
  return {
    id: recordId,
    user_id: 0,
    hba1c: values.hba1c ?? null,
    glucose: values.glucose ?? null,
    bmi: values.bmi ?? null,
    age: values.age ?? null,
    systolic_bp: values.systolic_bp ?? null,
    diastolic_bp: values.diastolic_bp ?? null,
    total_cholesterol: values.total_cholesterol ?? null,
    ldl: values.ldl ?? null,
    hdl: values.hdl ?? null,
    smoker: values.smoker ?? null,
    chest_pain_type: values.chest_pain_type ?? null,
    st_slope: values.st_slope ?? null,
    created_at: createdAt,
    analyzed: true,
    analysis_result: null,
  };
}

export function AnalyzerPage() {
  const [params] = useSearchParams();
  const [analysis, setAnalysis] = useState<AnalysisResult | undefined>();
  const [record, setRecord] = useState<Partial<HealthRecordResponse> | undefined>();
  const [preview, setPreview] = useState<ExtractionPreview | null>(null);
  const [uploadContext, setUploadContext] = useState({
    symptoms: [] as string[],
    manualText: "",
    useLLM: false,
  });
  const tab = params.get("tab") || "pdf";

  const schema = useInputSchema();
  const previewMutation = usePreviewReport();
  const confirmMutation = useConfirmExtraction();
  const manualMutation = useSubmitHealthData();
  const isLoading =
    previewMutation.isPending || confirmMutation.isPending || manualMutation.isPending;
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
    setPreview(null);
  }, [tab]);

  const resetResults = () => {
    setAnalysis(undefined);
    setRecord(undefined);
  };

  const handlePreview = async (payload: ReportUploadPayload) => {
    resetResults();
    setUploadContext({
      symptoms: payload.symptoms,
      manualText: payload.manualText,
      useLLM: payload.useLLM,
    });
    try {
      const response = await previewMutation.mutateAsync({
        file: payload.file,
        useLLM: payload.useLLM,
      });
      setPreview(response);
    } catch {
      // The failure is reported from the mutation's error state.
    }
  };

  const handleConfirm = async (values: Record<string, number | boolean>) => {
    if (!preview) {
      return;
    }
    try {
      const response = await confirmMutation.mutateAsync({
        extraction_id: preview.extraction_id,
        ...values,
        symptoms: uploadContext.symptoms,
        manual_text: uploadContext.manualText,
        use_llm: uploadContext.useLLM,
      });
      setAnalysis({
        ...(response.final_assessment ?? {}),
        unified: response.unified ?? null,
        risk_level: response.risk_level ?? response.final_assessment?.risk_level ?? null,
        alert: response.alert ?? null,
        report: response.report ?? null,
        workflow_status: response.workflow_status ?? null,
      });
      setRecord(
        toRecord(
          values as Partial<HealthInput>,
          response.record_id,
          response.timestamp,
        ),
      );
      setPreview(null);
    } catch {
      // The failure is reported from the mutation's error state.
    }
  };

  const handleManualSubmit = async (payload: HealthInput) => {
    resetResults();
    try {
      const response = await manualMutation.mutateAsync(payload);
      setAnalysis({
        ...(response.final_assessment ?? {}),
        unified: response.unified ?? null,
        risk_level: response.risk_level ?? response.final_assessment?.risk_level ?? null,
        alert: response.alert ?? null,
        report: response.report ?? null,
        workflow_status: response.workflow_status ?? null,
      });
      setRecord(toRecord(payload, response.record_id, response.timestamp));
    } catch {
      // The failure is reported from the mutation's error state.
    }
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
            onSubmit={handleManualSubmit}
            isLoading={manualMutation.isPending}
            schema={schema.data}
            schemaError={
              schema.isError
                ? schema.error?.message ?? "The parameters this API accepts could not be loaded."
                : null
            }
            schemaLoading={schema.isLoading}
          />
        ) : (
          <PdfUploadPanel
            onPreview={handlePreview}
            isLoading={previewMutation.isPending}
            error={
              previewMutation.isError
                ? previewMutation.error?.message ?? "The report could not be read."
                : null
            }
          />
        )}
      </div>

      {preview && (
        <div className="max-w-3xl">
          <ExtractionReview
            key={preview.extraction_id}
            preview={preview}
            schema={schema.data}
            onConfirm={handleConfirm}
            onCancel={() => setPreview(null)}
            isLoading={confirmMutation.isPending}
          />
        </div>
      )}

      {isLoading && (
        <div ref={loaderRef} className="max-w-3xl">
          <SectionCard title={confirmMutation.isPending ? "Assessing the confirmed values" : "Working"}>
            <div className="flex items-center gap-3 text-sm text-slate-600">
              <span className="h-5 w-5 animate-spin rounded-full border-2 border-emerald-500 border-t-transparent" />
              {confirmMutation.isPending
                ? "The confirmed values are being assessed. This may take a moment."
                : previewMutation.isPending
                  ? "Your health report is being read. This may take a moment."
                  : "Your health data is being processed. This may take a moment."}
            </div>
          </SectionCard>
        </div>
      )}

      {(previewMutation.isError || confirmMutation.isError || manualMutation.isError) && (
        <SectionCard title="Analysis failed" className="border-red-200 bg-red-50">
          <p className="text-sm text-red-700">
            {(
              confirmMutation.error ??
              previewMutation.error ??
              manualMutation.error
            )?.message ?? "The report could not be analyzed."}
          </p>
        </SectionCard>
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
