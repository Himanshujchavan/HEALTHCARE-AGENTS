import { AnalysisResult, HealthRecordResponse } from "../../models/health";
import { RiskBadge } from "../shared/RiskBadge";
import { SectionCard } from "../shared/SectionCard";

interface ResultsOverviewProps {
  analysis?: AnalysisResult;
  record?: Partial<HealthRecordResponse>;
  showSymptomSection?: boolean;
}

export function ResultsOverview({ analysis, record, showSymptomSection = true }: ResultsOverviewProps) {
  const riskLevel = (analysis?.risk_level as string) || "Unknown";
  const alert = analysis?.alert as boolean | undefined;
  const summary = (analysis?.summary as string) || "Summary not available yet.";
  const report = (analysis?.report as string) || "";
  const error = (analysis?.error as string) || "";
  const riskLabel = String(riskLevel || "Unknown");
  const scoreValue = analysis?.score ?? analysis?.final_assessment?.score;
  const normalizedScore =
    typeof scoreValue === "number"
      ? scoreValue > 1
        ? scoreValue / 100
        : scoreValue
      : null;
  const scoreText =
    typeof normalizedScore === "number"
      ? `${Math.round(normalizedScore * 100)}%`
      : scoreValue
      ? String(scoreValue)
      : null;

  return (
    <div className="space-y-6">
      <div className="grid gap-6 lg:grid-cols-[1.2fr_0.8fr]">
        <SectionCard title="Risk level">
          <div>
            <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Assessment</p>
            <h3 className="mt-2 text-2xl font-semibold text-slate-900">
              {riskLevel} risk
            </h3>
            <p className="mt-2 text-sm text-slate-600">{summary}</p>
            {scoreText && (
              <p className="mt-2 text-sm font-semibold text-slate-700">
                Score: {scoreText}
              </p>
            )}
            {error && <p className="mt-2 text-sm text-red-600">{error}</p>}
            <div className="mt-4">
              <RiskBadge level={riskLevel} />
            </div>
          </div>
        </SectionCard>

        <SectionCard title="Alert status">
          <div className="space-y-3">
            <p className="text-sm text-slate-600">
              {alert === undefined
                ? "Alert status will update once the analysis completes."
                : alert
                ? "Immediate follow-up recommended."
                : "No immediate alerts detected."}
            </p>
            {alert && (
              <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                Contact a healthcare professional if symptoms worsen.
              </div>
            )}
          </div>
        </SectionCard>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <SectionCard title="HbA1c">
          <p className="text-3xl font-semibold text-slate-900">{record?.hba1c ?? "--"}</p>
          <p className="text-xs text-slate-500">Latest lab value</p>
        </SectionCard>
        <SectionCard title="Glucose">
          <p className="text-3xl font-semibold text-slate-900">{record?.glucose ?? "--"}</p>
          <p className="text-xs text-slate-500">mg/dL</p>
        </SectionCard>
        <SectionCard title="BMI">
          <p className="text-3xl font-semibold text-slate-900">{record?.bmi ?? "--"}</p>
          <p className="text-xs text-slate-500">Body mass index</p>
        </SectionCard>
      </div>

      <SectionCard title="AI generated report">
        <div className="space-y-4 text-sm text-slate-700">
          <p>{report || "Your AI-generated preventive report will appear here."}</p>
        </div>
      </SectionCard>

      {showSymptomSection && (
        <SectionCard title="Symptom analysis">
          <p className="text-sm text-slate-600">
            {analysis?.symptom_summary
              ? (analysis.symptom_summary as string)
              : "Symptom alignment will appear once the analysis completes."}
          </p>
        </SectionCard>
      )}

      <SectionCard title="Report actions">
        <div className="flex flex-wrap gap-3">
          <button className="rounded-full border border-slate-200 px-4 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-100">
            Download PDF
          </button>
          <button className="rounded-full border border-slate-200 px-4 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-100">
            Save analysis
          </button>
          <button className="rounded-full bg-emerald-600 px-4 py-2 text-sm font-semibold text-white hover:bg-emerald-700">
            Share with doctor
          </button>
        </div>
      </SectionCard>
    </div>
  );
}
