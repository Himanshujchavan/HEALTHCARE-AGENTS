import { AnalysisResult, DiseaseAssessment, HealthRecordResponse, UnifiedAssessment } from "../../models/health";
import { RiskBadge } from "../shared/RiskBadge";
import { SectionCard } from "../shared/SectionCard";
import { StatusBadge, statusLabel } from "../shared/StatusBadge";
import { AgentStatusPanel } from "../dashboard/AgentStatus";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

interface ResultsOverviewProps {
  analysis?: AnalysisResult;
  record?: Partial<HealthRecordResponse>;
  showSymptomSection?: boolean;
}

function valueOrMissing(value: unknown): string {
  return value === null || value === undefined || value === "" ? "Not available" : String(value);
}

const DISEASE_LABELS: Record<string, string> = {
  diabetes: "Diabetes",
  cardiovascular: "Cardiovascular",
};

function diseaseLabel(name: string): string {
  return (
    DISEASE_LABELS[name] ??
    name.charAt(0).toUpperCase() + name.slice(1)
  );
}

/**
 * A single condition's assessment.
 *
 * The four states are kept apart on purpose: a completed assessment, one
 * completed with errors, one that failed, and one that was never applicable
 * mean very different things, and a screen that shows them all as
 * "not available" hides the difference.
 */
function DiseaseCard({ name, assessment }: {
  name: string;
  assessment?: UnifiedAssessment["assessments"]["diabetes"];
}) {
  const status = assessment?.status ?? "unavailable";
  const reason = assessment?.reason ?? null;

  const stateNote = (() => {
    switch (status) {
      case "completed":
        return null;
      case "completed_with_errors":
        return "The assessment completed, but some of its components could not run, so the score rests on less than the full formula.";
      case "partial":
        return "Only part of the assessment could be computed, so the score is a lower bound rather than a complete assessment.";
      case "failed":
        return "The assessment for this condition failed. No score was produced.";
      case "not_applicable":
        return "No required values were submitted for this condition, so it was not assessed.";
      case "unavailable":
      default:
        return "Not enough data to assess this condition.";
    }
  })();

  const hasScore = status === "completed" || status === "completed_with_errors" || status === "partial";

  return (
    <SectionCard title={`${diseaseLabel(name)} risk assessment`}>
      <div className="space-y-3">
        <div className="flex items-center justify-between gap-3">
          <div>
            <p className="text-2xl font-semibold text-slate-900">
              {hasScore ? assessment?.risk_level : statusLabel(status)}
            </p>
            <p className="text-xs text-slate-500">
              {hasScore
                ? assessment?.score === null || assessment?.score === undefined
                  ? "Score not available"
                  : `Model score: ${assessment?.score}`
                : "No score was produced"}
            </p>
          </div>
          {hasScore ? (
            <RiskBadge level={assessment?.risk_level ?? "Unknown"} />
          ) : (
            <StatusBadge status={status} />
          )}
        </div>

        {stateNote && (
          <p
            className={`rounded-xl border px-3 py-2 text-xs ${
              status === "failed"
                ? "border-red-200 bg-red-50 text-red-800"
                : status === "completed_with_errors" || status === "partial"
                  ? "border-amber-200 bg-amber-50 text-amber-900"
                  : "border-slate-200 bg-slate-50 text-slate-600"
            }`}
          >
            {stateNote}
          </p>
        )}

        {reason && <p className="text-xs text-slate-500">{reason}</p>}

        {hasScore && (
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Factors</p>
            {assessment?.factors?.length ? (
              <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-700">
                {assessment.factors.map((factor) => <li key={factor}>{factor}</li>)}
              </ul>
            ) : (
              <p className="mt-2 text-sm text-slate-600">No contributing factors were returned.</p>
            )}
          </div>
        )}

        {(assessment?.model_confidence || assessment?.model_variant) && (
          <p className="text-xs text-slate-500">
            {assessment?.model_variant && <>Model variant: {assessment.model_variant}. </>}
            {assessment?.model_confidence && <>Confidence: {assessment.model_confidence}.</>}
          </p>
        )}
      </div>
    </SectionCard>
  );
}

/** The trend payload, as the results screen reads it. */
interface TrendView {
  status?: string;
  trend_score?: number;
  direction?: string;
  visits_used?: number;
  changes?: Record<string, { direction?: string; delta?: number }>;
}

export function ResultsOverview({ analysis, record, showSymptomSection = true }: ResultsOverviewProps) {
  const unified = analysis?.unified as unknown as UnifiedAssessment | undefined;
  const legacyRisk = String(analysis?.risk_level ?? "Unavailable");
  const trend = unified?.trend as unknown as TrendView | undefined;
  const trendChanges = trend?.changes ?? {};
  const alert = unified?.alerts?.[0];
  const explanation = typeof unified?.explanation === "string"
    ? unified.explanation
    : String(analysis?.explanation ?? analysis?.report ?? "Explanation is not available.");
  const trendScore = typeof trend?.trend_score === "number" ? trend.trend_score : 0;
  const assessments: Record<string, DiseaseAssessment> = unified?.assessments ?? {};

  return (
    <div className="space-y-6">
      <div className="rounded-2xl border border-amber-200 bg-amber-50 px-5 py-4 text-sm text-amber-900">
        {unified?.model_disclaimer ?? "Risk predictions support clinical review and are not confirmed medical diagnoses."}
      </div>

      {unified ? (
        <div className="grid gap-6 lg:grid-cols-2">
          {Object.entries(assessments).map(([name, assessment]) => (
            <DiseaseCard key={name} name={name} assessment={assessment} />
          ))}
        </div>
      ) : (
        <SectionCard title="Risk assessment">
          <div className="flex items-center justify-between">
            <p className="text-2xl font-semibold text-slate-900">{legacyRisk}</p>
            <RiskBadge level={legacyRisk} />
          </div>
        </SectionCard>
      )}

      {unified?.agent_status && Object.keys(unified.agent_status).length > 0 && (
        <SectionCard title="Agent status">
          <AgentStatusPanel agentStatus={unified.agent_status} />
        </SectionCard>
      )}

      <div className="grid gap-6 md:grid-cols-3">
        {[
          ["HbA1c", record?.hba1c, "%"],
          ["Glucose", record?.glucose, "mg/dL"],
          ["Blood pressure", record?.systolic_bp, "mmHg systolic"],
          ["Cholesterol", record?.total_cholesterol, "mg/dL"],
          ["Age", record?.age, "years"],
          ["BMI", record?.bmi, "kg/m2"],
        ].map(([label, value, unit]) => (
          <SectionCard key={String(label)} title={String(label)}>
            <p className="text-2xl font-semibold text-slate-900">{valueOrMissing(value)}</p>
            <p className="text-xs text-slate-500">{String(unit)}</p>
          </SectionCard>
        ))}
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <SectionCard title="Trend comparison">
          <p className="mb-3 text-sm text-slate-600">
            {trend?.status === "insufficient_history"
              ? "No previous assessment is available for comparison."
              : `${String(trend?.direction ?? "stable")} across ${String(trend?.visits_used ?? 0)} assessments`}
          </p>
          <ResponsiveContainer width="100%" height={100}>
            <LineChart data={[{ name: "trend", value: trendScore }]}>
              <XAxis dataKey="name" hide />
              <YAxis domain={[0, 1]} hide />
              <Tooltip />
              <Line type="monotone" dataKey="value" stroke="#059669" strokeWidth={3} dot />
            </LineChart>
          </ResponsiveContainer>
          <div className="mt-3 space-y-1 text-xs text-slate-600">
            {Object.entries(trendChanges).map(([metric, change]) => (
              <p key={metric}>{metric}: {change.direction ?? "stable"} ({change.delta ?? 0})</p>
            ))}
          </div>
        </SectionCard>

        <SectionCard title="Alert and next steps">
          <p className="text-sm font-semibold text-slate-900">
            {alert ? `${alert.severity} · ${alert.alert ? "Attention recommended" : "No immediate alert"}` : "No alert result"}
          </p>
          <p className="mt-2 text-sm text-slate-600">{alert?.explanation ?? "Alert details are not available."}</p>
          {alert?.recommended_next_steps && (
            <p className="mt-3 text-sm text-slate-700">{alert.recommended_next_steps}</p>
          )}
        </SectionCard>

        <SectionCard title="Differential agent">
          {unified?.differential.status === "completed" ? (
            <pre className="max-h-40 overflow-auto whitespace-pre-wrap text-xs text-slate-600">
              {JSON.stringify(unified.differential.results, null, 2)}
            </pre>
          ) : (
            <p className="text-sm text-slate-600">
              {unified?.differential.status
                ? `The differential agent did not complete (${statusLabel(unified.differential.status)}).`
                : "No valid differential result was returned."}
            </p>
          )}
        </SectionCard>
      </div>

      <SectionCard title="Explanation">
        <p className="text-sm leading-6 text-slate-700">{explanation}</p>
      </SectionCard>

      {showSymptomSection && (
        <SectionCard title="Input status">
          <p className="flex flex-wrap items-center gap-2 text-sm text-slate-600">
            <span>
              Analysis date: {record?.created_at ? new Date(record.created_at).toLocaleString() : "Not available"}
            </span>
            <span>·</span>
            <span>
              Workflow: {String(unified?.status ?? analysis?.workflow_status ?? "Not available")}
            </span>
            <StatusBadge status={String(unified?.status ?? analysis?.workflow_status ?? "unknown")} />
          </p>
        </SectionCard>
      )}
    </div>
  );
}
