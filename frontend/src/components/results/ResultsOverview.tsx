import { AnalysisResult, HealthRecordResponse, UnifiedAssessment } from "../../models/health";
import { RiskBadge } from "../shared/RiskBadge";
import { SectionCard } from "../shared/SectionCard";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

interface ResultsOverviewProps {
  analysis?: AnalysisResult;
  record?: Partial<HealthRecordResponse>;
  showSymptomSection?: boolean;
}

function valueOrMissing(value: unknown): string {
  return value === null || value === undefined || value === "" ? "Not available" : String(value);
}

function DiseaseCard({ name, assessment }: {
  name: string;
  assessment?: UnifiedAssessment["assessments"]["diabetes"];
}) {
  const unavailable = !assessment || assessment.status === "unavailable";
  return (
    <SectionCard title={`${name} risk assessment`}>
      {unavailable ? (
        <p className="text-sm text-slate-600">Not enough data to assess this condition.</p>
      ) : (
        <div className="space-y-3">
          <div className="flex items-center justify-between gap-3">
            <div>
              <p className="text-2xl font-semibold text-slate-900">{assessment.risk_level}</p>
              <p className="text-xs text-slate-500">
                {assessment.score === null || assessment.score === undefined
                  ? "Score not available"
                  : `Model score: ${assessment.score}`}
              </p>
            </div>
            <RiskBadge level={assessment.risk_level} />
          </div>
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Factors</p>
            {assessment.factors.length > 0 ? (
              <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-700">
                {assessment.factors.map((factor) => <li key={factor}>{factor}</li>)}
              </ul>
            ) : (
              <p className="mt-2 text-sm text-slate-600">No contributing factors were returned.</p>
            )}
          </div>
        </div>
      )}
    </SectionCard>
  );
}

export function ResultsOverview({ analysis, record, showSymptomSection = true }: ResultsOverviewProps) {
  const unified = analysis?.unified as unknown as UnifiedAssessment | undefined;
  const legacyRisk = String(analysis?.risk_level ?? "Unavailable");
  const trend = unified?.trend;
  const trendChanges = (trend?.changes ?? {}) as Record<string, { direction?: string; delta?: number }>;
  const alert = unified?.alerts?.[0];
  const explanation = typeof unified?.explanation === "string"
    ? unified.explanation
    : String(analysis?.explanation ?? analysis?.report ?? "Explanation is not available.");
  const trendScore = typeof trend?.trend_score === "number" ? trend.trend_score : 0;

  return (
    <div className="space-y-6">
      <div className="rounded-2xl border border-amber-200 bg-amber-50 px-5 py-4 text-sm text-amber-900">
        {unified?.model_disclaimer ?? "Risk predictions support clinical review and are not confirmed medical diagnoses."}
      </div>

      {unified ? (
        <div className="grid gap-6 lg:grid-cols-2">
          <DiseaseCard name="Diabetes" assessment={unified.assessments.diabetes} />
          <DiseaseCard name="Cardiovascular" assessment={unified.assessments.cardiovascular} />
        </div>
      ) : (
        <SectionCard title="Risk assessment">
          <div className="flex items-center justify-between">
            <p className="text-2xl font-semibold text-slate-900">{legacyRisk}</p>
            <RiskBadge level={legacyRisk} />
          </div>
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
            <p className="text-sm text-slate-600">No valid differential result was returned.</p>
          )}
        </SectionCard>
      </div>

      <SectionCard title="Explanation">
        <p className="text-sm leading-6 text-slate-700">{explanation}</p>
      </SectionCard>

      {showSymptomSection && (
        <SectionCard title="Input status">
          <p className="text-sm text-slate-600">
            Analysis date: {record?.created_at ? new Date(record.created_at).toLocaleString() : "Not available"}
            {" · "}Workflow: {String(unified?.status ?? analysis?.workflow_status ?? "Not available")}
          </p>
        </SectionCard>
      )}
    </div>
  );
}
