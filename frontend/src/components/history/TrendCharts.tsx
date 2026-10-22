import { useMemo, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { AssessmentHistorySummary, AssessmentSeriesPoint } from "../../models/health";
import { SectionCard } from "../shared/SectionCard";

interface TrendChartsProps {
  summary?: AssessmentHistorySummary;
  isLoading?: boolean;
  error?: string | null;
}

function formatDate(value?: string | null): string {
  if (!value) {
    return "";
  }
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleDateString();
}

function toChartPoints(points: AssessmentSeriesPoint[]) {
  return points.map((point) => ({
    ...point,
    date: formatDate(point.created_at),
    value: point.value ?? null,
  }));
}

/**
 * The history charts, drawn from `/history/summary`.
 *
 * The endpoint returns one time series per parameter and one for the
 * assessment score, so the charts cover both the measurements and the
 * assessments they produced.
 */
export function TrendCharts({ summary, isLoading, error }: TrendChartsProps) {
  const parameterNames = useMemo(
    () => Object.keys(summary?.parameters ?? {}).sort(),
    [summary?.parameters],
  );
  const [selected, setSelected] = useState<string>("");

  const activeParameter = selected && parameterNames.includes(selected)
    ? selected
    : parameterNames[0] ?? "";

  const scorePoints = useMemo(
    () => toChartPoints(summary?.scores ?? []),
    [summary?.scores],
  );
  const parameterPoints = useMemo(
    () => toChartPoints(summary?.parameters?.[activeParameter] ?? []),
    [summary?.parameters, activeParameter],
  );

  return (
    <div className="space-y-6">
      <SectionCard title="Assessment score over time">
        {isLoading && <p className="text-sm text-slate-600">Loading the history summary…</p>}
        {error && <p className="text-sm text-red-700">{error}</p>}
        {!isLoading && !error && scorePoints.length === 0 && (
          <p className="text-sm text-slate-600">
            No assessment scores are available yet.
          </p>
        )}
        {scorePoints.length > 0 && (
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={scorePoints}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="date" tick={{ fontSize: 12 }} />
              <YAxis domain={[0, 1]} tick={{ fontSize: 12 }} />
              <Tooltip
                formatter={(value: unknown) =>
                  value === null || value === undefined ? "No score" : String(value)
                }
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="value"
                name="Assessment score"
                stroke="#059669"
                strokeWidth={2.5}
                dot
                connectNulls={false}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </SectionCard>

      <SectionCard title="Measurements over time">
        {parameterNames.length === 0 ? (
          <p className="text-sm text-slate-600">
            No measurements have been recorded yet.
          </p>
        ) : (
          <>
            <label className="mb-4 block space-y-2 text-sm text-slate-600">
              Parameter
              <select
                data-testid="trend-parameter-select"
                className="rounded-xl border border-slate-200 px-4 py-2 text-sm"
                value={activeParameter}
                onChange={(event) => setSelected(event.target.value)}
              >
                {parameterNames.map((name) => (
                  <option key={name} value={name}>
                    {name.replace(/_/g, " ")}
                  </option>
                ))}
              </select>
            </label>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={parameterPoints}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="date" tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="value"
                  name={activeParameter.replace(/_/g, " ")}
                  stroke="#2563eb"
                  strokeWidth={2.5}
                  dot
                  connectNulls={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </>
        )}
      </SectionCard>

      {summary?.risk_levels && Object.keys(summary.risk_levels).length > 0 && (
        <SectionCard title="Risk levels recorded">
          <div className="flex flex-wrap gap-2">
            {Object.entries(summary.risk_levels).map(([level, count]) => (
              <span
                key={level}
                className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs text-slate-700"
              >
                <span className="font-semibold">{level}</span>
                <span>{count}</span>
              </span>
            ))}
          </div>
        </SectionCard>
      )}
    </div>
  );
}
