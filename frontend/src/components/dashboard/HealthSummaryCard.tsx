import { LatestRecordResponse } from "../../models/health";
import { RiskBadge } from "../shared/RiskBadge";

interface HealthSummaryCardProps {
  latest?: LatestRecordResponse;
}

export function HealthSummaryCard({ latest }: HealthSummaryCardProps) {
  const riskLevel = (latest?.analysis?.risk_level as string) || "Unknown";
  const alert = latest?.analysis?.alert as boolean | undefined;
  const lastDate = latest?.record?.created_at
    ? new Date(latest.record.created_at).toLocaleDateString()
    : "No reports";

  return (
    <div className="rounded-2xl border border-slate-200 bg-white/90 p-6 shadow-sm">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Latest status</p>
          <h3 className="mt-2 text-2xl font-semibold text-slate-900">{riskLevel} risk</h3>
        </div>
        <RiskBadge level={riskLevel} />
      </div>

      <div className="mt-6 grid grid-cols-2 gap-4 text-sm text-slate-600">
        <div>
          <p className="text-xs uppercase text-slate-400">HbA1c</p>
          <p className="text-lg font-semibold text-slate-900">
            {latest?.record?.hba1c ?? "--"}
          </p>
        </div>
        <div>
          <p className="text-xs uppercase text-slate-400">Glucose</p>
          <p className="text-lg font-semibold text-slate-900">
            {latest?.record?.glucose ?? "--"}
          </p>
        </div>
        <div>
          <p className="text-xs uppercase text-slate-400">Last report</p>
          <p className="text-sm font-medium text-slate-900">{lastDate}</p>
        </div>
        <div>
          <p className="text-xs uppercase text-slate-400">Alert</p>
          <p className="text-sm font-medium text-slate-900">
            {alert === undefined ? "Unknown" : alert ? "Triggered" : "Clear"}
          </p>
        </div>
      </div>
    </div>
  );
}
