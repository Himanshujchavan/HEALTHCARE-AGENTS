import { HealthRecordResponse } from "../../models/health";
import { RiskBadge } from "../shared/RiskBadge";

interface RecentReportsProps {
  reports?: HealthRecordResponse[];
}

export function RecentReports({ reports }: RecentReportsProps) {
  if (!reports || reports.length === 0) {
    return (
      <div className="rounded-2xl border border-dashed border-slate-200 bg-white/70 p-6 text-sm text-slate-500">
        No reports yet. Upload a PDF or add manual data to get started.
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {reports.slice(0, 5).map((report) => (
        <div
          key={report.id}
          className="flex items-center justify-between rounded-2xl border border-slate-200 bg-white/80 px-5 py-4"
        >
          <div>
            <p className="text-sm font-semibold text-slate-900">Report #{report.id}</p>
            <p className="text-xs text-slate-500">
              {new Date(report.created_at).toLocaleDateString()} · HbA1c {report.hba1c}
            </p>
          </div>
          <RiskBadge level={report.analyzed ? "Analyzed" : "Pending"} />
        </div>
      ))}
    </div>
  );
}
