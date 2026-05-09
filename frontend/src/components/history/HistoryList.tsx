import { useMemo, useState } from "react";
import { HealthRecordResponse } from "../../models/health";
import { SectionCard } from "../shared/SectionCard";

interface HistoryListProps {
  reports?: HealthRecordResponse[];
  onReanalyze?: (recordId: number) => void;
}

export function HistoryList({ reports, onReanalyze }: HistoryListProps) {
  const [query, setQuery] = useState("");
  const [sortDir, setSortDir] = useState("desc");

  const filtered = useMemo(() => {
    const list = reports ?? [];
    const q = query.trim().toLowerCase();
    const sorted = [...list].sort((a, b) =>
      sortDir === "desc"
        ? new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
        : new Date(a.created_at).getTime() - new Date(b.created_at).getTime(),
    );

    if (!q) {
      return sorted;
    }

    return sorted.filter((report) =>
      `${report.id}`.includes(q) || `${report.hba1c}`.includes(q) || `${report.glucose}`.includes(q),
    );
  }, [reports, query, sortDir]);

  return (
    <SectionCard title="History">
      <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-center">
        <input
          type="text"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search by report ID or metric"
          className="flex-1 rounded-xl border border-slate-200 px-4 py-2 text-sm"
        />
        <select
          value={sortDir}
          onChange={(event) => setSortDir(event.target.value)}
          className="rounded-xl border border-slate-200 px-4 py-2 text-sm"
        >
          <option value="desc">Newest first</option>
          <option value="asc">Oldest first</option>
        </select>
      </div>

      <div className="space-y-3">
        {filtered.length === 0 && (
          <div className="rounded-xl border border-dashed border-slate-200 px-4 py-6 text-sm text-slate-500">
            No reports match your search.
          </div>
        )}

        {filtered.map((report) => (
          <div
            key={report.id}
            className="flex flex-col gap-3 rounded-2xl border border-slate-200 bg-white/80 px-5 py-4 md:flex-row md:items-center md:justify-between"
          >
            <div>
              <p className="text-sm font-semibold text-slate-900">Report #{report.id}</p>
              <p className="text-xs text-slate-500">
                {new Date(report.created_at).toLocaleDateString()} · HbA1c {report.hba1c}
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              <button className="rounded-full border border-slate-200 px-3 py-1 text-xs font-semibold text-slate-600">
                View report
              </button>
              <button
                type="button"
                onClick={() => onReanalyze?.(report.id)}
                className="rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-xs font-semibold text-emerald-800"
              >
                Re-analyze
              </button>
              <button className="rounded-full border border-slate-200 px-3 py-1 text-xs font-semibold text-slate-600">
                Delete
              </button>
            </div>
          </div>
        ))}
      </div>
    </SectionCard>
  );
}
