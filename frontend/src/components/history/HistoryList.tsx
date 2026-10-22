import { useMemo, useState } from "react";
import { AssessmentHistoryItem } from "../../models/health";
import { SectionCard } from "../shared/SectionCard";
import { StatusBadge } from "../shared/StatusBadge";

interface HistoryListProps {
  items?: AssessmentHistoryItem[];
  onReanalyze?: (recordId: number) => void;
  onView?: (recordId: number) => void;
  onDelete?: (recordId: number) => void;
}

const DISEASE_LABELS: Record<string, string> = {
  diabetes: "Diabetes",
  cardiovascular: "Cardiovascular",
};

function parameterSummary(parameters: AssessmentHistoryItem["parameters"]): string {
  const parts: string[] = [];
  const hba1c = parameters.hba1c;
  const glucose = parameters.glucose;
  const systolic = parameters.systolic_bp;

  if (hba1c !== undefined && hba1c !== null) {
    parts.push(`HbA1c ${hba1c}`);
  }
  if (glucose !== undefined && glucose !== null) {
    parts.push(`Glucose ${glucose}`);
  }
  if (systolic !== undefined && systolic !== null) {
    parts.push(`BP ${systolic}`);
  }

  return parts.length > 0 ? parts.join(" · ") : "No parameters recorded";
}

/** One entry per condition, with the level the condition was given. */
function DiseaseChips({ diseases }: { diseases: AssessmentHistoryItem["diseases"] }) {
  const entries = Object.entries(diseases ?? {});
  if (entries.length === 0) {
    return null;
  }
  return (
    <div className="mt-2 flex flex-wrap gap-2">
      {entries.map(([name, entry]) => (
        <span
          key={name}
          className="inline-flex items-center gap-1.5 rounded-full border border-slate-200 bg-slate-50 px-2.5 py-0.5 text-xs text-slate-600"
        >
          <span className="font-semibold">{DISEASE_LABELS[name] ?? name}</span>
          <span>{String(entry?.risk_level ?? "Not assessed")}</span>
          {entry?.status && <StatusBadge status={String(entry.status)} />}
        </span>
      ))}
    </div>
  );
}

export function HistoryList({ items, onReanalyze, onView, onDelete }: HistoryListProps) {
  const [query, setQuery] = useState("");
  const [sortDir, setSortDir] = useState("desc");

  const filtered = useMemo(() => {
    const list = items ?? [];
    const q = query.trim().toLowerCase();
    const sorted = [...list].sort((a, b) =>
      sortDir === "desc"
        ? new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
        : new Date(a.created_at).getTime() - new Date(b.created_at).getTime(),
    );

    if (!q) {
      return sorted;
    }

    return sorted.filter((item) =>
      `${item.record_id}`.includes(q) ||
      `${item.disease ?? ""}`.toLowerCase().includes(q) ||
      `${item.risk_level ?? ""}`.toLowerCase().includes(q) ||
      `${item.status ?? ""}`.toLowerCase().includes(q) ||
      `${item.parameters.hba1c ?? ""}`.includes(q) ||
      `${item.parameters.glucose ?? ""}`.includes(q),
    );
  }, [items, query, sortDir]);

  return (
    <SectionCard title="History">
      <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-center">
        <input
          type="text"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search by report ID, condition or metric"
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

        {filtered.map((item) => (
          <div
            key={item.record_id}
            className="flex flex-col gap-3 rounded-2xl border border-slate-200 bg-white/80 px-5 py-4 md:flex-row md:items-center md:justify-between"
          >
            <div>
              <p className="flex flex-wrap items-center gap-2 text-sm font-semibold text-slate-900">
                Report #{item.record_id} · {item.disease ?? "Health"} ·{" "}
                {item.risk_level ?? "Not assessed"}
                <StatusBadge status={item.status} />
              </p>
              <p className="text-xs text-slate-500">
                {new Date(item.created_at).toLocaleDateString()} ·{" "}
                {parameterSummary(item.parameters)}
                {item.score !== null && item.score !== undefined
                  ? ` · score ${item.score}`
                  : ""}
                {item.source === "report" ? " · from report" : ""}
              </p>
              <DiseaseChips diseases={item.diseases} />
            </div>
            <div className="flex flex-wrap gap-2">
              <button
                type="button"
                onClick={() => onView?.(item.record_id)}
                className="rounded-full border border-slate-200 px-3 py-1 text-xs font-semibold text-slate-600"
              >
                View report
              </button>
              <button
                type="button"
                onClick={() => onReanalyze?.(item.record_id)}
                className="rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-xs font-semibold text-emerald-800"
              >
                Re-analyze
              </button>
              <button
                type="button"
                onClick={() => {
                  if (!onDelete) {
                    return;
                  }

                  const confirmed = window.confirm(
                    `Delete report #${item.record_id}? This cannot be undone.`,
                  );
                  if (confirmed) {
                    onDelete(item.record_id);
                  }
                }}
                className="rounded-full border border-slate-200 px-3 py-1 text-xs font-semibold text-slate-600"
              >
                Delete
              </button>
            </div>
          </div>
        ))}
      </div>
    </SectionCard>
  );
}
