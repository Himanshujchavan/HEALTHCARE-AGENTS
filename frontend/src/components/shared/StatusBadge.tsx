import { WorkflowStatus } from "../../models/health";

/**
 * The status vocabulary the backend uses, with a readable label and a colour
 * that matches the severity of the state.
 *
 * `completed_with_errors` is amber, not red: a score was produced, but from
 * fewer components than the documented formula. `failed` and `unreadable` are
 * red, because no usable assessment came back at all.
 */
const STATUS_STYLES: Record<string, string> = {
  completed: "border-emerald-200 bg-emerald-50 text-emerald-800",
  completed_with_errors: "border-amber-200 bg-amber-50 text-amber-800",
  partial: "border-amber-200 bg-amber-50 text-amber-800",
  failed: "border-red-200 bg-red-50 text-red-800",
  unreadable: "border-red-200 bg-red-50 text-red-800",
  unavailable: "border-slate-200 bg-slate-100 text-slate-600",
  not_applicable: "border-slate-200 bg-slate-50 text-slate-500",
  skipped: "border-slate-200 bg-slate-50 text-slate-500",
  pending: "border-sky-200 bg-sky-50 text-sky-700",
  unknown: "border-slate-200 bg-slate-100 text-slate-600",
};

const STATUS_LABELS: Record<string, string> = {
  completed: "Completed",
  completed_with_errors: "Completed with errors",
  partial: "Partial",
  failed: "Failed",
  unreadable: "Unreadable",
  unavailable: "Unavailable",
  not_applicable: "Not applicable",
  skipped: "Skipped",
  pending: "Pending",
  unknown: "Unknown",
};

export function statusLabel(status: string | null | undefined): string {
  if (!status) {
    return "Unknown";
  }
  return STATUS_LABELS[status] ?? status.replace(/_/g, " ");
}

interface StatusBadgeProps {
  status?: WorkflowStatus | string | null;
  className?: string;
}

export function StatusBadge({ status, className = "" }: StatusBadgeProps) {
  const key = status ?? "unknown";
  const style = STATUS_STYLES[key] ?? STATUS_STYLES.unknown;
  return (
    <span
      data-testid="status-badge"
      data-status={key}
      className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold ${style} ${className}`}
    >
      {statusLabel(key)}
    </span>
  );
}
