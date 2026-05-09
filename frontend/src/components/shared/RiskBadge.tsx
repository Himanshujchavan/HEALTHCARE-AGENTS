import type { RiskLevel } from "../../models/health";

const styles: Record<string, string> = {
  Low: "bg-emerald-100 text-emerald-900",
  Moderate: "bg-amber-100 text-amber-900",
  High: "bg-orange-100 text-orange-900",
  Critical: "bg-red-100 text-red-900",
  Analyzed: "bg-slate-100 text-slate-700",
  Pending: "bg-slate-100 text-slate-700",
  Unknown: "bg-slate-100 text-slate-700",
};

interface RiskBadgeProps {
  level: RiskLevel | string | null | undefined;
}

export function RiskBadge({ level }: RiskBadgeProps) {
  const label = level || "Unknown";
  const style = styles[label] || styles.Unknown;
  return (
    <span
      className={`inline-flex items-center rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-wide ${style}`}
    >
      {label}
    </span>
  );
}
