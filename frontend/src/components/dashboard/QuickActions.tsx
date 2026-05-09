import { Link } from "react-router-dom";

const actions = [
  { label: "Upload PDF Report", to: "/analyzer?tab=pdf" },
  { label: "Manual Data Entry", to: "/analyzer?tab=manual" },
  { label: "View History", to: "/history" },
];

export function QuickActions() {
  return (
    <div className="grid gap-3 md:grid-cols-2">
      {actions.map((action) => (
        <Link
          key={action.label}
          to={action.to}
          className="rounded-2xl border border-slate-200 bg-white/80 px-5 py-4 text-sm font-semibold text-slate-800 shadow-sm transition hover:-translate-y-0.5 hover:border-emerald-200 hover:bg-emerald-50"
        >
          {action.label}
        </Link>
      ))}
    </div>
  );
}
