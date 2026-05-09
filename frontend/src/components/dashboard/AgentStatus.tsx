const agents = [
  { name: "Report Analyzer", status: "Ready" },
  { name: "Risk Predictor", status: "Ready" },
  { name: "Symptom Checker", status: "Ready" },
  { name: "Alert Agent", status: "Ready" },
];

export function AgentStatus() {
  return (
    <div className="grid gap-3 md:grid-cols-2">
      {agents.map((agent) => (
        <div
          key={agent.name}
          className="rounded-2xl border border-slate-200 bg-white/80 px-5 py-4"
        >
          <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Agent</p>
          <p className="mt-2 text-sm font-semibold text-slate-900">{agent.name}</p>
          <p className="text-xs text-emerald-600">{agent.status}</p>
        </div>
      ))}
    </div>
  );
}
