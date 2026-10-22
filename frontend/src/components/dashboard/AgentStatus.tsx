import { AgentStatus as AgentStatusValue } from "../../models/health";
import { StatusBadge } from "../shared/StatusBadge";

interface AgentStatusPanelProps {
  /** The agent statuses the workflow reported. */
  agentStatus?: Record<string, AgentStatusValue>;
}

/** The agents the workflow runs, in the order it runs them. */
const AGENT_ORDER = ["analysis", "risk", "symptom", "alert", "fusion", "trend", "differential"];

const AGENT_LABELS: Record<string, string> = {
  analysis: "Report Analyzer",
  risk: "Risk Predictor",
  symptom: "Symptom Checker",
  alert: "Alert Agent",
  fusion: "Result Fusion",
  trend: "Trend Agent",
  differential: "Differential Agent",
};

function agentLabel(name: string): string {
  return (
    AGENT_LABELS[name] ??
    name.charAt(0).toUpperCase() + name.slice(1).replace(/_/g, " ")
  );
}

/**
 * The state of each agent in the workflow.
 *
 * Without a workflow result there is nothing to report, so the panel says so
 * rather than showing every agent as ready — a "ready" badge next to an
 * assessment that was never run would be a claim the screen cannot support.
 */
export function AgentStatusPanel({ agentStatus }: AgentStatusPanelProps) {
  const entries = Object.entries(agentStatus ?? {});
  if (entries.length === 0) {
    return (
      <p className="text-sm text-slate-600">
        No agent statuses were reported for this assessment.
      </p>
    );
  }

  const ordered = [
    ...AGENT_ORDER.filter((name) => entries.some(([key]) => key === name)),
    ...entries.map(([name]) => name).filter((name) => !AGENT_ORDER.includes(name)),
  ];

  return (
    <div className="grid gap-3 md:grid-cols-2" data-testid="agent-status-panel">
      {ordered.map((name) => {
        const entry = entries.find(([key]) => key === name)?.[1];
        const status = entry?.status ?? "unknown";
        const detail = entry?.reason ?? entry?.error ?? null;
        return (
          <div
            key={name}
            className="rounded-2xl border border-slate-200 bg-white/80 px-5 py-4"
          >
            <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Agent</p>
            <p className="mt-2 text-sm font-semibold text-slate-900">{agentLabel(name)}</p>
            <div className="mt-2">
              <StatusBadge status={status} />
            </div>
            {detail && <p className="mt-2 text-xs text-slate-500">{detail}</p>}
          </div>
        );
      })}
    </div>
  );
}
