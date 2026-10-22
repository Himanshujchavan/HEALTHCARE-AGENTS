import { LatestRecordResponse, HealthRecordResponse } from "../../models/health";
import { HealthSummaryCard } from "./HealthSummaryCard";
import { QuickActions } from "./QuickActions";
import { RecentReports } from "./RecentReports";
import { SectionCard } from "../shared/SectionCard";
import { SectionHeader } from "../shared/SectionHeader";
import { StatusBadge } from "../shared/StatusBadge";
import { AgentStatusPanel } from "./AgentStatus";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

interface DashboardOverviewProps {
  latest?: LatestRecordResponse;
  reports?: HealthRecordResponse[];
}

export function DashboardOverview({ latest, reports }: DashboardOverviewProps) {
  const unified = latest?.unified;
  return (
    <div className="space-y-8">
      <SectionHeader
        eyebrow="Overview"
        title="Daily health snapshot"
        subtitle="Your most recent risk assessment and quick actions."
      />

      <div className="grid gap-6 lg:grid-cols-[1.4fr_1fr]">
        <HealthSummaryCard latest={latest} />
        <SectionCard title="Analyze Health Report">
          <QuickActions />
        </SectionCard>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {Object.entries(unified?.assessments ?? {}).map(([disease, assessment]) => {
          const assessed =
            assessment?.status === "completed" ||
            assessment?.status === "completed_with_errors" ||
            assessment?.status === "partial";
          return (
            <SectionCard key={disease} title={`${disease === "diabetes" ? "Diabetes" : "Cardiovascular"} summary`}>
              <div className="flex items-center justify-between gap-3">
                <p className="text-2xl font-semibold text-slate-900">
                  {assessed ? assessment?.risk_level : "Not assessed"}
                </p>
                <StatusBadge status={assessment?.status ?? "unavailable"} />
              </div>
              <p className="mt-2 text-sm text-slate-600">
                {assessment?.factors?.length
                  ? assessment.factors.join(", ")
                  : assessed
                    ? "No contributing factors returned."
                    : assessment?.reason ?? "No contributing factors returned."}
              </p>
            </SectionCard>
          );
        })}
      </div>

      {unified?.agent_status && Object.keys(unified.agent_status).length > 0 && (
        <SectionCard title="Agent status">
          <AgentStatusPanel agentStatus={unified.agent_status} />
        </SectionCard>
      )}

      <SectionCard title="Recent reports">
        <RecentReports reports={reports} />
      </SectionCard>

      <SectionCard title="Measurement trends">
        {reports && reports.length > 1 ? (
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={[...reports].reverse().map((report) => ({
              date: new Date(report.created_at).toLocaleDateString(),
              glucose: report.glucose ?? null,
              hba1c: report.hba1c ?? null,
              systolic_bp: report.systolic_bp ?? null,
            }))}>
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="glucose" name="Glucose" stroke="#059669" connectNulls={false} />
              <Line type="monotone" dataKey="hba1c" name="HbA1c" stroke="#2563eb" connectNulls={false} />
              <Line type="monotone" dataKey="systolic_bp" name="Systolic BP" stroke="#d97706" connectNulls={false} />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <p className="text-sm text-slate-600">At least two assessments are needed to show measurement trends.</p>
        )}
      </SectionCard>
    </div>
  );
}
