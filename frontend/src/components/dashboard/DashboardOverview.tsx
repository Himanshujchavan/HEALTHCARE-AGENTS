import { LatestRecordResponse, HealthRecordResponse } from "../../models/health";
import { HealthSummaryCard } from "./HealthSummaryCard";
import { QuickActions } from "./QuickActions";
import { RecentReports } from "./RecentReports";
import { SectionCard } from "../shared/SectionCard";
import { SectionHeader } from "../shared/SectionHeader";
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
        {(["diabetes", "cardiovascular"] as const).map((disease) => {
          const assessment = unified?.assessments[disease];
          return (
            <SectionCard key={disease} title={`${disease === "diabetes" ? "Diabetes" : "Cardiovascular"} summary`}>
              <p className="text-2xl font-semibold text-slate-900">
                {assessment?.status === "unavailable" ? "Not assessed" : assessment?.risk_level ?? "Not available"}
              </p>
              <p className="mt-2 text-sm text-slate-600">
                {assessment?.factors?.length ? assessment.factors.join(", ") : "No contributing factors returned."}
              </p>
            </SectionCard>
          );
        })}
      </div>

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
