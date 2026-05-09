import { LatestRecordResponse, HealthRecordResponse } from "../../models/health";
import { HealthSummaryCard } from "./HealthSummaryCard";
import { QuickActions } from "./QuickActions";
import { RecentReports } from "./RecentReports";
import { SectionCard } from "../shared/SectionCard";
import { SectionHeader } from "../shared/SectionHeader";

interface DashboardOverviewProps {
  latest?: LatestRecordResponse;
  reports?: HealthRecordResponse[];
}

export function DashboardOverview({ latest, reports }: DashboardOverviewProps) {
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

      <SectionCard title="Recent reports">
        <RecentReports reports={reports} />
      </SectionCard>
    </div>
  );
}
