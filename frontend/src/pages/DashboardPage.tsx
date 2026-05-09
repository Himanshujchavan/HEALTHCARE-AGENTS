import { DashboardOverview } from "../components/dashboard/DashboardOverview";
import { useHealthRecords, useLatestRecord } from "../api/hooks";

export function DashboardPage() {
  const latest = useLatestRecord();
  const reports = useHealthRecords(0, 5);

  return <DashboardOverview latest={latest.data} reports={reports.data} />;
}
