import { HistoryList } from "../components/history/HistoryList";
import { useHealthRecords, useReanalyzeRecord } from "../api/hooks";

export function HistoryPage() {
  const reports = useHealthRecords();
  const reanalyze = useReanalyzeRecord();

  return (
    <HistoryList
      reports={reports.data}
      onReanalyze={(recordId) => reanalyze.mutate(recordId)}
    />
  );
}
