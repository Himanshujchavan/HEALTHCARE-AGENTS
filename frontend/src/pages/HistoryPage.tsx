import { useNavigate } from "react-router-dom";
import { HistoryList } from "../components/history/HistoryList";
import { useDeleteHealthRecord, useHealthRecords, useReanalyzeRecord } from "../api/hooks";

export function HistoryPage() {
  const navigate = useNavigate();
  const reports = useHealthRecords();
  const reanalyze = useReanalyzeRecord();
  const deleteRecord = useDeleteHealthRecord();

  return (
    <HistoryList
      reports={reports.data}
      onView={(recordId) => navigate(`/results/${recordId}`)}
      onReanalyze={(recordId) =>
        reanalyze.mutate(recordId, {
          onSuccess: () => navigate(`/results/${recordId}`),
        })
      }
      onDelete={(recordId) => deleteRecord.mutate(recordId)}
    />
  );
}
