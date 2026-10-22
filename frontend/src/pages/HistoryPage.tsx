import { useNavigate } from "react-router-dom";
import { HistoryList } from "../components/history/HistoryList";
import { TrendCharts } from "../components/history/TrendCharts";
import {
  useAssessmentHistory,
  useAssessmentHistorySummary,
  useDeleteHealthRecord,
  useReanalyzeRecord,
} from "../api/hooks";

export function HistoryPage() {
  const navigate = useNavigate();
  const history = useAssessmentHistory();
  const summary = useAssessmentHistorySummary();
  const reanalyze = useReanalyzeRecord();
  const deleteRecord = useDeleteHealthRecord();

  return (
    <div className="space-y-6">
      <TrendCharts
        summary={summary.data}
        isLoading={summary.isLoading}
        error={summary.isError ? (summary.error?.message ?? "The history summary could not be loaded.") : null}
      />
      <HistoryList
        items={history.data?.items}
        onView={(recordId) => navigate(`/results/${recordId}`)}
        onReanalyze={(recordId) =>
          reanalyze.mutate(recordId, {
            onSuccess: () => navigate(`/results/${recordId}`),
          })
        }
        onDelete={(recordId) => deleteRecord.mutate(recordId)}
      />
    </div>
  );
}
