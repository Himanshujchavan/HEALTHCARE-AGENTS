import { useParams } from "react-router-dom";
import { ResultsOverview } from "../components/results/ResultsOverview";
import { useHealthAnalysis, useHealthRecord } from "../api/hooks";

export function ResultsPage() {
  const params = useParams();
  const recordId = Number(params.recordId);
  const record = useHealthRecord(recordId);
  const analysis = useHealthAnalysis(recordId);

  return <ResultsOverview analysis={analysis.data?.analysis} record={record.data} />;
}
