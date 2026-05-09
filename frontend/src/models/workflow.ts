export type WorkflowStepStatus = "pending" | "running" | "complete" | "error";

export interface WorkflowStep {
  key: string;
  label: string;
  status: WorkflowStepStatus;
  description?: string;
}
