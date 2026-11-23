import type { WorkflowSummary } from "../../api/workflows";

export function workflowDisplayName(workflow: WorkflowSummary): string {
  return workflow.presentation?.displayName ?? workflow.ref.name;
}

export function workflowSelector(workflow: WorkflowSummary): string {
  return `${workflow.ref.name}@${workflow.ref.version}`;
}
