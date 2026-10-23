import type { WorkflowResource } from "../api/workflows";
import { NucleiWebFields } from "./nuclei-web-fields";
import { nucleiWebWidget } from "./nuclei-web";

// Explicit, bundled widgets only. An exact version plus compatible input and
// parameter contracts is required; a changed contract keeps the generic form.
export const workflowWidgets = {
  webScan: { ...nucleiWebWidget, Fields: NucleiWebFields },
};
export function resolveWorkflowWidget(workflow: WorkflowResource) {
  return Object.values(workflowWidgets).find((widget) =>
    widget.supports(workflow),
  );
}
