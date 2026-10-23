import { expect, it } from "vitest";
import { webScanWorkflow } from "../test/web-scans";
import { resolveWorkflowWidget } from "./registry";
it("offers the widget only for its exact supported published input contract", () => {
  expect(resolveWorkflowWidget(webScanWorkflow)?.id).toBe("nuclei-web");
  for (const contract of [
    { ...webScanWorkflow, ref: { name: "different", version: "1" } },
    { ...webScanWorkflow, ref: { name: "nuclei-web", version: "2" } },
    {
      ...webScanWorkflow,
      parameters: {
        ...webScanWorkflow.parameters,
        new_required: { required: true },
      },
    },
    {
      ...webScanWorkflow,
      inputs: {
        rulepack: { required: false, mediaTypes: ["application/yaml"] },
      },
    },
  ])
    expect(resolveWorkflowWidget(contract)).toBeUndefined();
});
