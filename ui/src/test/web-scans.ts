import { artifactMetadata, conforming } from "./builders";
import {
  runRuntimeConfiguration,
  workflowResource,
} from "./builders-execution";

export const webScanWorkflow = workflowResource("nuclei-web", "1", {
  parameters: {
    target: { required: true },
    template_ids: { required: false },
    tags: { required: false },
    severity: { required: false },
    rate_limit: { required: false },
  },
  inputs: {
    rulepack: {
      required: false,
      sourceScope: "user",
      mediaTypes: [
        "application/yaml",
        "application/vnd.contractor.nuclei-rulepack+zip",
      ],
    },
  },
});
export const webScanPack = artifactMetadata("rules", "web-check", {
  mediaType: "application/yaml",
});
export function webScanCreated(projectId?: string) {
  return conforming("CreateRunResponse", {
    runId: "run-web",
    state: "pending",
    runtimeLabels: [],
    labels: {},
    runtimeConfiguration: runRuntimeConfiguration(),
    ...(projectId === undefined ? {} : { projectId }),
  });
}
