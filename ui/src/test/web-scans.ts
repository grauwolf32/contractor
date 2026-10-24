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

export function scanSweep(
  state: "active" | "cancelling" | "completed" | "cancelled" = "active",
) {
  const terminal = state === "completed" || state === "cancelled";
  return conforming("ScanSweep", {
    sweepId: "sweep-one",
    state,
    workflow: "nuclei-web@1",
    snapshotDigest: `sha256:${"a".repeat(64)}`,
    createdAt: "2026-10-23T00:00:00Z",
    settledAt: terminal ? "2026-10-23T00:01:00Z" : null,
    counts: {
      total: 2,
      pending: terminal ? 0 : 2,
      admitted: terminal ? 1 : 0,
      skipped: terminal ? 1 : 0,
      failed: 0,
      cancelled: 0,
      unsettled: terminal ? 0 : 2,
    },
  });
}
export function sweepMember(ordinal: number, admitted = false) {
  return conforming("ScanSweepMember", {
    ordinal,
    projectId: `p${ordinal}`,
    target: "https://fixture.example/",
    state: admitted ? "admitted" : "pending",
    reason: "",
    runId: admitted ? `run-${ordinal}` : null,
    runState: admitted ? "failed" : null,
    settled: admitted,
  });
}
