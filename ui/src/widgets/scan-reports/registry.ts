import { isRecord } from "../../api/json-guards";

export interface ScanReportWidget {
  readonly requiresScanComplete: boolean;
  readonly requiresDiscoveryState: boolean;
  readonly noTargetsOutcome?: string;
  readonly note?: string;
  readonly discoveryLimitations?: (
    observation: Record<string, unknown>,
  ) => string | undefined;
  readonly completedOutcome: (observation: Record<string, unknown>) => string;
}
const standard: ScanReportWidget = {
  requiresScanComplete: false,
  requiresDiscoveryState: false,
  completedOutcome: () => "Completed",
};
// Tool-result widgets match the report's versioned tool contract, independently
// of the Workflow that produced it. Common outcome/error checks stay in view.
const widgets = new Map<string, ScanReportWidget>([
  ["scan_nuclei", standard],
  ["scan_naabu", standard],
  ["scan_sqlmap", standard],
  ["scan_ffuf", { ...standard, requiresScanComplete: true }],
  [
    "scan_katana",
    {
      ...standard,
      requiresDiscoveryState: true,
      noTargetsOutcome: "No targets discovered",
      note: " Katana discovery is bounded and does not establish that every target was found.",
      completedOutcome: (observation) =>
        observation.discoveryComplete === false
          ? "Completed (bounded discovery)"
          : "Completed",
      discoveryLimitations: (observation) =>
        isRecord(observation.coverage) &&
        Array.isArray(observation.coverage.limitations) &&
        observation.coverage.limitations.every(
          (reason) => typeof reason === "string",
        )
          ? observation.coverage.limitations.join(", ")
          : undefined,
    },
  ],
]);
export function scanReportWidget(tool: string) {
  return widgets.get(tool);
}
