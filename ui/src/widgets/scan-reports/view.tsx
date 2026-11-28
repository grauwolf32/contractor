import { DIGEST } from "../../api/contract";
import { isRecord } from "../../api/json-guards";
import { scanReportWidget, type ScanReportWidget } from "./registry";
import { Facts } from "../../ui";

type WorkerErrorOutcome = "unavailable" | "incomplete" | "failed";

// The job outcome the Planner records for each Worker errorCode. Tests pin it
// to api/testdata/scan/worker-error-codes.json.
const WORKER_ERROR_OUTCOMES: ReadonlyMap<string, WorkerErrorOutcome> = new Map([
  ["semgrep_rules_unavailable", "unavailable"],
  ["semgrep_version_unsupported", "unavailable"],
  ["semgrep_source_unavailable", "incomplete"],
  ["semgrep_source_invalid", "incomplete"],
  ["semgrep_rulepack_unavailable", "incomplete"],
  ["semgrep_rulepack_invalid", "incomplete"],
  ["semgrep_rules_mismatch", "incomplete"],
  ["semgrep_scan_incomplete", "incomplete"],

  ["scanner_unavailable", "unavailable"],
  ["nuclei_templates_unavailable", "unavailable"],
  ["nuclei_rulepack_invalid", "incomplete"],
  ["nuclei_package_invalid", "incomplete"],
  ["nuclei_package_mismatch", "incomplete"],
  ["nuclei_selection_invalid", "incomplete"],
  ["scan_closed", "incomplete"],
  ["scan_proxy_unsupported", "incomplete"],
  ["scan_target_denied", "incomplete"],
  ["scan_target_unresolved", "incomplete"],
  ["scan_request_file_unavailable", "incomplete"],
  ["scan_wordlist_file_unavailable", "incomplete"],
  ["scan_artifact_unavailable", "incomplete"],
  ["scan_output_exists", "incomplete"],
  ["scan_timeout", "incomplete"],
  ["output_limit_exceeded", "incomplete"],
  ["scan_incomplete", "incomplete"],
  ["scan_request_failed", "incomplete"],
  ["invalid_scanner_output", "incomplete"],
  ["scanner_failed", "failed"],
  ["scan_artifact_failed", "failed"],
  ["no_discovered_targets", "failed"],
]);

function isCount(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
}

function executionStatus(
  widget: ScanReportWidget,
  observation: Record<string, unknown>,
): string {
  const errorCode = observation.errorCode;
  if (
    (observation.status !== "completed" && observation.status !== "failed") ||
    (observation.exitCode !== null &&
      !Number.isSafeInteger(observation.exitCode)) ||
    (errorCode !== null && typeof errorCode !== "string") ||
    Object.entries(observation).some(
      ([name, value]) =>
        (name.endsWith("Truncated") ||
          name === "scanComplete" ||
          name === "discoveryComplete" ||
          name === "outputLimitExceeded") &&
        typeof value !== "boolean",
    ) ||
    (widget.requiresDiscoveryState &&
      typeof observation.discoveryComplete !== "boolean") ||
    (observation.invalidResultLines !== undefined &&
      !isCount(observation.invalidResultLines)) ||
    (observation.requestErrors !== undefined &&
      observation.requestErrors !== null &&
      !isCount(observation.requestErrors))
  ) {
    return "Unknown";
  }
  if (typeof errorCode === "string") {
    // Like the Planner, trust only a known code and let it decide the outcome.
    switch (WORKER_ERROR_OUTCOMES.get(errorCode)) {
      case "unavailable":
        return "Unavailable";
      case "incomplete":
        return "Incomplete";
      case "failed":
        return errorCode === "no_discovered_targets"
          ? (widget.noTargetsOutcome ?? "Failed")
          : "Failed";
      default:
        return "Unknown";
    }
  }
  if (
    observation.scanComplete === false ||
    observation.outputLimitExceeded === true ||
    Object.entries(observation).some(
      ([name, value]) => name.endsWith("Truncated") && value === true,
    ) ||
    (typeof observation.requestErrors === "number" &&
      observation.requestErrors > 0) ||
    (typeof observation.invalidResultLines === "number" &&
      observation.invalidResultLines > 0)
  ) {
    return "Incomplete";
  }
  if (
    observation.status === "failed" ||
    (typeof observation.exitCode === "number" && observation.exitCode !== 0)
  ) {
    return "Failed";
  }
  if (
    observation.status === "completed" &&
    observation.exitCode === 0 &&
    errorCode === null &&
    observation.stdoutTruncated === false &&
    observation.stderrTruncated === false &&
    observation.outputLimitExceeded === false &&
    (!widget.requiresScanComplete || observation.scanComplete === true) &&
    (observation.scanComplete === undefined ||
      observation.scanComplete === true)
  ) {
    return widget.completedOutcome(observation);
  }
  return "Unknown";
}

export function ScanReportSummary({ source }: { source: string }) {
  let report: unknown;
  try {
    report = JSON.parse(source);
  } catch {
    return null;
  }
  if (
    !isRecord(report) ||
    report.schemaVersion !== 1 ||
    typeof report.tool !== "string" ||
    typeof report.inputDigest !== "string" ||
    !DIGEST.test(report.inputDigest) ||
    !isRecord(report.inputArtifacts) ||
    !isRecord(report.observation)
  ) {
    return null;
  }
  const widget = scanReportWidget(report.tool);
  if (widget === undefined) return null;
  const observation = report.observation;
  return (
    <section className="materials-scan" aria-label="Scanner execution">
      <h3 className="materials-subtitle">Scanner execution</h3>
      <Facts
        layout="row"
        className="materials-facts"
        items={[
          { term: "Tool", value: report.tool },
          {
            term: "Technical outcome",
            value: executionStatus(widget, observation),
          },
          {
            term: "Error code",
            value:
              typeof observation.errorCode === "string"
                ? observation.errorCode
                : undefined,
          },
          {
            term: "Reported items",
            value: Array.isArray(observation.results)
              ? observation.results.length
              : undefined,
          },
          {
            term: "Discovery limitations",
            value: widget.discoveryLimitations?.(observation),
          },
        ]}
      />
      <p className="ui-caption">
        This describes scanner execution. A completed scan or an empty report
        does not establish that the target is free of vulnerabilities.
        {widget.note ?? null}
      </p>
    </section>
  );
}
