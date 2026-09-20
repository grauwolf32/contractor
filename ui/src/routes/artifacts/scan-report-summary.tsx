const SCAN_TOOLS = new Set([
  "scan_nuclei",
  "scan_naabu",
  "scan_sqlmap",
  "scan_ffuf",
]);

const INCOMPLETE_CODES = new Set([
  "scan_incomplete",
  "scan_timeout",
  "scan_request_failed",
  "invalid_scanner_output",
  "output_limit_exceeded",
]);

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isCount(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
}

function executionStatus(
  tool: string,
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
          name === "outputLimitExceeded") &&
        typeof value !== "boolean",
    ) ||
    (observation.invalidResultLines !== undefined &&
      !isCount(observation.invalidResultLines)) ||
    (observation.requestErrors !== undefined &&
      observation.requestErrors !== null &&
      !isCount(observation.requestErrors))
  ) {
    return "Unknown";
  }
  if (
    errorCode === "scanner_unavailable" ||
    errorCode === "nuclei_templates_unavailable" ||
    errorCode === "scan_proxy_unsupported"
  ) {
    return "Unavailable";
  }
  if (
    (typeof errorCode === "string" && INCOMPLETE_CODES.has(errorCode)) ||
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
    (typeof errorCode === "string" && errorCode !== "") ||
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
    (tool !== "scan_ffuf" || observation.scanComplete === true) &&
    (observation.scanComplete === undefined ||
      observation.scanComplete === true)
  ) {
    return "Completed";
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
    !SCAN_TOOLS.has(report.tool) ||
    typeof report.inputDigest !== "string" ||
    !/^sha256:[a-f0-9]{64}$/.test(report.inputDigest) ||
    !isRecord(report.inputArtifacts) ||
    !isRecord(report.observation)
  ) {
    return null;
  }
  const observation = report.observation;
  return (
    <section aria-label="Scanner execution">
      <h3>Scanner execution</h3>
      <dl className="metadata-grid">
        <div>
          <dt>Tool</dt>
          <dd>{report.tool}</dd>
        </div>
        <div>
          <dt>Technical outcome</dt>
          <dd>{executionStatus(report.tool, observation)}</dd>
        </div>
        {typeof observation.errorCode === "string" ? (
          <div>
            <dt>Error code</dt>
            <dd>{observation.errorCode}</dd>
          </div>
        ) : null}
        {Array.isArray(observation.results) ? (
          <div>
            <dt>Reported items</dt>
            <dd>{observation.results.length}</dd>
          </div>
        ) : null}
      </dl>
      <p className="muted-copy">
        This describes scanner execution. A completed scan or an empty report
        does not establish that the target is free of vulnerabilities.
      </p>
    </section>
  );
}
