interface PreparedScan {
  targetUrl?: string;
  testParameters: readonly string[];
}
const widgets = new Map<
  string,
  {
    coverageKind: string;
    title: string;
    description: string;
    targetLabel: string;
    targetValue: (scan: PreparedScan) => string;
  }
>([
  [
    "nuclei",
    {
      coverageKind: "nuclei-url-template-scan",
      title: "Nuclei · URL template check",
      description:
        "Checks templates against the fixed URL. It does not replay the OpenAPI method, request body or authenticated session.",
      targetLabel: "Fixed URL",
      targetValue: (scan) => scan.targetUrl ?? "Unavailable",
    },
  ],
  [
    "sqlmap",
    {
      coverageKind: "sqlmap-request-scan",
      title: "SQLMap · Request check",
      description:
        "Checks the prepared request and selected test parameters. A finished scan leaves security coverage inconclusive until its observations are verified.",
      targetLabel: "Prepared request",
      targetValue: (scan) =>
        scan.testParameters.join(", ") || "No test parameters",
    },
  ],
]);
export function auditScanWidget(scanner: string | undefined) {
  return scanner === undefined ? undefined : widgets.get(scanner);
}
