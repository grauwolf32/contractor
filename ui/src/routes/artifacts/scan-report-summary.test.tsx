import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import workerErrorCodes from "../../../../api/testdata/scan/worker-error-codes.json";
import { LoadedArtifactPreview } from "../../widgets/artifact-preview/view";

const OUTCOME_LABELS: Record<string, string> = {
  unavailable: "Unavailable",
  incomplete: "Incomplete",
  failed: "Failed",
};

const observation = {
  status: "completed",
  exitCode: 0,
  errorCode: null,
  stdoutTruncated: false,
  stderrTruncated: false,
  outputLimitExceeded: false,
  scanComplete: true,
  results: [],
};

function report(overrides: Record<string, unknown> = {}, tool = "scan_ffuf") {
  return JSON.stringify({
    schemaVersion: 1,
    tool,
    inputDigest: `sha256:${"a".repeat(64)}`,
    inputArtifacts: {},
    observation: { ...observation, ...overrides },
  });
}

describe("Scan report preview", () => {
  it.each([
    [true, "Completed"],
    [false, "Incomplete"],
  ])(
    "shows Semgrep coverage without confirming its matches",
    async (scanComplete, outcome) => {
      render(
        <LoadedArtifactPreview
          mediaType="application/json"
          source={report({ scanComplete }, "scan_semgrep")}
        />,
      );
      const summary = await screen.findByRole("region", {
        name: "Scanner execution",
      });
      expect(summary).toHaveTextContent(outcome);
      expect(summary).toHaveTextContent("possible issues pending review");
    },
  );

  it.each([
    [
      "Completed (bounded discovery)",
      {
        discoveryComplete: false,
        scanComplete: undefined,
        coverage: { limitations: ["bounded_crawl", "exhaustion_unverified"] },
      },
    ],
    [
      "No targets discovered",
      {
        status: "failed",
        errorCode: "no_discovered_targets",
        discoveryComplete: false,
        scanComplete: undefined,
      },
    ],
    [
      "Incomplete",
      {
        status: "failed",
        errorCode: "invalid_scanner_output",
        discoveryComplete: false,
        scanComplete: undefined,
      },
    ],
    [
      "Incomplete",
      {
        status: "failed",
        errorCode: "scan_request_failed",
        discoveryComplete: false,
        scanComplete: undefined,
      },
    ],
    ["Unknown", { discoveryComplete: "false", scanComplete: undefined }],
  ])("shows Katana technical outcome %s", async (outcome, overrides) => {
    const source = report(overrides, "scan_katana");
    const { container } = render(
      <LoadedArtifactPreview mediaType="application/json" source={source} />,
    );
    const summary = await screen.findByRole("region", {
      name: "Scanner execution",
    });
    expect(summary).toHaveTextContent("scan_katana");
    expect(summary).toHaveTextContent(outcome);
    if (outcome !== "Completed (bounded discovery)") {
      expect(summary).not.toHaveTextContent("Completed (bounded discovery)");
    }
    expect(summary).toHaveTextContent(
      "does not establish that every target was found",
    );
    expect(container.querySelector("pre")?.textContent).toBe(source);
  });

  it.each([
    ["Completed", { scanComplete: true }],
    ["Incomplete", { scanComplete: false }],
    ["Incomplete", { resultsTruncated: true }],
    ["Incomplete", { stdoutTruncated: true }],
    ["Incomplete", { invalidResultLines: 1 }],
    ["Incomplete", { requestErrors: 1 }],
    ["Incomplete", { status: "failed", errorCode: "scan_timeout" }],
    [
      "Unavailable",
      {
        status: "failed",
        errorCode: "scanner_unavailable",
        scanComplete: false,
      },
    ],
    ["Incomplete", { status: "failed", errorCode: "scan_proxy_unsupported" }],
    [
      "Failed",
      { status: "failed", errorCode: "scanner_failed", stdoutTruncated: true },
    ],
    ["Unknown", { status: "failed", errorCode: "scanner_exploded" }],
    ["Failed", { status: "failed", errorCode: "scanner_failed" }],
    ["Failed", { exitCode: 2 }],
    ["Unknown", { exitCode: null }],
    ["Unknown", { status: "unexpected" }],
    ["Unknown", { scanComplete: undefined }],
    ["Unknown", { resultsTruncated: "yes" }],
    ["Unknown", { invalidResultLines: -1 }],
    ["Unknown", { requestErrors: "0" }],
    ["Unknown", { exitCode: 0.5 }],
  ])(
    "shows %s technical outcome for %j and keeps source available",
    async (outcome, overrides) => {
      const source = report(overrides);
      const { container } = render(
        <LoadedArtifactPreview mediaType="application/json" source={source} />,
      );
      expect(
        await screen.findByRole("region", { name: "Scanner execution" }),
      ).toBeVisible();
      expect(screen.getByText(outcome)).toBeVisible();
      expect(
        screen.getByText(
          /does not establish that the target is free of vulnerabilities/,
        ),
      ).toBeVisible();
      expect(container.querySelector("pre")?.textContent).toBe(source);
    },
  );

  it.each(workerErrorCodes.codes)(
    "shows the shared Planner outcome for $errorCode",
    async ({ errorCode, exitCode, outcome }) => {
      render(
        <LoadedArtifactPreview
          mediaType="application/json"
          source={report({ status: "failed", errorCode, exitCode })}
        />,
      );
      const summary = await screen.findByRole("region", {
        name: "Scanner execution",
      });
      expect(summary).toHaveTextContent(
        `Technical outcome${OUTCOME_LABELS[outcome]}`,
      );
    },
  );

  it.each(["scan_nuclei", "scan_naabu", "scan_sqlmap"])(
    "renders the registered %s report without ffuf-specific accounting",
    async (tool) => {
      const source = report({ scanComplete: undefined }, tool);
      render(
        <LoadedArtifactPreview mediaType="application/json" source={source} />,
      );
      const summary = await screen.findByRole("region", {
        name: "Scanner execution",
      });
      expect(summary).toHaveTextContent(tool);
      expect(summary).toHaveTextContent("Technical outcomeCompleted");
    },
  );
  it.each(["scan_future", "__proto__", "toString"])(
    "keeps an unregistered %s report as raw evidence",
    async (tool) => {
      const source = report({}, tool);
      const { container } = render(
        <LoadedArtifactPreview mediaType="application/json" source={source} />,
      );
      expect(await screen.findByText(source)).toBeVisible();
      expect(
        screen.queryByRole("region", { name: "Scanner execution" }),
      ).toBeNull();
      expect(container.querySelector("pre")?.textContent).toBe(source);
    },
  );

  it("does not treat arbitrary JSON as a scanner report", async () => {
    render(
      <LoadedArtifactPreview
        mediaType="application/json"
        source='{"status":"completed"}'
      />,
    );
    expect(await screen.findByText('{"status":"completed"}')).toBeVisible();
    expect(
      screen.queryByRole("region", { name: "Scanner execution" }),
    ).toBeNull();
  });
});
