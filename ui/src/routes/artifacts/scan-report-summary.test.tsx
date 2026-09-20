import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { LoadedArtifactPreview } from "./loaded-preview";

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

function report(overrides: Record<string, unknown> = {}) {
  return JSON.stringify({
    schemaVersion: 1,
    tool: "scan_ffuf",
    inputDigest: `sha256:${"a".repeat(64)}`,
    inputArtifacts: {},
    observation: { ...observation, ...overrides },
  });
}

describe("Scan report preview", () => {
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
    ["Unavailable", { status: "failed", errorCode: "scan_proxy_unsupported" }],
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
