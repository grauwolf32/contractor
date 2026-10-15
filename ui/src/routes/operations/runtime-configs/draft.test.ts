import { describe, expect, it } from "vitest";

import {
  RUNTIME_CONFIG_VERSION,
  RUNTIME_INFRASTRUCTURE_ID,
  schemaLimit,
} from "../../../api/contract";
import { buildDocument, emptyConfigDraft } from "./draft";

const MiB = 1024 * 1024;
const telemetry = {
  ...emptyConfigDraft(),
  name: "debug",
  workerTelemetry: true,
  workerTelemetryEndpoint: "https://otel.example/v1/traces",
};

describe("RuntimeConfig draft", () => {
  it("reports the name and version rules the contract defines", () => {
    const { errors } = buildDocument(
      { ...telemetry, name: "Debug", version: "-1" },
      [],
    );
    expect(errors).toContain(
      `Name must match [a-z][a-z0-9_-]* and contain at most ${RUNTIME_INFRASTRUCTURE_ID.maxLength} characters.`,
    );
    expect(errors).toContain(
      `Version must match [A-Za-z0-9][A-Za-z0-9._+-]* and contain at most ${RUNTIME_CONFIG_VERSION.maxLength} characters.`,
    );
  });

  it("bounds the export batch by the contract's maximum", () => {
    const maximum =
      schemaLimit(
        "WorkerTelemetryExportConfig",
        "maximum",
        "properties",
        "batchSizeBytes",
      ) / MiB;
    const fits = buildDocument(
      {
        ...telemetry,
        workerTelemetryBatchSize: String(maximum),
        workerTelemetryMaxPendingSize: String(maximum),
      },
      [],
    );
    expect(fits.errors).toEqual([]);
    expect(fits.document?.spec.worker?.telemetry?.export?.batchSizeBytes).toBe(
      maximum * MiB,
    );
    const { errors } = buildDocument(
      { ...telemetry, workerTelemetryBatchSize: String(maximum + 1) },
      [],
    );
    expect(errors).toContain(
      `Worker telemetry batch size (MiB) must be an integer from 1 to ${maximum}.`,
    );
  });
});
