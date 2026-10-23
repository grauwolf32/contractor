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

describe("Interactsh infrastructure draft", () => {
  it("authors an atomic configuration, preserves clear and enforces timeout bounds", () => {
    const draft = {
      ...emptyConfigDraft(),
      name: "oast-lab",
      interactsh: true,
      interactshEndpoint: "https://oast.example/prefix",
      interactshCredential: "oast-token",
      interactshTimeout: "30",
    };
    expect(buildDocument(draft, []).document?.spec.worker?.interactsh).toEqual({
      adapter: "interactsh-http@1",
      endpoint: draft.interactshEndpoint,
      credential: "oast-token",
      requestTimeoutSeconds: 30,
    });
    expect(
      buildDocument({ ...draft, interactshClear: true }, []).document?.spec
        .worker?.interactsh,
    ).toBeNull();
    expect(
      buildDocument({ ...draft, interactshTimeout: "121" }, []).errors,
    ).not.toEqual([]);
    expect(
      buildDocument(
        { ...draft, interactshEndpoint: "https://user:secret@oast.example" },
        [],
      ).errors,
    ).not.toEqual([]);
  });
});
