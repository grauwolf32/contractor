import { describe, expect, it } from "vitest";

import type { ArtifactMetadata } from "../api/artifacts";
import type { WorkflowResource } from "../api/workflows";
import {
  artifactOptionKey,
  emptyExecutionOverrides,
  NO_CREDENTIAL_OVERRIDE,
  validateRunDraft,
} from "./validation";

const workflow: WorkflowResource = {
  ref: { name: "openapi-from-workspace", version: "fixture-1" },
  entryStage: "dependencies",
  parameters: { objective: { required: true }, note: { required: false } },
  inputs: {
    source: { required: true, mediaTypes: ["application/zip"] },
    existing: { required: false, mediaTypes: ["application/yaml"] },
  },
  outputs: { openapi: { required: true, mediaTypes: ["application/yaml"] } },
  stages: {},
};

const source: ArtifactMetadata = {
  artifact: {
    namespace: "projects",
    name: "source",
    revision: "revision-4",
  },
  mediaType: "application/zip",
  size: 1024,
  current: true,
  frozen: false,
  createdAt: "2026-08-31T12:00:00Z",
};

describe("Run draft validation", () => {
  it("rejects missing required values and incompatible exact Artifacts", () => {
    const incompatible = {
      ...source,
      mediaType: "text/plain",
    };
    const key = artifactOptionKey(incompatible.artifact);
    const result = validateRunDraft(
      workflow,
      {
        runtimeLabels: [],
        metadataLabels: [],
        parameters: {},
        artifacts: { source: key },
        overrides: emptyExecutionOverrides(),
      },
      new Map([[key, incompatible]]),
    );
    expect(result.request).toBeUndefined();
    expect(result.errors).toEqual({
      "artifact:source": "Artifact media type text/plain is not accepted.",
      "parameter:objective": "Required string parameter is missing.",
    });
  });

  it("builds exact refs and omits unused optional values", () => {
    const key = artifactOptionKey(source.artifact);
    const result = validateRunDraft(
      workflow,
      {
        runtimeLabels: ["debug"],
        metadataLabels: [
          { id: "label-2", key: "eval.id", value: "eval-01" },
          { id: "label-1", key: "purpose", value: "eval" },
        ],
        parameters: { objective: "Build OpenAPI" },
        artifacts: { source: key },
        overrides: emptyExecutionOverrides(),
      },
      new Map([[key, source]]),
    );
    expect(result.errors).toEqual({});
    expect(result.request).toEqual({
      workflow: `${workflow.ref.name}@${workflow.ref.version}`,
      runtimeLabels: ["debug"],
      labels: { "eval.id": "eval-01", purpose: "eval" },
      parameters: { objective: "Build OpenAPI" },
      artifacts: { source: source.artifact },
    });
  });

  it("uses only selector and active credential fields in global overrides", () => {
    const key = artifactOptionKey(source.artifact);
    const overrides = emptyExecutionOverrides();
    overrides.planner.modelPolicy = "planner-strong@2";
    overrides.planner.credential = NO_CREDENTIAL_OVERRIDE;
    overrides.workers.llmGateway = "local@1";
    overrides.workers.credential = "credential-worker";
    const result = validateRunDraft(
      workflow,
      {
        runtimeLabels: [],
        metadataLabels: [],
        parameters: { objective: "Build OpenAPI", note: "" },
        artifacts: { source: key },
        overrides,
      },
      new Map([[key, source]]),
    );
    expect(result.request?.executionConfig).toEqual({
      planner: { modelPolicy: "planner-strong@2", credential: null },
      workers: { llmGateway: "local@1", credential: "credential-worker" },
    });
    expect(result.request?.parameters).toEqual({
      objective: "Build OpenAPI",
      note: "",
    });
  });

  it("reports exact metadata key, value, duplicate, and count failures", () => {
    const labels = [
      { id: "upper", key: "Eval.ID", value: "one" },
      { id: "reserved", key: "contractor.owner", value: "one" },
      { id: "empty", key: "eval.case", value: "" },
      { id: "first", key: "eval.id", value: "one" },
      { id: "second", key: "eval.id", value: "two" },
      ...Array.from({ length: 28 }, (_, index) => ({
        id: `extra-${index}`,
        key: `extra.${index}`,
        value: "value",
      })),
    ];
    const result = validateRunDraft(
      { ...workflow, parameters: {}, inputs: {} },
      {
        runtimeLabels: [],
        metadataLabels: labels,
        parameters: {},
        artifacts: {},
        overrides: emptyExecutionOverrides(),
      },
      new Map(),
    );
    expect(result.request).toBeUndefined();
    expect(result.errors).toMatchObject({
      metadataLabels: "A Run can have at most 32 metadata labels.",
      "metadataLabel:upper:key":
        "Use lowercase ASCII segments separated by ., _ or -.",
      "metadataLabel:reserved:key": "The contractor. prefix is reserved.",
      "metadataLabel:empty:value": "Label value is required.",
      "metadataLabel:first:key": "Label key eval.id is duplicated.",
      "metadataLabel:second:key": "Label key eval.id is duplicated.",
    });
  });
});
