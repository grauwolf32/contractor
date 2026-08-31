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
  ref: { name: "openapi-from-source", version: "1" },
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
        parameters: { objective: "Build OpenAPI" },
        artifacts: { source: key },
        overrides: emptyExecutionOverrides(),
      },
      new Map([[key, source]]),
    );
    expect(result.errors).toEqual({});
    expect(result.request).toEqual({
      workflow: "openapi-from-source@1",
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
});
