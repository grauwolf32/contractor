import { describe, expect, it } from "vitest";

import type { ArtifactMetadata } from "../../api/artifacts";
import type { WorkflowSummary } from "../../api/workflows";
import { buildWorkflowCompatibility } from "./recommendations";

function artifact(
  namespace: string,
  name: string,
  revision: string,
  mediaType: string,
): ArtifactMetadata {
  return {
    artifact: { namespace, name, revision },
    mediaType,
    size: 10,
    current: true,
    frozen: false,
    createdAt: "2026-09-01T10:00:00Z",
  };
}

const openAPI: WorkflowSummary = {
  ref: { name: "openapi-from-source", version: "1" },
  entryStage: "analyze",
  parameters: { objective: { required: true } },
  inputs: { source: { required: true, mediaTypes: ["application/zip"] } },
  outputs: {
    openapi: {
      required: true,
      mediaTypes: ["application/yaml"],
      primary: true,
    },
    validation: { required: true, mediaTypes: ["text/plain"] },
  },
};

const likeC4: WorkflowSummary = {
  ref: { name: "likec4-from-source", version: "1" },
  entryStage: "analyze",
  parameters: {},
  inputs: { source: { required: true, mediaTypes: ["application/zip"] } },
  outputs: {
    likec4: {
      required: true,
      mediaTypes: ["text/vnd.likec4"],
      primary: true,
    },
  },
};

describe("Project Workflow compatibility", () => {
  it("recommends independent workflows and preselects one exact source", () => {
    const source = artifact(
      "sources",
      "service",
      "revision-source-1",
      "application/zip",
    );
    const result = buildWorkflowCompatibility([likeC4, openAPI], [source]);

    expect(result.map((item) => item.workflow.ref.name)).toEqual([
      "likec4-from-source",
      "openapi-from-source",
    ]);
    expect(result.every((item) => item.compatible && !item.suppressed)).toBe(
      true,
    );
    expect(result[0]?.preselected).toEqual({
      source: "sources/service@revision-source-1",
    });
    expect(result[1]?.preselected).toEqual({
      source: "sources/service@revision-source-1",
    });
  });

  it("suppresses only workflows whose complete primary output set exists", () => {
    const result = buildWorkflowCompatibility(
      [openAPI, likeC4],
      [
        artifact("sources", "service", "revision-source-1", "application/zip"),
        artifact(
          "outputs",
          "openapi",
          "revision-openapi-1",
          "application/yaml",
        ),
      ],
    );
    const openAPIResult = result.find(
      (item) => item.workflow.ref.name === "openapi-from-source",
    );
    const likeC4Result = result.find(
      (item) => item.workflow.ref.name === "likec4-from-source",
    );
    expect(openAPIResult).toMatchObject({ compatible: true, suppressed: true });
    expect(likeC4Result).toMatchObject({ compatible: true, suppressed: false });
  });

  it("requires explicit choice for multiple candidates and ignores historical bindings", () => {
    const first = artifact(
      "sources",
      "one",
      "revision-source-1",
      "application/zip",
    );
    const second = artifact(
      "sources",
      "two",
      "revision-source-2",
      "application/zip",
    );
    const historical = {
      ...artifact("sources", "old", "revision-source-old", "application/zip"),
      current: false,
    };
    const result = buildWorkflowCompatibility(
      [openAPI],
      [second, historical, first],
    )[0]!;

    expect(result.candidates.source?.map(artifactIdentity)).toEqual([
      "sources/one@revision-source-1",
      "sources/two@revision-source-2",
    ]);
    expect(result.preselected).toEqual({});
  });

  it("keeps incompatible workflows in All with exact missing slot names", () => {
    const result = buildWorkflowCompatibility(
      [openAPI],
      [artifact("docs", "readme", "revision-doc-1", "text/markdown")],
    )[0]!;
    expect(result.compatible).toBe(false);
    expect(result.suppressed).toBe(false);
    expect(result.missingRequiredInputs).toEqual(["source"]);
  });
});

function artifactIdentity(metadata: ArtifactMetadata): string {
  const { namespace, name, revision } = metadata.artifact;
  return `${namespace}/${name}@${revision}`;
}
