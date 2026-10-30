import { describe, expect, it, vi } from "vitest";

import fixture from "../../../api/testdata/v1alpha1/runtime-label-set-cases.json";
import type { RuntimeConfig } from "../config/runtime-config";
import {
  emptyExecutionOverrides,
  validateRunDraft,
} from "../run-drafts/validation";
import { PublicAPI } from "./client";
import { replaceRuntimeAgentPrincipalLabels } from "./operations";
import {
  canonicalRunRuntimeConfiguration,
  canonicalRuntimeLabels,
  RUNTIME_LABEL_LIMIT,
} from "./runtime-configuration";
import type { WorkflowResource } from "./workflows";

const runtimeConfig: RuntimeConfig = {
  uiVersion: "0.1.0",
  supportedApiVersions: ["contractor.public.v1"],
  apiBaseUrl: "http://127.0.0.1:8080",
};

const workflow: WorkflowResource = {
  ref: { name: "labels", version: "1" },
  entryStage: "only",
  parameters: {},
  inputs: {},
  outputs: {},
  stages: {},
};

const digest = `sha256:${"1".repeat(64)}`;

function pin(label: string) {
  return {
    label,
    bindingRevision: "1",
    config: { name: "labels", version: "1", digest },
  };
}

// The Runtime label set rule (RCF-R117 to RCF-R120) gives the Server's and
// the Runtime's verdicts on the shared cases in every browser check.
describe("Runtime label sets", () => {
  it("bounds the set as the Server does", () => {
    expect(fixture.maxLabels).toBe(RUNTIME_LABEL_LIMIT);
    expect(fixture.cases.length).toBeGreaterThan(0);
  });

  it.each(fixture.cases)("$name", async (test) => {
    const canonical = "canonical" in test ? test.canonical : undefined;
    expect(canonicalRuntimeLabels(test.labels)).toEqual(
      test.valid ? canonical : undefined,
    );

    // A Run draft names the set in any order and sends its canonical form.
    const draft = validateRunDraft(
      workflow,
      {
        runtimeLabels: test.labels,
        metadataLabels: [],
        parameters: {},
        artifacts: {},
        artifactReviews: {},
        overrides: emptyExecutionOverrides(),
      },
      () => undefined,
    );
    expect(draft.errors.runtimeLabels === undefined).toBe(test.valid);
    expect(draft.request?.runtimeLabels).toEqual(
      test.valid ? canonical : undefined,
    );

    // A label replacement sends the canonical set or nothing.
    const sent: Request[] = [];
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input: RequestInfo | URL) => {
        sent.push(input instanceof Request ? input : new Request(input));
        throw new Error("stop after sending");
      }),
    );
    api.csrf.replace("a".repeat(43));
    const replaced = replaceRuntimeAgentPrincipalLabels(
      api,
      "a".repeat(64),
      test.labels,
      "1",
      "labels-key",
    );
    if (test.valid) {
      await expect(replaced).rejects.not.toBeInstanceOf(TypeError);
      expect(sent).toHaveLength(1);
      expect(await sent[0]!.json()).toEqual({ labels: canonical });
    } else {
      await expect(replaced).rejects.toBeInstanceOf(TypeError);
      expect(sent).toHaveLength(0);
    }

    // A returned Run accepts only a set already in canonical order.
    const stored =
      test.valid && JSON.stringify(test.labels) === JSON.stringify(canonical);
    expect(
      canonicalRunRuntimeConfiguration({
        default: pin("default"),
        labels: test.labels.map(pin),
      }),
    ).toBe(stored);
  });
});
