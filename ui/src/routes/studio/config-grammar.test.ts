import configIdentityCases from "../../../../api/testdata/v1alpha1/config-identity-cases.json";
import mediaTypeCases from "../../../../api/testdata/v1alpha1/media-type-cases.json";
import { describe, expect, it } from "vitest";

import { patchDraft, starter, type Draft, type Path } from "./document";
import { validateDraft } from "./validation";

// Each refusal names the grammar it broke, not some other rule.
const NAME_MESSAGE = expect.stringMatching(/^Use a name of at most 128 /);
const VERSION_MESSAGE = expect.stringMatching(/^Use a version of at most 64 /);
const SELECTOR_MESSAGE = expect.stringMatching(/^Use a versioned selector /);

// Studio refuses exactly what the Server refuses: the shared cases hold its
// configuration names, versions, selectors and file media types to the
// Server grammar.
const { id, version } = configIdentityCases;
const validNames = [...id.valid, "a".repeat(id.maxLength)];
const invalidNames = [...id.invalid, "a".repeat(id.maxLength + 1)];
const validVersions = [...version.valid, "1".repeat(version.maxLength)];
const invalidVersions = [...version.invalid, "1".repeat(version.maxLength + 1)];
const validSelectors = [
  ...validNames.map((name) => `${name}@1`),
  ...validVersions.map((value) => `a@${value}`),
];
const invalidSelectors = [
  ...invalidNames.map((name) => `${name}@1`),
  ...invalidVersions.map((value) => `a@${value}`),
  "a",
  "@1",
  "a@",
  "a@1@2",
];

function errorsAt(draft: Draft, path: Path): string[] {
  return validateDraft(draft)
    .filter(
      (problem) =>
        problem.severity === "error" &&
        JSON.stringify(problem.path) === JSON.stringify(path),
    )
    .map((problem) => problem.message);
}

function workflowWith(path: Path, value: unknown): Draft {
  return patchDraft(starter("Workflow"), path, value);
}

describe("Studio configuration grammar", () => {
  const namePath: Path = ["metadata", "name"];
  const versionPath: Path = ["metadata", "version"];

  it.each(validNames)("accepts the shared valid name %j", (value) => {
    expect(errorsAt(workflowWith(namePath, value), namePath)).toEqual([]);
  });

  it.each(invalidNames)("refuses the shared invalid name %j", (value) => {
    expect(errorsAt(workflowWith(namePath, value), namePath)).toEqual([
      NAME_MESSAGE,
    ]);
  });

  it.each(validVersions)("accepts the shared valid version %j", (value) => {
    expect(errorsAt(workflowWith(versionPath, value), versionPath)).toEqual([]);
  });

  it.each(invalidVersions)("refuses the shared invalid version %j", (value) => {
    expect(errorsAt(workflowWith(versionPath, value), versionPath)).toEqual([
      VERSION_MESSAGE,
    ]);
  });

  it("refuses a version written as a YAML number", () => {
    expect(errorsAt(workflowWith(versionPath, 1), versionPath)).toEqual([
      VERSION_MESSAGE,
    ]);
  });

  const selectorPaths = [
    ["spec", "stages", "start", "planner"],
    ["spec", "stages", "start", "agents", "worker", "template"],
  ].map((path) => ({ path, name: path.join(".") }));

  it.each(selectorPaths)(
    "accepts every shared valid selector at $name",
    ({ path }) => {
      for (const selector of validSelectors)
        expect(errorsAt(workflowWith(path, selector), path), selector).toEqual(
          [],
        );
    },
  );

  it.each(selectorPaths)(
    "refuses every invalid selector at $name",
    ({ path }) => {
      for (const selector of invalidSelectors)
        expect(errorsAt(workflowWith(path, selector), path), selector).toEqual([
          SELECTOR_MESSAGE,
        ]);
    },
  );

  it("holds Agent and check type selectors to the same grammar", () => {
    for (const [kind, path] of [
      ["AgentTemplate", ["spec", "modelPolicy"]],
      ["AgentTemplate", ["spec", "runtime"]],
      ["AuditProfile", ["spec", "workflows", "check", "ref"]],
    ] as const) {
      for (const selector of validSelectors)
        expect(
          errorsAt(patchDraft(starter(kind), [...path], selector), [...path]),
          `${kind} ${selector}`,
        ).toEqual([]);
      for (const selector of invalidSelectors)
        expect(
          errorsAt(patchDraft(starter(kind), [...path], selector), [...path]),
          `${kind} ${selector}`,
        ).toEqual([SELECTOR_MESSAGE]);
    }
  });

  it("holds execution selectors and credential IDs to the same grammar", () => {
    const selection = (row: Record<string, unknown>) =>
      workflowWith(["spec", "executionConfig"], { planner: row });
    const path: Path = ["spec", "executionConfig", "planner"];
    for (const selector of validSelectors)
      for (const key of ["modelPolicy", "llmGateway"])
        expect(
          errorsAt(selection({ [key]: selector }), [...path, key]),
          `${key} ${selector}`,
        ).toEqual([]);
    for (const selector of invalidSelectors)
      for (const key of ["modelPolicy", "llmGateway"])
        expect(
          errorsAt(selection({ [key]: selector }), [...path, key]),
          `${key} ${selector}`,
        ).toHaveLength(1);
    for (const credential of validNames)
      expect(
        errorsAt(selection({ credential }), [...path, "credential"]),
        credential,
      ).toEqual([]);
    for (const credential of invalidNames)
      expect(
        errorsAt(selection({ credential }), [...path, "credential"]),
        credential,
      ).toHaveLength(1);
  });

  const inputTypes: Path = ["spec", "inputs", "source", "mediaTypes"];
  const withTypes = (mediaTypes: unknown) =>
    workflowWith(["spec", "inputs"], {
      source: { required: true, mediaTypes },
    });

  it.each(mediaTypeCases.valid)(
    "accepts the shared valid media type %j",
    (value) => {
      expect(errorsAt(withTypes([value]), inputTypes)).toEqual([]);
    },
  );

  it.each(mediaTypeCases.invalid.filter((value) => value !== "*/*"))(
    "refuses the shared invalid media type %j",
    (value) => {
      expect(errorsAt(withTypes([value]), inputTypes)).toHaveLength(1);
    },
  );

  it("accepts the wildcard alone, each media type once", () => {
    expect(errorsAt(withTypes(["*/*"]), inputTypes)).toEqual([]);
    expect(errorsAt(withTypes(["text/plain", "*/*"]), inputTypes)).toHaveLength(
      1,
    );
    expect(
      errorsAt(withTypes(["text/plain", "text/plain"]), inputTypes),
    ).toHaveLength(1);
    expect(errorsAt(withTypes([]), inputTypes)).toHaveLength(1);
  });
});
