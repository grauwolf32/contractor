import { describe, expect, it } from "vitest";

import {
  compareVersions,
  formatList,
  groupFamilies,
  groupRows,
  missingLabel,
  presentCheckType,
  readinessOf,
  scopeFieldsUsed,
  scopeSummary,
} from "./check-types";
import {
  asvsPilot,
  asvsReview,
  checklist,
  materialFixture,
  nuclei,
  observe,
  openapiYaml,
  profileFixture,
  sourceZip,
  top10,
  trace,
  unsupported,
  wstgLive,
  wstgLiveDetail,
} from "./test-fixtures";

describe("check type versions", () => {
  it("orders numeric versions as numbers and others naturally", () => {
    expect(["2", "10", "9", "1.10", "1.9"].sort(compareVersions)).toEqual([
      "1.9",
      "1.10",
      "2",
      "9",
      "10",
    ]);
    expect(compareVersions("beta-2", "beta-10")).toBeLessThan(0);
  });

  it("prefers the newest version the Server can run", () => {
    const newest = profileFixture("source-review", {
      version: "10",
      serverCompatible: false,
    });
    const runnable = profileFixture("source-review", { version: "9" });
    const older = profileFixture("source-review", { version: "2" });
    const [family] = groupFamilies([older, newest, runnable]);
    expect(family?.versions.map((profile) => profile.ref.version)).toEqual([
      "10",
      "9",
      "2",
    ]);
    expect(family?.preferred.ref.version).toBe("9");
  });

  it("falls back to the newest version when none can run", () => {
    const [family] = groupFamilies([
      profileFixture("old", { version: "1", serverCompatible: false }),
      profileFixture("old", { version: "3", serverCompatible: false }),
    ]);
    expect(family?.preferred.ref.version).toBe("3");
  });
});

describe("check type rows", () => {
  it("folds scope variants of one standard into one row, widest first", () => {
    const rows = groupRows(
      groupFamilies([asvsPilot, trace, asvsReview, top10, observe]),
    );
    expect(
      rows.map((row) => row.variants.map((family) => family.name)),
    ).toEqual([
      ["openapi-operation-trace"],
      ["openapi-operation-observe"],
      ["owasp-top10-2025-source-risk"],
      ["owasp-asvs-5-0-l1-source-review", "owasp-asvs-5-0-l1-source-pilot"],
    ]);
  });

  it("keeps check types apart when their inputs or rules differ", () => {
    const liveVariant = profileFixture("wstg-live", {
      ...wstgLive,
      version: "1",
    });
    const sourceVariant = profileFixture("wstg-source", {
      mode: "requirements-verification",
      standards: [{ scheme: "owasp-wstg", version: "4.2" }],
    });
    const rows = groupRows(groupFamilies([liveVariant, sourceVariant]));
    expect(rows).toHaveLength(2);
  });

  it("labels known check types and describes others from their contract", () => {
    expect(presentCheckType(trace).label).toBe("API endpoint trace");
    const custom = profileFixture("payments-review", {
      mode: "requirements-verification",
      standards: [{ scheme: "pci", version: "4" }],
    });
    expect(presentCheckType(custom)).toEqual({
      label: "Payments review",
      description:
        "Requirements verification against pci 4, one requirement at a time.",
      order: 1000,
    });
  });
});

describe("readiness", () => {
  const ready = { hasLiveTarget: false, needsLiveTarget: false };

  it("is ready when every required input has a material in a matching format", () => {
    expect(
      readinessOf(trace, { ...ready, materials: [sourceZip, openapiYaml] }),
    ).toEqual({ state: "ready" });
  });

  it("names each missing input with its formats", () => {
    const readiness = readinessOf(checklist, { ...ready, materials: [] });
    expect(readiness).toEqual({
      state: "missing",
      missing: [
        {
          kind: "input",
          name: "checklist",
          label: "Checklist",
          formats: "JSON or YAML",
        },
        { kind: "input", name: "source", label: "Source code", formats: "ZIP" },
      ],
    });
    expect(
      readiness.state === "missing" ? readiness.missing.map(missingLabel) : [],
    ).toEqual(["Checklist (JSON or YAML)", "Source code (ZIP)"]);
  });

  it("needs a material of its own for each required input", () => {
    // The API spec input also takes ZIP bundles, so one ZIP matches both
    // inputs by format; which one lacks a material is not known.
    const readiness = readinessOf(trace, { ...ready, materials: [sourceZip] });
    expect(readiness).toEqual({
      state: "missing",
      missing: [
        {
          kind: "shared",
          names: ["openapi", "source"],
          labels: ["API spec", "Source code"],
          count: 1,
        },
      ],
    });
    expect(
      readiness.state === "missing" ? readiness.missing.map(missingLabel) : [],
    ).toEqual(["Another material for API spec or source code"]);
    expect(
      readinessOf(trace, { ...ready, materials: [sourceZip, openapiYaml] }),
    ).toEqual({ state: "ready" });
    // A later input can take a material an earlier one held.
    expect(
      readinessOf(trace, { ...ready, materials: [openapiYaml, sourceZip] }),
    ).toEqual({ state: "ready" });
  });

  it("does not count one JSON as both an API spec and scan settings", () => {
    const spec = materialFixture("openapi", "application/json");
    const settings = materialFixture("scan-settings", "application/json");
    const readiness = readinessOf(nuclei, { ...ready, materials: [spec] });
    expect(
      readiness.state === "missing" ? readiness.missing.map(missingLabel) : [],
    ).toEqual(["Another material for API spec or scan settings"]);
    expect(
      readinessOf(nuclei, { ...ready, materials: [spec, settings] }),
    ).toEqual({ state: "ready" });
  });

  it("ignores optional inputs", () => {
    const profile = profileFixture("notes", {
      inputs: {
        source: { required: true, mediaTypes: ["application/zip"] },
        notes: { required: false, mediaTypes: ["text/*"] },
      },
    });
    expect(readinessOf(profile, { ...ready, materials: [sourceZip] })).toEqual({
      state: "ready",
    });
  });

  it("needs the project's live target for check types that test one", () => {
    const context = materialFixture("brief", "text/markdown");
    expect(
      readinessOf(wstgLive, {
        materials: [context],
        hasLiveTarget: false,
        needsLiveTarget: true,
      }),
    ).toEqual({ state: "missing", missing: [{ kind: "live-target" }] });
    expect(
      readinessOf(wstgLive, {
        materials: [context],
        hasLiveTarget: true,
        needsLiveTarget: true,
      }),
    ).toEqual({ state: "ready" });
  });

  it("reports check types the Server cannot run with its reasons", () => {
    expect(readinessOf(unsupported, { ...ready, materials: [] })).toEqual({
      state: "unavailable",
      reasons: ["multiple_rounds_unsupported"],
    });
  });

  it("reads scope fields only from workflow parameters", () => {
    expect([...scopeFieldsUsed(wstgLiveDetail)].sort()).toEqual([
      "authorizationScope",
      "target",
    ]);
    expect(scopeFieldsUsed(wstgLive).size).toBe(0);
  });
});

describe("formats and scope", () => {
  it("names formats in words", () => {
    expect(formatList(["application/zip"])).toBe("ZIP");
    expect(formatList(["application/json", "application/yaml"])).toBe(
      "JSON or YAML",
    );
    expect(formatList(["text/plain", "text/markdown"])).toBe(
      "text or Markdown",
    );
  });

  it("describes what each inventory covers", () => {
    expect(scopeSummary(asvsReview).size).toBe("70 requirements");
    expect(scopeSummary(asvsReview).detail).toContain("Level 1");
    expect(scopeSummary(top10)).toEqual({
      size: "Up to 10 requirements",
      detail: "From owasp-web-top10 2025.",
    });
    expect(scopeSummary(trace)).toEqual({
      size: "Every endpoint in the API spec",
      detail: "At most 256 endpoints.",
    });
    expect(scopeSummary(nuclei).size).toBe(
      "The endpoints your scan settings select",
    );
    expect(scopeSummary(checklist).size).toBe("Every item of your checklist");
  });
});
