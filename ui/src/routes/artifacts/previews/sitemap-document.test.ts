import { describe, expect, it } from "vitest";
import fixture from "../../../test/fixtures/sitemap-preview.json" with { type: "json" };
import shared from "../../../../../api/testdata/project-sitemap-cases.json" with { type: "json" };
import { parseSitemapDocument } from "./sitemap-document";
import { createArtifactPreviewPlan } from "../preview-plan";
import {
  MAXIMUM_PREVIEW_BYTES,
  SITEMAP_MEDIA_TYPE,
} from "../../../api/artifacts";

describe("Saved sitemap contract", () => {
  it("selects a dedicated renderer without altering persisted values", async () => {
    const source = JSON.stringify(fixture);
    expect(parseSitemapDocument(source)).toEqual(fixture);
    await expect(
      createArtifactPreviewPlan(SITEMAP_MEDIA_TYPE, source),
    ).resolves.toEqual({ kind: "sitemap", document: fixture });
    await expect(
      createArtifactPreviewPlan("application/json", source),
    ).resolves.toEqual({ kind: "source" });
  });
  it("distinguishes a valid empty document from invalid shared cases", async () => {
    expect(parseSitemapDocument(JSON.stringify(shared.empty))).toEqual(
      shared.empty,
    );
    for (const source of [
      ...shared.invalid,
      "{",
      JSON.stringify({ ...fixture, version: "project-sitemap@2" }),
      JSON.stringify({
        ...fixture,
        groups: [...fixture.groups, fixture.groups[0]],
      }),
    ]) {
      expect(parseSitemapDocument(source)).toBeUndefined();
      await expect(
        createArtifactPreviewPlan(SITEMAP_MEDIA_TYPE, source),
      ).resolves.toEqual({ kind: "invalid-sitemap" });
    }
  });
  it("uses the common preview byte boundary", () => {
    const source = JSON.stringify(shared.empty);
    expect(
      parseSitemapDocument(source.padEnd(MAXIMUM_PREVIEW_BYTES, " ")),
    ).toEqual(shared.empty);
    expect(
      parseSitemapDocument(source.padEnd(MAXIMUM_PREVIEW_BYTES + 1, " ")),
    ).toBeUndefined();
  });
});
