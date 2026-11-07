import { describe, expect, it } from "vitest";
import fixture from "../../../test/fixtures/engagement-preview.json";
import shared from "../../../../../api/testdata/v1alpha1/engagement-cases.json";
import {
  ENGAGEMENT_MEDIA_TYPE,
  MAXIMUM_PREVIEW_BYTES,
} from "../../../api/artifacts";
import { createArtifactPreviewPlan } from "../preview-plan";
import {
  parseEngagementDocument,
  type EngagementDocument,
} from "./engagement-document";

describe("Saved engagement presentation", () => {
  it("reads current shared examples without changing saved policy values", async () => {
    for (const row of shared.documents.filter((row) => row.valid)) {
      expect(parseEngagementDocument(row.text), row.name).toEqual(
        JSON.parse(row.text),
      );
    }
    const source = JSON.stringify(fixture);
    expect(parseEngagementDocument(source)).toEqual(fixture);
    await expect(
      createArtifactPreviewPlan(ENGAGEMENT_MEDIA_TYPE, source),
    ).resolves.toEqual({ kind: "engagement", document: fixture });
    await expect(
      createArtifactPreviewPlan("application/json", source),
    ).resolves.toEqual({ kind: "source" });
  });
  it("rejects malformed syntax, repeated members and incompatible document shapes", async () => {
    const names = new Set([
      "unknown-schema",
      "null-optional",
      "missing-anonymous",
      "unknown-identity-kind",
      "empty-cidr",
    ]);
    for (const source of [
      "{",
      '{"schema":"contractor.engagement.v1","schema":"contractor.engagement.v1"}',
      ...shared.documents
        .filter((row) => names.has(row.name))
        .map((row) => row.text),
      JSON.stringify({ ...fixture, secret: "not a policy field" }),
      JSON.stringify({ ...fixture, schema: "contractor.engagement.v2" }),
    ]) {
      expect(parseEngagementDocument(source)).toBeUndefined();
      await expect(
        createArtifactPreviewPlan(ENGAGEMENT_MEDIA_TYPE, source),
      ).resolves.toEqual({ kind: "invalid-engagement" });
    }
  });
  it("keeps absent limits absent and respects the common byte ceiling", () => {
    const withoutLimits: EngagementDocument = structuredClone(fixture);
    delete withoutLimits.limits;
    const source = JSON.stringify(withoutLimits);
    expect(parseEngagementDocument(source)?.limits).toBeUndefined();
    expect(
      parseEngagementDocument(source.padEnd(MAXIMUM_PREVIEW_BYTES, " ")),
    ).toEqual(withoutLimits);
    expect(
      parseEngagementDocument(source.padEnd(MAXIMUM_PREVIEW_BYTES + 1, " ")),
    ).toBeUndefined();
  });
});
