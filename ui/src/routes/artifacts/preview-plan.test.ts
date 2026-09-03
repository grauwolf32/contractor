import { describe, expect, it } from "vitest";

import { createArtifactPreviewPlan } from "./preview-plan";

describe("Artifact preview planning", () => {
  it("selects dedicated Markdown and LikeC4 renderers", async () => {
    await expect(
      createArtifactPreviewPlan("text/markdown", "# Report"),
    ).resolves.toEqual({ kind: "markdown" });
    await expect(
      createArtifactPreviewPlan("text/vnd.likec4", "model {}"),
    ).resolves.toEqual({ kind: "likec4" });
  });

  it("detects OpenAPI in generic JSON and YAML media types", async () => {
    await expect(
      createArtifactPreviewPlan(
        "application/json",
        JSON.stringify({ openapi: "3.1.0", info: { title: "API" } }),
      ),
    ).resolves.toMatchObject({
      kind: "openapi",
      document: { openapi: "3.1.0" },
    });
    await expect(
      createArtifactPreviewPlan(
        "application/yaml",
        "swagger: '2.0'\ninfo:\n  title: Legacy API\n",
      ),
    ).resolves.toMatchObject({
      kind: "openapi",
      document: { swagger: "2.0" },
    });
  });

  it("keeps generic or malformed structured data as source", async () => {
    await expect(
      createArtifactPreviewPlan("application/json", '{"result": true}'),
    ).resolves.toEqual({ kind: "source" });
    await expect(
      createArtifactPreviewPlan("application/yaml", "openapi: ["),
    ).resolves.toEqual({ kind: "source" });
  });

  it("removes active resources and external references from OpenAPI", async () => {
    const plan = await createArtifactPreviewPlan(
      "application/yaml",
      [
        "openapi: 3.1.0",
        "info:",
        "  title: Safe API",
        "  x-logo:",
        "    url: https://example.invalid/logo.svg",
        "  description: '![tracking](https://example.invalid/pixel) ![local][asset]'",
        "components:",
        "  schemas:",
        "    Remote:",
        "      $ref: ./remote.yaml#/Remote",
      ].join("\n"),
    );

    expect(plan.kind).toBe("openapi");
    if (plan.kind !== "openapi") {
      throw new Error("expected OpenAPI render plan");
    }
    expect(plan.document).toMatchObject({
      info: {
        title: "Safe API",
        description: "[image omitted: tracking] [image omitted: local]",
      },
      components: {
        schemas: {
          Remote: {
            description: "External reference omitted from inline preview.",
          },
        },
      },
    });
    expect((plan.document.info as Record<string, unknown>)["x-logo"]).toBe(
      undefined,
    );
  });
});
