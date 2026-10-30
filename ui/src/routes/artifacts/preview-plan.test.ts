import { describe, expect, it } from "vitest";

import {
  createArtifactPreviewPlan,
  MAXIMUM_AUTOMATIC_OPENAPI_BYTES,
} from "./preview-plan";

// Pads a document with a YAML comment to exactly `bytes` UTF-8 bytes.
function padded(document: string, bytes: number): string {
  const head = `${document}\n# `;
  return head + "x".repeat(bytes - new TextEncoder().encode(head).length);
}

describe("Artifact preview planning", () => {
  it("selects dedicated Markdown and LikeC4 renderers", async () => {
    await expect(
      createArtifactPreviewPlan("text/markdown", "# Report"),
    ).resolves.toEqual({ kind: "markdown" });
    await expect(
      createArtifactPreviewPlan("text/vnd.likec4", "model {}"),
    ).resolves.toEqual({ kind: "likec4" });
  });

  it("selects the colored renderer for unified diffs", async () => {
    await expect(
      createArtifactPreviewPlan(
        "text/x-diff",
        "--- a/report.txt\n+++ b/report.txt\n",
      ),
    ).resolves.toEqual({ kind: "diff" });
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

  it("leaves an OpenAPI document above LIM-UI-17 unparsed until asked for", async () => {
    const yaml =
      "openapi: 3.1.0\ninfo:\n  title: API\n  version: '1'\npaths: {}";
    const atLimit = padded(yaml, MAXIMUM_AUTOMATIC_OPENAPI_BYTES);
    await expect(
      createArtifactPreviewPlan("application/yaml", atLimit),
    ).resolves.toMatchObject({ kind: "openapi" });
    for (const [mediaType, source] of [
      ["application/yaml", padded(yaml, MAXIMUM_AUTOMATIC_OPENAPI_BYTES + 1)],
      [
        "text/yaml",
        padded("swagger: '2.0'", MAXIMUM_AUTOMATIC_OPENAPI_BYTES + 1),
      ],
      [
        "application/json",
        JSON.stringify({
          info: {
            title: "API",
            description: "x".repeat(MAXIMUM_AUTOMATIC_OPENAPI_BYTES),
          },
          openapi: "3.0.3",
        }),
      ],
    ] as const) {
      await expect(
        createArtifactPreviewPlan(mediaType, source),
      ).resolves.toEqual({
        kind: "openapi-on-request",
      });
    }
    // Large structured data that declares no OpenAPI version stays source,
    // including an "openapi" member below the top level.
    for (const [mediaType, source] of [
      [
        "application/yaml",
        padded("info:\n  openapi: 3.1.0", MAXIMUM_AUTOMATIC_OPENAPI_BYTES + 1),
      ],
      [
        "application/yaml",
        padded("openapi: 4.0.0", MAXIMUM_AUTOMATIC_OPENAPI_BYTES + 1),
      ],
      [
        "application/json",
        JSON.stringify({
          result: { openapi: "3.1.0" },
          padding: "x".repeat(MAXIMUM_AUTOMATIC_OPENAPI_BYTES),
        }),
      ],
    ] as const) {
      await expect(
        createArtifactPreviewPlan(mediaType, source),
      ).resolves.toEqual({
        kind: "source",
      });
    }
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
