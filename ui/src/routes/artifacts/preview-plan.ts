export type ArtifactPreviewPlan =
  | { kind: "source" }
  | { kind: "diff" }
  | { kind: "markdown" }
  | { kind: "likec4" }
  | { kind: "openapi"; document: Record<string, unknown> };

const OPENAPI_MEDIA_TYPES = new Set([
  "application/json",
  "application/vnd.oai.openapi",
  "application/vnd.oai.openapi+json",
  "application/vnd.oai.openapi+yaml",
  "application/x-yaml",
  "application/yaml",
  "text/yaml",
]);

export async function createArtifactPreviewPlan(
  mediaType: string,
  source: string,
): Promise<ArtifactPreviewPlan> {
  if (mediaType === "text/x-diff") {
    return { kind: "diff" };
  }
  if (mediaType === "text/markdown" || mediaType === "text/x-markdown") {
    return { kind: "markdown" };
  }
  if (mediaType === "text/vnd.likec4") {
    return { kind: "likec4" };
  }
  if (!OPENAPI_MEDIA_TYPES.has(mediaType)) {
    return { kind: "source" };
  }

  const { parseOpenApiDocument } = await import("./previews/openapi-document");
  const document = parseOpenApiDocument(source);
  return document === undefined
    ? { kind: "source" }
    : { kind: "openapi", document };
}
