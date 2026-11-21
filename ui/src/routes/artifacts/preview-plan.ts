import {
  HTTP_EXCHANGE_MEDIA_TYPE,
  SITEMAP_MEDIA_TYPE,
  ENGAGEMENT_MEDIA_TYPE,
} from "../../api/artifacts";
import type { SitemapDocument } from "./previews/sitemap-document";
import type { EngagementDocument } from "./previews/engagement-document";
import { AUTOMATIC_DOCUMENT_BYTES } from "../../app/document-limits";
import {
  parseHTTPExchange,
  type HTTPExchangeDocument,
} from "../../api/http-exchange";

export type ArtifactPreviewPlan =
  | { kind: "source"; memorySnapshot?: true }
  | { kind: "diff" }
  | { kind: "markdown" }
  | { kind: "likec4" }
  | { kind: "http"; document: HTTPExchangeDocument }
  | { kind: "invalid-http" }
  | { kind: "sitemap"; document: SitemapDocument }
  | { kind: "invalid-sitemap" }
  | { kind: "engagement"; document: EngagementDocument }
  | { kind: "invalid-engagement" }
  | { kind: "openapi"; document: Record<string, unknown> }
  | { kind: "openapi-on-request" };

// LIM-UI-17: a larger OpenAPI document opens on Source and is parsed and
// rendered only when asked for, because Scalar renders every operation of an
// untagged document at once and holds the page for seconds at 1 MiB.
const MAXIMUM_AUTOMATIC_OPENAPI_BYTES = AUTOMATIC_DOCUMENT_BYTES;
export const AUTOMATIC_OPENAPI_LIMIT_LABEL = `${MAXIMUM_AUTOMATIC_OPENAPI_BYTES / 1024} KiB`;

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
  // This labels a declared source format, without interpreting or validating
  // its claims. The ordinary exact source reader retains all original bytes.
  if (mediaType === "application/json") {
    try {
      const value: unknown = JSON.parse(source);
      if (
        value !== null &&
        typeof value === "object" &&
        "schemaVersion" in value &&
        value.schemaVersion === "contractor.memory-snapshot/v1"
      )
        return { kind: "source", memorySnapshot: true };
    } catch {
      // Malformed JSON stays available as ordinary source text.
    }
  }
  if (mediaType === ENGAGEMENT_MEDIA_TYPE) {
    const { parseEngagementDocument } =
      await import("./previews/engagement-document");
    const document = parseEngagementDocument(source);
    return document === undefined
      ? { kind: "invalid-engagement" }
      : { kind: "engagement", document };
  }
  if (mediaType === SITEMAP_MEDIA_TYPE) {
    const { parseSitemapDocument } =
      await import("./previews/sitemap-document");
    const document = parseSitemapDocument(source);
    return document === undefined
      ? { kind: "invalid-sitemap" }
      : { kind: "sitemap", document };
  }
  if (mediaType === HTTP_EXCHANGE_MEDIA_TYPE) {
    const document = parseHTTPExchange(source);
    return document === undefined
      ? { kind: "invalid-http" }
      : { kind: "http", document };
  }
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

  const { declaresOpenApi, parseOpenApiDocument } =
    await import("./previews/openapi-document");
  if (
    new TextEncoder().encode(source).byteLength >
    MAXIMUM_AUTOMATIC_OPENAPI_BYTES
  ) {
    return declaresOpenApi(source)
      ? { kind: "openapi-on-request" }
      : { kind: "source" };
  }
  const document = parseOpenApiDocument(source);
  return document === undefined
    ? { kind: "source" }
    : { kind: "openapi", document };
}
