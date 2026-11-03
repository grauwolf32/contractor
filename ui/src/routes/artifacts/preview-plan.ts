import { HTTP_EXCHANGE_MEDIA_TYPE } from "../../api/artifacts";
import { AUTOMATIC_DOCUMENT_BYTES } from "../../app/document-limits";
import {
  parseHTTPExchange,
  type HTTPExchangeDocument,
} from "../../api/http-exchange";

export type ArtifactPreviewPlan =
  | { kind: "source" }
  | { kind: "diff" }
  | { kind: "markdown" }
  | { kind: "likec4" }
  | { kind: "http"; document: HTTPExchangeDocument }
  | { kind: "invalid-http" }
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
