import { sha256Hex } from "../app/digest";
import type { PublicAPI } from "./client";
import { conformsTo, RESOURCE_ID, schemaLimit } from "./contract";
import { invalidAPIResponse, requireData } from "./error";
import type { components } from "./generated/public";

type Schemas = components["schemas"];
export type FindingExportRequest = Schemas["PublishFindingExportRequest"];
export type PublishedFindingExport = Schemas["PublishedFindingExport"];
export type FindingExportPart = "content" | "snapshot" | "diagnostics";
export const MAX_EXPORT_SELECTION = schemaLimit(
  "PublishFindingExportRequest",
  "maxItems",
  "properties",
  "findings",
);

function requireAudit(auditId: string): void {
  if (!RESOURCE_ID.test(auditId))
    throw new TypeError("Check identity is invalid");
}

export async function publishFindingExport(
  api: PublicAPI,
  auditId: string,
  request: FindingExportRequest,
  signal?: AbortSignal,
): Promise<PublishedFindingExport> {
  requireAudit(auditId);
  if (
    !conformsTo("PublishFindingExportRequest", request) ||
    new Set(request.findings.map((f) => f.findingId)).size !==
      request.findings.length
  )
    throw new TypeError("Export selection is invalid");
  const result = await api.request((client) =>
    client.POST("/v1/audits/{auditId}/findings/exports", {
      params: { path: { auditId } },
      body: request,
      ...(signal === undefined ? {} : { signal }),
    }),
  );
  const value = requireData(result);
  // Schema validation alone cannot bind the returned counts to this request.
  if (
    value.selectedCount !== request.findings.length ||
    value.resultCount > value.selectedCount ||
    (request.includeDuplicates === true &&
      value.resultCount !== value.selectedCount)
  )
    throw invalidAPIResponse(
      result.response.status,
      "Export counts did not match the selection",
    );
  return value;
}

/** Read only a frozen part, bound by its public schema and exact descriptor. */
export async function downloadFindingExport(
  api: PublicAPI,
  auditId: string,
  published: PublishedFindingExport,
  part: FindingExportPart,
  signal?: AbortSignal,
): Promise<{ blob: Blob; filename: string }> {
  requireAudit(auditId);
  if (!conformsTo("PublishedFindingExport", published))
    throw new TypeError("Export identity is invalid");
  const descriptor = part === "content" ? published.artifact : published[part];
  const maximum = schemaLimit(
    part === "diagnostics"
      ? "FindingExportDiagnosticsContent"
      : "FindingExportContent",
    "maxLength",
  );
  if (descriptor.sizeBytes > maximum)
    throw invalidAPIResponse(0, "Export exceeds the download limit");
  const response = await api.fetch(
    `/v1/audits/${encodeURIComponent(auditId)}/findings/exports/${encodeURIComponent(published.exportId)}/${part}`,
    {
      method: "GET",
      headers: { Accept: descriptor.mediaType },
      ...(signal === undefined ? {} : { signal }),
    },
  );
  if (!response.ok) throw await api.error(response);
  const invalid = () =>
    invalidAPIResponse(
      response.status,
      "Downloaded export did not match its saved version",
    );
  const length = response.headers.get("Content-Length");
  if (
    response.headers.get("Content-Type") !== descriptor.mediaType ||
    response.headers.get("ETag") !== `"${descriptor.ref.revision}"` ||
    (length !== null && length !== String(descriptor.sizeBytes)) ||
    response.body === null
  ) {
    await response.body?.cancel();
    throw invalid();
  }
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let size = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > descriptor.sizeBytes || size > maximum) {
        await reader.cancel();
        throw invalid();
      }
      chunks.push(value);
    }
  } finally {
    reader.releaseLock();
  }
  if (size !== descriptor.sizeBytes) throw invalid();
  const bytes = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  if (`sha256:${await sha256Hex(bytes)}` !== descriptor.digest) throw invalid();
  const suffix = part === "content" ? ".sarif" : `.${part}.json`;
  return {
    blob: new Blob([bytes], { type: descriptor.mediaType }),
    filename: `audit-${auditId}${suffix}`,
  };
}
