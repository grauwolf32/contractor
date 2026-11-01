import type { PublicAPI } from "./client";
import { RESOURCE_ID } from "./contract";
import { requireData } from "./error";
import type { components } from "./generated/public";

export type FindingPreview = components["schemas"]["FindingComparisonPreview"];
type CandidatesPage = components["schemas"]["FindingCandidatesPage"];

function requireID(id: string): void {
  if (!RESOURCE_ID.test(id)) throw new TypeError("Finding identity is invalid");
}

// Response schemas are enforced by PublicAPI's generated operation validators.
export async function listFindingCandidates(
  api: PublicAPI,
  auditId: string,
  findingId: string,
  request: {
    cursor?: string;
    sourceReceiptId?: string;
    scope?: "project" | "audit";
  } = {},
  signal?: AbortSignal,
): Promise<CandidatesPage> {
  requireID(auditId);
  requireID(findingId);
  if (request.sourceReceiptId !== undefined) requireID(request.sourceReceiptId);
  return requireData(
    await api.request((client) =>
      client.GET(
        "/v1/audits/{auditId}/findings/{findingId}/duplicate-candidates",
        {
          ...(signal === undefined ? {} : { signal }),
          params: {
            path: { auditId, findingId },
            query: { limit: 20, ...request },
          },
        },
      ),
    ),
  );
}
