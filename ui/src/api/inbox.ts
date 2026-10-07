/**
 * Public API reads that only the Inbox makes (docs/design/ui/v3b-build-contract.md
 * §8: new wrappers go in a new file per area, existing ones stay as they are).
 */
import {
  AUDIT_ID_PATTERN,
  AUDIT_PAGE_SIZE,
  type AuditItem,
  type AuditItemPage,
} from "./audits";
import type { PublicAPI } from "./client";
import { invalidAPIResponse, requireData } from "./error";

/** The item state of a work item whose action waits for the owner. */
const AWAITING_REVIEW: AuditItem["state"] = "awaiting_review";

/**
 * One page of a check's work items that wait for the owner's decision
 * (`GET /v1/audits/{auditId}/items?state=awaiting_review`): the subjects of
 * its pending active test approvals and requirement applicability requests.
 * The Server holds an item in this state for as long as its request is
 * pending, so the page stays small however many items the check has.
 */
export async function listAwaitingReviewItems(
  api: PublicAPI,
  auditId: string,
  cursor?: string,
): Promise<AuditItemPage> {
  if (!AUDIT_ID_PATTERN.test(auditId))
    throw new TypeError("Audit ID is invalid");
  const result = await api.request((client) =>
    client.GET("/v1/audits/{auditId}/items", {
      params: {
        path: { auditId },
        query: {
          limit: AUDIT_PAGE_SIZE,
          state: AWAITING_REVIEW,
          ...(cursor === undefined ? {} : { cursor }),
        },
      },
    }),
  );
  const value = requireData(result);
  if (
    !Array.isArray(value.items) ||
    value.page === null ||
    typeof value.page !== "object" ||
    typeof value.page.hasMore !== "boolean"
  )
    throw invalidAPIResponse(
      result.response.status,
      "Server returned an invalid Audit item page",
    );
  return { items: structuredClone(value.items), page: { ...value.page } };
}
