import type { Audit, AuditFinding, AuditReviewRequest } from "../api/audits";
import type { Project } from "../api/projects";

/**
 * The Server's `state=pending` rule for `GET /v1/reviews`: only requests that
 * can still be decided, so a stored pending request past its expiry is left
 * out (the Server judges it at its database clock).
 */
function decidable(review: AuditReviewRequest): boolean {
  return (
    review.expiresAt === undefined || Date.parse(review.expiresAt) > Date.now()
  );
}

/** Owner endpoints for UI fixtures. Per-check endpoints stay available for
 * detail, history and mutation assertions. */
export function ownerListResponse(
  url: URL,
  data: {
    projects: Project[];
    audits: Audit[];
    findings?: AuditFinding[];
    reviews?: AuditReviewRequest[];
  },
): Response | undefined {
  if (!["/v1/audits", "/v1/findings", "/v1/reviews"].includes(url.pathname))
    return;
  const matches = (field: string, value: string) =>
    !url.searchParams.has(field) ||
    url.searchParams.get(field)!.split(",").includes(value);
  const projects = new Set(
    data.projects
      .filter((p) => p.kind === "project" && p.lifecycle === "active")
      .map((p) => p.projectId),
  );
  const audits = data.audits.filter(
    (a) =>
      projects.has(a.projectId) &&
      matches(url.pathname === "/v1/audits" ? "state" : "auditState", a.state),
  );
  const auditIDs = new Set(audits.map((a) => a.auditId));
  const items =
    url.pathname === "/v1/audits"
      ? audits
      : url.pathname === "/v1/findings"
        ? (data.findings ?? []).filter(
            (f) =>
              auditIDs.has(f.auditId) &&
              matches("state", f.state) &&
              matches("verdict", f.analystVerdict ?? "unreviewed") &&
              (!url.searchParams.has("severity") ||
                matches("severity", f.analystSeverity ?? "")),
          )
        : (data.reviews ?? []).filter(
            (r) =>
              auditIDs.has(r.auditId) &&
              matches("state", r.state) &&
              (url.searchParams.get("state") !== "pending" || decidable(r)),
          );
  const offset = Number(url.searchParams.get("cursor") ?? 0);
  const limit = Number(url.searchParams.get("limit") ?? 50);
  const hasMore = offset + limit < items.length;
  return new Response(
    JSON.stringify({
      items: items.slice(offset, offset + limit),
      page: {
        hasMore,
        ...(hasMore ? { nextCursor: String(offset + limit) } : {}),
      },
    }),
    {
      headers: {
        "Content-Type": "application/json",
        "X-Contractor-API-Version": "contractor.public.v1",
      },
    },
  );
}
