import type { Audit, AuditFinding, AuditReviewRequest } from "../api/audits";
import type { Project } from "../api/projects";
import { json, type FakeCall, type Routes } from "./fake-server";

export interface OwnerListData {
  projects: Project[];
  audits: Audit[];
  findings?: AuditFinding[];
  reviews?: AuditReviewRequest[];
}

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

/** Whether the comma-separated filter `field` is absent or lists `value`. */
function matches(url: URL, field: string, value: string): boolean {
  return (
    !url.searchParams.has(field) ||
    url.searchParams.get(field)!.split(",").includes(value)
  );
}

/** One page of `items` at the numeric `cursor` offset. */
function offsetPage(url: URL, items: unknown[]): Response {
  const offset = Number(url.searchParams.get("cursor") ?? 0);
  const limit = Number(url.searchParams.get("limit") ?? 50);
  const hasMore = offset + limit < items.length;
  return json({
    items: items.slice(offset, offset + limit),
    page: {
      hasMore,
      ...(hasMore ? { nextCursor: String(offset + limit) } : {}),
    },
  });
}

/**
 * Answers an owner-wide list request (`GET /v1/audits`, `/v1/findings`,
 * `/v1/reviews`) from `data`, or undefined for any other path. Like the
 * Server it lists only checks of active ordinary Projects and applies the
 * state, verdict and severity filters.
 */
export function ownerListResponse(
  url: URL,
  data: OwnerListData,
): Response | undefined {
  const projects = new Set(
    data.projects
      .filter((p) => p.kind === "project" && p.lifecycle === "active")
      .map((p) => p.projectId),
  );
  const audits = (field: "state" | "auditState") =>
    data.audits.filter(
      (a) => projects.has(a.projectId) && matches(url, field, a.state),
    );
  switch (url.pathname) {
    case "/v1/audits":
      return offsetPage(url, audits("state"));
    case "/v1/findings": {
      const auditIds = new Set(audits("auditState").map((a) => a.auditId));
      return offsetPage(
        url,
        (data.findings ?? []).filter(
          (f) =>
            auditIds.has(f.auditId) &&
            matches(url, "state", f.state) &&
            matches(url, "verdict", f.analystVerdict ?? "unreviewed") &&
            (!url.searchParams.has("severity") ||
              matches(url, "severity", f.analystSeverity ?? "")),
        ),
      );
    }
    case "/v1/reviews": {
      const auditIds = new Set(audits("auditState").map((a) => a.auditId));
      return offsetPage(
        url,
        (data.reviews ?? []).filter(
          (r) =>
            auditIds.has(r.auditId) &&
            matches(url, "state", r.state) &&
            (url.searchParams.get("state") !== "pending" || decidable(r)),
        ),
      );
    }
    default:
      return undefined;
  }
}

/** The owner-wide list operations as fake Server routes over `data()`. */
export function ownerListRoutes(data: () => OwnerListData): Routes {
  const route = ({ url }: FakeCall) => ownerListResponse(url, data())!;
  return {
    "GET /v1/audits": route,
    "GET /v1/findings": route,
    "GET /v1/reviews": route,
  };
}
