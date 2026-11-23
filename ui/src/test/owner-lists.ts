import { findingNeedsReview, findingSeverity } from "../api/audits";
import type { Audit, ProjectFinding } from "../api/audits";
import type { Project } from "../api/projects";
import { findingSummary } from "./builders";
import type { FakeCall, Routes } from "./fake-server";
import { json } from "./responses";

export interface OwnerListData {
  projects: Project[];
  audits: Audit[];
  findings?: ProjectFinding[];
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
      return offsetPage(
        url,
        (data.findings ?? [])
          .filter(
            (f) =>
              projects.has(f.projectId) &&
              matches(url, "state", f.state ?? "") &&
              (url.searchParams.get("verdict") === "needs_review"
                ? findingNeedsReview(f)
                : matches(url, "verdict", f.analystVerdict ?? "")) &&
              (!url.searchParams.has("severity") ||
                matches(url, "severity", findingSeverity(f))),
          )
          .map(findingSummary),
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
  };
}
