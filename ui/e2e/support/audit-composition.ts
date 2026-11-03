import type { Page } from "@playwright/test";
import type { Audit } from "../../src/api/audits";
import { audit, conforming, project } from "../../src/test/builders";
import {
  fulfillContent,
  fulfillEmptyPageOrNotFound,
  fulfillJSON,
  mockAPI,
} from "./mock-api";

export const COMMON_PROJECT = "project_browser";
export const COMMON_AUDIT = "audit_browser";
export function commonAudit() {
  return audit(COMMON_PROJECT, COMMON_AUDIT, "paused", {
    profile: {
      name: "openapi-operation-observe",
      version: "1",
      digest: `sha256:${"a".repeat(64)}`,
    },
    taskCount: 2,
    reservedRunCount: 1,
    submittedRunCount: 1,
  });
}
export async function mockCommonTasks(page: Page, current: () => Audit) {
  await mockAPI(page, async (route, request, url) => {
    const value = current();
    if (url.pathname === `/v1/projects/${COMMON_PROJECT}`)
      return fulfillJSON(
        route,
        project(COMMON_PROJECT, { name: "Browser service" }),
        200,
        { etag: '"1"' },
      );
    if (url.pathname === `/v1/audits/${COMMON_AUDIT}`)
      return fulfillJSON(route, value, 200, { etag: `"${value.revision}"` });
    if (url.pathname === `/v1/audits/${COMMON_AUDIT}/tasks`) {
      const second = url.searchParams.has("cursor");
      const task = conforming("AuditTask", {
        document: {
          schema: "contractor.audit.task.v1",
          audit_id: COMMON_AUDIT,
          task_id: second ? "task_second" : "task_first",
          workflow: "inspect",
          objective: second
            ? "Inspect the second exact material"
            : "Inspect the selected exact material",
          inputs: {},
          parameters: {},
        },
        source: { kind: "manual", key: second ? "second" : "first" },
        ordinal: second ? 2 : 1,
        iteration: 1,
        state: second ? "ready" : "settled",
        attempts: second ? 0 : 1,
        ...(second
          ? {}
          : {
              result: {
                scope_kind: "audit",
                scope_id: COMMON_AUDIT,
                ref: {
                  namespace: "saved",
                  name: "result",
                  revision: "exact-result-r1",
                },
                digest: `sha256:${"b".repeat(64)}`,
                media_type: "application/json",
                size_bytes: 80,
              },
            }),
        created_at: value.createdAt,
        updated_at: value.updatedAt,
      });
      return fulfillJSON(route, {
        items: [task],
        page: second
          ? { hasMore: false }
          : { hasMore: true, nextCursor: "second" },
      });
    }
    if (url.pathname === `/v1/audits/${COMMON_AUDIT}/artifacts`) {
      if (
        url.searchParams.get("key") !== "task/task_first/result" ||
        url.searchParams.get("revision") !== "exact-result-r1"
      )
        throw new Error("Result must read the exact Task reference");
      return fulfillContent(
        route,
        JSON.stringify({
          rationale: "Exact saved observation",
          limitations: ["No live traffic"],
          long_text: "bounded ".repeat(200),
        }),
        "application/json",
      );
    }
    if (url.pathname === `/v1/audits/${COMMON_AUDIT}/events`)
      return fulfillJSON(route, {
        items: [],
        page: { hasMore: false },
        total: 0,
        throughSequence: 0,
      });
    if (
      request.method() === "GET" &&
      (url.pathname === "/v1/audits" ||
        url.pathname === `/v1/projects/${COMMON_PROJECT}/audits`)
    )
      return fulfillJSON(route, { items: [value], page: { hasMore: false } });
    await fulfillEmptyPageOrNotFound(route);
  });
}
