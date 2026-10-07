import { describe, expect, it } from "vitest";

import {
  fakeAPI,
  jsonResponse,
} from "../routes/projects/audits/check-test-support";
import { listAuditEvents, type AuditEventPage } from "./audits";

const valid: AuditEventPage = {
  items: [
    {
      auditId: "audit_example",
      sequence: 2,
      kind: "audit.resumed",
      entityId: "audit_example",
      summary: {},
      createdAt: "2026-10-07T10:00:00Z",
    },
  ],
  throughSequence: 2,
  total: 2,
  page: { hasMore: true, nextCursor: "older" },
};

describe("Audit event reads", () => {
  it("requests a bounded cancellable page and preserves immutable prefix metadata", async () => {
    const controller = new AbortController();
    const { api } = fakeAPI((request, url) => {
      expect(url.pathname).toBe("/v1/audits/audit_example/events");
      expect(Object.fromEntries(url.searchParams)).toEqual({
        limit: "50",
        cursor: "older",
      });
      expect(request.signal.aborted).toBe(false);
      return jsonResponse(valid);
    });
    expect(
      await listAuditEvents(
        api,
        "audit_example",
        { cursor: "older" },
        controller.signal,
      ),
    ).toEqual(valid);
  });

  it.each([
    { total: undefined },
    { throughSequence: undefined },
    { total: -1 },
    { throughSequence: 1 },
    { items: [{ ...valid.items[0]!, auditId: "foreign" }] },
    { items: [valid.items[0]!, valid.items[0]!] },
    { items: [{ ...valid.items[0]!, summary: null }] },
    { items: [], page: { hasMore: true, nextCursor: "older" } },
    { page: { hasMore: true } },
  ])("rejects malformed event pages %j", async (overrides) => {
    const { api } = fakeAPI(() => jsonResponse({ ...valid, ...overrides }));
    await expect(listAuditEvents(api, "audit_example")).rejects.toMatchObject({
      code: "invalid_api_response",
    });
  });
});
