import { describe, expect, it, vi } from "vitest";

import {
  AUDIT_COLLECTION_MAX_PAGES,
  collectAuditPages,
  mergeAuditCollections,
} from "./audit-collections";

function pagedLoader(pages: number) {
  return vi.fn(async (cursor?: string) => {
    const index = cursor === undefined ? 0 : Number(cursor);
    return {
      items: [index],
      page:
        index + 1 < pages
          ? { hasMore: true, nextCursor: String(index + 1) }
          : { hasMore: false },
    };
  });
}

describe("Audit collection pagination", () => {
  it.each([undefined, "", "repeated"])(
    "rejects a missing or repeated cursor (%s) instead of returning an incomplete collection",
    async (nextCursor) => {
      const load = vi.fn(async (cursor?: string) =>
        cursor === undefined
          ? { items: [1], page: { hasMore: true, nextCursor: "repeated" } }
          : {
              items: [2],
              page: {
                hasMore: true,
                ...(nextCursor === undefined ? {} : { nextCursor }),
              },
            },
      );
      await expect(collectAuditPages(load)).rejects.toThrow(
        "invalid Audit pagination cursor",
      );
      expect(load).toHaveBeenCalledTimes(2);
    },
  );

  it("does not return only the first page when a later page fails", async () => {
    await expect(
      collectAuditPages(async (cursor) => {
        if (cursor !== undefined) throw new Error("Page unavailable");
        return { items: [1], page: { hasMore: true, nextCursor: "next" } };
      }),
    ).rejects.toThrow("Page unavailable");
  });

  it("reads every page of a collection within the page cap", async () => {
    const load = pagedLoader(3);
    await expect(collectAuditPages(load)).resolves.toEqual({
      items: [0, 1, 2],
      truncated: false,
    });
    expect(load).toHaveBeenCalledTimes(3);
  });

  it("stops at the default page cap and reports the continuation cursor", async () => {
    const load = pagedLoader(AUDIT_COLLECTION_MAX_PAGES + 2);
    await expect(collectAuditPages(load)).resolves.toEqual({
      items: [0, 1, 2, 3, 4],
      truncated: true,
      nextCursor: "5",
    });
    expect(load).toHaveBeenCalledTimes(AUDIT_COLLECTION_MAX_PAGES);
  });

  it("continues from a cursor with an explicit cap and finishes the collection", async () => {
    const load = pagedLoader(7);
    await expect(
      collectAuditPages(load, { cursor: "5", maxPages: 2 }),
    ).resolves.toEqual({ items: [5, 6], truncated: false });
    expect(load.mock.calls.map(([cursor]) => cursor)).toEqual(["5", "6"]);
    await expect(
      collectAuditPages(load, { cursor: "2", maxPages: 1 }),
    ).resolves.toEqual({ items: [2], truncated: true, nextCursor: "3" });
  });

  it("rejects a cursor that leads back to the starting page", async () => {
    await expect(
      collectAuditPages(
        async () => ({
          items: [1],
          page: { hasMore: true, nextCursor: "start" },
        }),
        { cursor: "start" },
      ),
    ).rejects.toThrow("invalid Audit pagination cursor");
  });

  it.each([0, -1, 1.5, Number.NaN])(
    "rejects an invalid page cap (%s)",
    async (maxPages) => {
      await expect(
        collectAuditPages(pagedLoader(1), { maxPages }),
      ).rejects.toThrow("page cap is invalid");
    },
  );

  it("merges consecutive batches and keeps only the last continuation", () => {
    expect(
      mergeAuditCollections([
        { items: [1, 2], truncated: true, nextCursor: "a" },
        { items: [3], truncated: true, nextCursor: "b" },
      ]),
    ).toEqual({ items: [1, 2, 3], truncated: true, nextCursor: "b" });
    expect(
      mergeAuditCollections([
        { items: [1], truncated: true, nextCursor: "a" },
        { items: [2], truncated: false },
      ]),
    ).toEqual({ items: [1, 2], truncated: false });
    expect(mergeAuditCollections([])).toEqual({ items: [], truncated: false });
  });
});
