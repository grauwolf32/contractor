import { describe, expect, it, vi } from "vitest";

import { collectAuditPages } from "./audit-collections";

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
});
