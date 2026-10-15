import { describe, expect, it, vi } from "vitest";

import { loadAllPages, loadCompletePages } from "./pages";

// LIM-UI-15 in docs/spec/platform/limits.md.
const MAXIMUM_LOADED_PAGES = 20;

/** A Server list of `pages` pages of 50 items, with a cursor per page. */
function pagedList(pages: number) {
  return vi.fn(async (cursor: string | undefined) => {
    const index = cursor === undefined ? 0 : Number(cursor);
    return {
      items: Array.from({ length: 50 }, (_, item) => `${index}:${item}`),
      page:
        index + 1 < pages
          ? { hasMore: true, nextCursor: String(index + 1) }
          : { hasMore: false },
    };
  });
}

describe("loadAllPages", () => {
  it("reads every page of a list within the cap", async () => {
    const load = pagedList(3);
    await expect(loadAllPages(load)).resolves.toMatchObject({
      truncated: false,
      error: null,
    });
    expect(load).toHaveBeenCalledTimes(3);
  });

  it("stops at the cap for an owner with more pages and reports truncation", async () => {
    const load = pagedList(MAXIMUM_LOADED_PAGES * 3);
    const loaded = await loadAllPages(load);
    expect(load).toHaveBeenCalledTimes(MAXIMUM_LOADED_PAGES);
    expect(loaded.items).toHaveLength(MAXIMUM_LOADED_PAGES * 50);
    expect(loaded).toMatchObject({ truncated: true, error: null });
  });

  it("is not truncated when the last allowed page is the last page", async () => {
    const loaded = await loadAllPages(pagedList(MAXIMUM_LOADED_PAGES));
    expect(loaded).toMatchObject({ truncated: false, error: null });
  });

  it("throws a failed first page and keeps pages read before a failed continuation", async () => {
    await expect(
      loadAllPages(async () => {
        throw new Error("down");
      }),
    ).rejects.toThrow("down");
    const load = pagedList(5);
    load.mockImplementationOnce(async () => ({
      items: ["0:0"],
      page: { hasMore: true, nextCursor: "1" },
    }));
    load.mockRejectedValueOnce(new Error("lost"));
    await expect(loadAllPages(load)).resolves.toMatchObject({
      items: ["0:0"],
      truncated: false,
      error: new Error("lost"),
    });
  });

  it.each([
    ["a missing cursor", { hasMore: true }],
    ["a repeated cursor", { hasMore: true, nextCursor: "again" }],
  ])("ends at %s", async (_, page) => {
    const load = vi.fn(async () => ({ items: [1], page }));
    const loaded = await loadAllPages(load);
    expect(loaded.error?.message).toBe(
      "Server returned an invalid page continuation",
    );
    expect(load.mock.calls.length).toBeLessThanOrEqual(2);
  });

  it("stops between pages when aborted", async () => {
    const controller = new AbortController();
    const load = pagedList(5);
    load.mockImplementationOnce(async () => {
      controller.abort();
      return { items: ["0:0"], page: { hasMore: true, nextCursor: "1" } };
    });
    await expect(
      loadAllPages(load, { signal: controller.signal }),
    ).rejects.toThrow();
    expect(load).toHaveBeenCalledTimes(1);
  });
});

describe("loadCompletePages", () => {
  it("returns a complete list and refuses a truncated or broken one", async () => {
    await expect(
      loadCompletePages("Workflow inventory", pagedList(2)),
    ).resolves.toHaveLength(100);
    await expect(
      loadCompletePages(
        "Workflow inventory",
        pagedList(MAXIMUM_LOADED_PAGES + 1),
      ),
    ).rejects.toThrow(
      "Workflow inventory has more entries than can be read at once, so it is incomplete.",
    );
    const load = pagedList(3);
    load.mockImplementationOnce(async () => ({
      items: ["0:0"],
      page: { hasMore: true },
    }));
    await expect(loadCompletePages("Workflow inventory", load)).rejects.toThrow(
      "Server returned an invalid page continuation",
    );
  });
});
