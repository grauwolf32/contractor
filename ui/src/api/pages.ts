/** PageInfo, or an Eval page whose missing cursor is null. */
interface PageInfo {
  hasMore: boolean;
  nextCursor?: string | null;
}

/**
 * The most pages one load-all reader reads: 20 keyset pages, so a list stops
 * at 1,000 items at the usual 50 per page (4,000 for the owner-wide lists at
 * 200) instead of growing without bound (AGENTS.md 3.3). A reader that stops
 * here reports the list as truncated.
 */
export const MAXIMUM_LOADED_PAGES = 20;

/** A list read page by page, and whether it holds every item. */
export interface LoadedPages<T> {
  items: T[];
  /** More pages remained after the page cap; continue from `nextCursor`. */
  truncated: boolean;
  nextCursor?: string;
  /** A continuation failed; `items` holds the pages read before it. */
  error: Error | null;
}

/** The error of a list that must be complete but is longer than the cap. */
export class IncompleteListError extends Error {
  constructor(what: string) {
    super(
      `${what} has more entries than can be read at once, so it is incomplete.`,
    );
    this.name = "IncompleteListError";
  }
}

export type PageLoader<T> = (
  cursor: string | undefined,
) => Promise<{ items: T[]; page: PageInfo }>;

export interface LoadPagesOptions {
  signal?: AbortSignal;
  /** Pages to read at most; MAXIMUM_LOADED_PAGES by default. */
  maximumPages?: number;
  /** Continue a truncated list from its `nextCursor`. */
  cursor?: string;
}

/**
 * Reads keyset pages in order, at most `maximumPages`. A failed first page
 * throws, so a query keeps its last successful list; a failed or invalid
 * continuation keeps the pages already read and sets `error`.
 */
export async function loadAllPages<T>(
  load: PageLoader<T>,
  options: LoadPagesOptions = {},
): Promise<LoadedPages<T>> {
  const maximumPages = options.maximumPages ?? MAXIMUM_LOADED_PAGES;
  if (!Number.isInteger(maximumPages) || maximumPages < 1) {
    throw new TypeError("Page cap must be a positive integer");
  }
  const items: T[] = [];
  const seen = new Set<string>();
  let cursor = options.cursor;
  if (cursor !== undefined) seen.add(cursor);
  for (let read = 0; ; read++) {
    options.signal?.throwIfAborted();
    let result: Awaited<ReturnType<PageLoader<T>>>;
    try {
      result = await load(cursor);
    } catch (error) {
      if (read === 0) throw error;
      options.signal?.throwIfAborted();
      return {
        items,
        truncated: false,
        error: error instanceof Error ? error : new Error(String(error)),
      };
    }
    items.push(...result.items);
    if (!result.page.hasMore) return { items, truncated: false, error: null };
    // A continuation cursor must exist and never repeat: the PageInfo
    // schema cannot tie nextCursor to hasMore or to earlier pages.
    const next = result.page.nextCursor;
    if (next === undefined || next === null || next === "" || seen.has(next)) {
      return {
        items,
        truncated: false,
        error: new Error("Server returned an invalid page continuation"),
      };
    }
    if (read + 1 >= maximumPages) {
      return { items, truncated: true, nextCursor: next, error: null };
    }
    seen.add(next);
    cursor = next;
  }
}

/**
 * Reads every page of a list that is useless when partial (a picker that
 * must offer every choice, a match over the whole inventory). It never
 * returns part of the list: a failed or invalid continuation throws its
 * error, and a list longer than the page cap throws IncompleteListError.
 */
export async function loadCompletePages<T>(
  what: string,
  load: PageLoader<T>,
  options: LoadPagesOptions = {},
): Promise<T[]> {
  const loaded = await loadAllPages(load, options);
  if (loaded.error !== null) throw loaded.error;
  if (loaded.truncated) throw new IncompleteListError(what);
  return loaded.items;
}
