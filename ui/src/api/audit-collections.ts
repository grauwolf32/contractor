import { loadAllPages } from "./pages";

/**
 * Default number of pages a combined Audit view reads in one batch. With the
 * server page size of 50 this bounds a batch at 250 records; the caller shows
 * a "Load more" control and continues from `nextCursor` when `truncated`.
 */
const AUDIT_COLLECTION_MAX_PAGES = 5;

export interface AuditCollectionPage<T> {
  items: T[];
  page: { hasMore: boolean; nextCursor?: string };
}

export interface AuditCollection<T> {
  items: T[];
  /** More pages exist beyond the page cap; continue from `nextCursor`. */
  truncated: boolean;
  nextCursor?: string;
}

interface CollectAuditPagesOptions {
  /** Page to continue from (the `nextCursor` of a truncated batch). */
  cursor?: string;
  /** Maximum number of pages to read in this batch (at least 1). */
  maxPages?: number;
}

/**
 * Collect up to `maxPages` pages of an Audit collection starting at `cursor`.
 * A batch is never presented as complete when it is not: an invalid or
 * repeated cursor and a failed later page both reject the whole batch, and a
 * batch that stops at the page cap reports `truncated` with the cursor to
 * continue from.
 */
export async function collectAuditPages<T>(
  load: (cursor?: string) => Promise<AuditCollectionPage<T>>,
  options: CollectAuditPagesOptions = {},
): Promise<AuditCollection<T>> {
  const batch = await loadAllPages(load, {
    maximumPages: options.maxPages ?? AUDIT_COLLECTION_MAX_PAGES,
    ...(options.cursor === undefined ? {} : { cursor: options.cursor }),
  });
  if (batch.error !== null) throw batch.error;
  return {
    items: batch.items,
    truncated: batch.truncated,
    ...(batch.nextCursor === undefined ? {} : { nextCursor: batch.nextCursor }),
  };
}

/** Concatenate consecutive batches; the last batch decides continuation. */
export function mergeAuditCollections<T>(
  batches: readonly AuditCollection<T>[],
): AuditCollection<T> {
  const last = batches.at(-1);
  return {
    items: batches.flatMap((batch) => batch.items),
    truncated: last?.truncated ?? false,
    ...(last?.truncated && last.nextCursor !== undefined
      ? { nextCursor: last.nextCursor }
      : {}),
  };
}
