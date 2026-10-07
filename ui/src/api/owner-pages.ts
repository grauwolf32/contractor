import type { components } from "./generated/public";

export interface OwnerCollection<T> {
  items: T[];
  error: Error | null;
}

/** Read every keyset page sequentially. A failed continuation keeps the
 * pages already read and marks the collection incomplete. A failed head
 * throws so React Query retains its last successful collection on refresh.
 */
export async function collectOwnerPages<T>(
  load: (
    cursor?: string,
  ) => Promise<{ items: T[]; page: components["schemas"]["PageInfo"] }>,
): Promise<OwnerCollection<T>> {
  const items: T[] = [];
  const seen = new Set<string>();
  let cursor: string | undefined;
  let loaded = false;
  try {
    for (;;) {
      const result = await load(cursor);
      loaded = true;
      items.push(...result.items);
      if (!result.page.hasMore) return { items, error: null };
      const next = result.page.nextCursor;
      if (typeof next !== "string" || next.length === 0 || seen.has(next)) {
        throw new Error("Server returned an invalid page continuation");
      }
      seen.add(next);
      cursor = next;
    }
  } catch (error) {
    if (!loaded) throw error;
    return {
      items,
      error: error instanceof Error ? error : new Error(String(error)),
    };
  }
}
