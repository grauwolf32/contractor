/** Collect every page for a combined Audit view; never present a partial list as complete. */
export async function collectAuditPages<T>(
  load: (cursor?: string) => Promise<{
    items: T[];
    page: { hasMore: boolean; nextCursor?: string };
  }>,
): Promise<T[]> {
  const items: T[] = [];
  const seen = new Set<string>();
  let cursor: string | undefined;
  for (;;) {
    const page = await load(cursor);
    items.push(...page.items);
    if (!page.page.hasMore) return items;
    const next = page.page.nextCursor;
    if (next === undefined || next === "" || seen.has(next)) {
      throw new Error(
        "The server returned an invalid Audit pagination cursor.",
      );
    }
    seen.add(next);
    cursor = next;
  }
}
