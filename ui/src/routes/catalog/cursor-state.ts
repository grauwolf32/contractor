import { useLocation, useSearchParams } from "react-router";

function stateRecord(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

export function withoutCatalogPagination(
  value: unknown,
): Record<string, unknown> {
  const result = { ...stateRecord(value) };
  delete result.catalogPagination;
  return result;
}

function pageNumber(raw: string | null): number {
  if (raw === null || !/^[1-9][0-9]*$/.test(raw)) return 1;
  const parsed = Number(raw);
  return Number.isSafeInteger(parsed) ? parsed : 1;
}

export function useCatalogCursorState(
  cursorParam = "cursor",
  pageParam = "page",
) {
  const location = useLocation();
  const [searchParams, setSearchParams] = useSearchParams();
  const cursor = searchParams.get(cursorParam) || undefined;
  const page =
    cursor === undefined ? 1 : pageNumber(searchParams.get(pageParam));
  const filters = new URLSearchParams(searchParams);
  filters.delete(cursorParam);
  filters.delete(pageParam);
  filters.sort();
  const context = JSON.stringify([
    location.pathname,
    cursorParam,
    pageParam,
    filters.toString(),
  ]);
  const saved = stateRecord(stateRecord(location.state).catalogPagination);
  const previousCursors: Array<string | null> =
    saved.context === context &&
    saved.cursor === (cursor ?? null) &&
    saved.page === page &&
    Array.isArray(saved.previousCursors) &&
    saved.previousCursors.length < page &&
    saved.previousCursors.every((value, index, values) =>
      value === null
        ? page - values.length + index === 1
        : typeof value === "string" && value.length > 0,
    )
      ? saved.previousCursors
      : [];

  function goTo(
    nextCursor: string | undefined,
    nextPage: number,
    prior: Array<string | null>,
  ): void {
    const next = new URLSearchParams(searchParams);
    if (nextCursor === undefined) {
      next.delete(cursorParam);
      next.delete(pageParam);
    } else {
      next.set(cursorParam, nextCursor);
      next.set(pageParam, String(nextPage));
    }
    setSearchParams(next, {
      state: {
        ...withoutCatalogPagination(location.state),
        catalogPagination: {
          context,
          cursor: nextCursor ?? null,
          page: nextPage,
          previousCursors: prior,
        },
      },
    });
  }

  return {
    cursor,
    page,
    canGoBack: cursor !== undefined && previousCursors.length > 0,
    isFirstPage: cursor === undefined,
    nextPage: (nextCursor: string) => {
      if (Number.isSafeInteger(page + 1)) {
        goTo(nextCursor, page + 1, [...previousCursors, cursor ?? null]);
      }
    },
    previousPage: () => {
      if (cursor !== undefined && previousCursors.length > 0) {
        goTo(
          previousCursors.at(-1) ?? undefined,
          page - 1,
          previousCursors.slice(0, -1),
        );
      }
    },
    firstPage: () => goTo(undefined, 1, []),
  };
}
