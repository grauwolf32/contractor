import { useState } from "react";
import { type NavigateOptions, useSearchParams } from "react-router";

/** The continuation part of a public API list page. */
export interface PageContinuation {
  hasMore: boolean;
  nextCursor?: string | null | undefined;
}

/** CursorControls props for one position in a cursor stack. */
export interface CursorStackControls {
  canGoBack: boolean;
  nextCursor?: string;
  onBack: () => void;
  onNext: (cursor: string) => void;
}

/** Returns the cursor of the following page, if the page has one. */
export function nextPageCursor(
  page: PageContinuation | undefined,
): string | undefined {
  return page?.hasMore === true &&
    typeof page.nextCursor === "string" &&
    page.nextCursor.length > 0
    ? page.nextCursor
    : undefined;
}

function stackControls(
  cursors: readonly string[],
  change: (next: readonly string[]) => void,
  page: PageContinuation | undefined,
): CursorStackControls {
  const next = nextPageCursor(page);
  return {
    canGoBack: cursors.length > 0,
    ...(next === undefined ? {} : { nextCursor: next }),
    onBack: () => change(cursors.slice(0, -1)),
    onNext: (cursor) => change([...cursors, cursor]),
  };
}

/**
 * Cursors of the pages visited after the first one, kept in component state.
 * The stack is empty on the first page.
 */
export function useCursorStack() {
  const [cursors, setCursors] = useState<readonly string[]>([]);
  return {
    cursor: cursors.at(-1),
    reset: () => setCursors([]),
    controls: (page: PageContinuation | undefined) =>
      stackControls(cursors, setCursors, page),
  };
}

/**
 * Cursors of the pages visited after the first one, kept as repeated search
 * parameters next to the filters they were issued for, so any navigation
 * that changes the filters drops them together.
 */
export function useURLCursorStack({
  param = "cursor",
  navigateOptions = { preventScrollReset: true },
  onChange,
}: {
  param?: string;
  navigateOptions?: NavigateOptions;
  onChange?: () => void;
} = {}) {
  const [searchParams, setSearchParams] = useSearchParams();
  const cursors = searchParams.getAll(param);
  function change(nextCursors: readonly string[]): void {
    const next = new URLSearchParams(searchParams);
    next.delete(param);
    for (const value of nextCursors) next.append(param, value);
    setSearchParams(next, navigateOptions);
    onChange?.();
  }
  return {
    cursor: cursors.at(-1),
    reset: () => change([]),
    controls: (page: PageContinuation | undefined) =>
      stackControls(cursors, change, page),
  };
}
