import { useEffect, useState } from "react";
import { useLocation, useSearchParams } from "react-router";

import {
  useCatalogCursorState,
  withoutCatalogPagination,
} from "./cursor-state";

const SEARCH_DEBOUNCE_MS = 250;

export function useCatalogQueryState() {
  const location = useLocation();
  const pagination = useCatalogCursorState();
  const [searchParams, setSearchParams] = useSearchParams();
  const serializedSearchParams = searchParams.toString();
  const committedSearch = searchParams.get("q") ?? "";
  const [draft, setDraft] = useState({
    base: committedSearch,
    value: committedSearch,
  });
  // The draft belongs to the committed query it was typed against. When the
  // URL query changes, a draft that normalizes to the new query is this
  // draft's own debounced commit and keeps its text as typed (for example a
  // trailing space before the next word); any other change replaces it.
  let current = draft;
  if (draft.base !== committedSearch) {
    current = {
      base: committedSearch,
      value:
        draft.value.trim() === committedSearch ? draft.value : committedSearch,
    };
    setDraft(current);
  }
  const draftSearch = current.value;

  useEffect(() => {
    if (
      draftSearch === committedSearch ||
      draftSearch.trim() === committedSearch
    ) {
      return;
    }
    const timer = window.setTimeout(() => {
      const next = new URLSearchParams(serializedSearchParams);
      const normalized = draftSearch.trim();
      if (normalized === "") next.delete("q");
      else next.set("q", normalized);
      next.delete("cursor");
      next.delete("page");
      setSearchParams(next, {
        replace: true,
        state: withoutCatalogPagination(location.state),
      });
    }, SEARCH_DEBOUNCE_MS);
    return () => window.clearTimeout(timer);
  }, [
    committedSearch,
    draftSearch,
    serializedSearchParams,
    setSearchParams,
    location.state,
  ]);

  function changeDraftSearch(value: string): void {
    if (Array.from(value).length <= 200) {
      setDraft({ base: committedSearch, value });
    }
  }

  return {
    ...pagination,
    committedSearch,
    draftSearch,
    changeDraftSearch,
  };
}
