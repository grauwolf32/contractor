import { useEffect, useState } from "react";
import { useNavigate, useSearchParams } from "react-router";

const SEARCH_DEBOUNCE_MS = 250;

function pageNumber(raw: string | null): number {
  if (raw === null || !/^[1-9][0-9]*$/.test(raw)) return 1;
  const parsed = Number(raw);
  return Number.isSafeInteger(parsed) ? parsed : 1;
}

export function useCatalogQueryState() {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const serializedSearchParams = searchParams.toString();
  const committedSearch = searchParams.get("q") ?? "";
  const cursor = searchParams.get("cursor") ?? undefined;
  const page = pageNumber(searchParams.get("page"));
  const [draft, setDraft] = useState({
    base: committedSearch,
    value: committedSearch,
  });
  const draftSearch =
    draft.base === committedSearch ? draft.value : committedSearch;

  useEffect(() => {
    if (draftSearch === committedSearch) return;
    const timer = window.setTimeout(() => {
      const next = new URLSearchParams(serializedSearchParams);
      const normalized = draftSearch.trim();
      if (normalized === "") next.delete("q");
      else next.set("q", normalized);
      next.delete("cursor");
      next.delete("page");
      setSearchParams(next, { replace: true });
    }, SEARCH_DEBOUNCE_MS);
    return () => window.clearTimeout(timer);
  }, [committedSearch, draftSearch, serializedSearchParams, setSearchParams]);

  function changeDraftSearch(value: string): void {
    if (Array.from(value).length <= 200) {
      setDraft({ base: committedSearch, value });
    }
  }

  function nextPage(nextCursor: string): void {
    const next = new URLSearchParams(searchParams);
    next.set("cursor", nextCursor);
    next.set("page", String(page + 1));
    setSearchParams(next);
  }

  return {
    committedSearch,
    cursor,
    draftSearch,
    page,
    canGoBack: cursor !== undefined && page > 1,
    changeDraftSearch,
    nextPage,
    previousPage: () => void navigate(-1),
  };
}
