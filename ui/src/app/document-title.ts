import { useEffect } from "react";

const APPLICATION_NAME = "Contractor";

export function documentTitle(title?: string | null): string {
  const trimmed = title?.trim();
  return trimmed ? `${trimmed} · ${APPLICATION_NAME}` : APPLICATION_NAME;
}

/** Sets `document.title` for the current page; pass nothing for the bare app name. */
export function useDocumentTitle(title?: string | null): void {
  useEffect(() => {
    document.title = documentTitle(title);
  }, [title]);
}
