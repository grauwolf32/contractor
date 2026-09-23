import { useEffect } from "react";

const APPLICATION_NAME = "Contractor";

export function documentTitle(title?: string | null): string {
  const trimmed = title?.trim();
  return trimmed ? `${trimmed} · ${APPLICATION_NAME}` : APPLICATION_NAME;
}

/**
 * Sets `document.title` for the current page; pass nothing for the bare app
 * name. The title falls back to the bare app name when the page unmounts, so
 * a route without its own title never keeps showing the previous page's.
 */
export function useDocumentTitle(title?: string | null): void {
  useEffect(() => {
    const value = documentTitle(title);
    document.title = value;
    return () => {
      // A title another page has set since is left alone.
      if (document.title === value) {
        document.title = APPLICATION_NAME;
      }
    };
  }, [title]);
}
