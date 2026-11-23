import type { ComponentProps } from "react";

import { Button } from "../ui/button";

/** First, Previous and Next for a keyset-paged list. */
export function CursorControls({
  label,
  canGoBack,
  nextCursor,
  onBack,
  onNext,
  onFirst,
  size,
}: {
  label: string;
  canGoBack: boolean;
  nextCursor?: string;
  onBack: () => void;
  onNext: (cursor: string) => void;
  onFirst?: () => void;
  size?: ComponentProps<typeof Button>["size"];
}) {
  if (!canGoBack && nextCursor === undefined && onFirst === undefined) {
    return null;
  }
  return (
    <nav className="pagination" aria-label={label}>
      {onFirst === undefined ? null : (
        <Button size={size} onClick={onFirst}>
          First page
        </Button>
      )}
      <Button size={size} disabled={!canGoBack} onClick={onBack}>
        Previous
      </Button>
      <Button
        size={size}
        disabled={nextCursor === undefined}
        onClick={() => {
          if (nextCursor !== undefined) {
            onNext(nextCursor);
          }
        }}
      >
        Next
      </Button>
    </nav>
  );
}
