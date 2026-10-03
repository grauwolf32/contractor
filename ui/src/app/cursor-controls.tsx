export function CursorControls({
  label,
  canGoBack,
  nextCursor,
  onBack,
  onNext,
  onFirst,
}: {
  label: string;
  canGoBack: boolean;
  nextCursor?: string;
  onBack: () => void;
  onNext: (cursor: string) => void;
  onFirst?: () => void;
}) {
  if (!canGoBack && nextCursor === undefined && onFirst === undefined) {
    return null;
  }
  return (
    <nav className="pagination" aria-label={label}>
      {onFirst === undefined ? null : (
        <button className="secondary-button" type="button" onClick={onFirst}>
          First page
        </button>
      )}
      <button
        className="secondary-button"
        type="button"
        disabled={!canGoBack}
        onClick={onBack}
      >
        Previous
      </button>
      <button
        className="secondary-button"
        type="button"
        disabled={nextCursor === undefined}
        onClick={() => {
          if (nextCursor !== undefined) {
            onNext(nextCursor);
          }
        }}
      >
        Next
      </button>
    </nav>
  );
}
