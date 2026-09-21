/**
 * Secondary "Load more" control for a bounded collection. Renders nothing
 * once every record is loaded.
 */
export function LoadMoreControl({
  shown,
  noun,
  truncated,
  loading,
  error = null,
  onLoadMore,
  label = "Load more",
}: {
  shown: number;
  noun: string;
  truncated: boolean;
  loading: boolean;
  error?: Error | null;
  onLoadMore: () => void;
  label?: string;
}) {
  if (!truncated && !loading && error === null) return null;
  return (
    <div className="audit-load-more">
      <p role="status">
        Showing {shown} of ≥{shown} {noun}
        {error === null ? "" : ` · Loading more failed: ${error.message}`}
      </p>
      <button
        className="secondary-button"
        type="button"
        disabled={loading}
        onClick={onLoadMore}
      >
        {loading ? "Loading more…" : error === null ? label : "Retry"}
      </button>
    </div>
  );
}
