import { Icon } from "./icon";

/**
 * Shared manual refresh control. Icon-only by default (accessible name
 * "Refresh"); pass `label` for views without live updates. The icon spins and
 * the button is disabled while `isFetching`.
 */
export function RefreshButton({
  isFetching,
  onRefresh,
  label,
  disabled = false,
  className = "",
}: {
  isFetching: boolean;
  onRefresh: () => void;
  label?: string;
  disabled?: boolean;
  className?: string;
}) {
  const iconOnly = label === undefined;
  return (
    <button
      type="button"
      className={`secondary-button refresh-button ${iconOnly ? "icon-button" : ""} ${className}`.trim()}
      aria-label={iconOnly ? "Refresh" : undefined}
      title={iconOnly ? "Refresh" : undefined}
      aria-busy={isFetching || undefined}
      data-fetching={isFetching || undefined}
      disabled={disabled || isFetching}
      onClick={onRefresh}
    >
      <Icon name="refresh" />
      {iconOnly ? null : <span>{label}</span>}
    </button>
  );
}
