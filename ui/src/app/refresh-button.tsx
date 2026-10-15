import type { ComponentProps } from "react";

import { Button, IconButton } from "../ui/button";
import { Icon } from "./icon";

/**
 * Shared manual refresh control. Icon-only by default (accessible name
 * "Refresh"); pass `label` for views without live updates. The icon spins and
 * the button is disabled and busy while `isFetching`.
 */
export function RefreshButton({
  isFetching,
  onRefresh,
  label,
  disabled = false,
  size,
  className,
}: {
  isFetching: boolean;
  onRefresh: () => void;
  label?: string;
  disabled?: boolean;
  size?: ComponentProps<typeof Button>["size"];
  className?: string;
}) {
  const shared = {
    size,
    pending: isFetching,
    disabled,
    onClick: onRefresh,
    className:
      className === undefined
        ? "refresh-button"
        : `refresh-button ${className}`,
    "data-fetching": isFetching || undefined,
  };
  return label === undefined ? (
    <IconButton {...shared} label="Refresh" icon={<Icon name="refresh" />} />
  ) : (
    <Button {...shared} icon={<Icon name="refresh" />}>
      {label}
    </Button>
  );
}
