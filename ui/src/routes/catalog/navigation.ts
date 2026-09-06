import type { Location } from "react-router";

export interface CatalogReturnState {
  returnTo: string;
  returnLabel: string;
  returnState?: unknown;
}

export function locationDestination(location: Location): string {
  return `${location.pathname}${location.search}${location.hash}`;
}

export function catalogReturnState(
  value: unknown,
  fallback: CatalogReturnState,
): CatalogReturnState {
  if (
    typeof value !== "object" ||
    value === null ||
    !("returnTo" in value) ||
    !("returnLabel" in value) ||
    typeof value.returnTo !== "string" ||
    typeof value.returnLabel !== "string" ||
    !value.returnTo.startsWith("/") ||
    value.returnTo.startsWith("//") ||
    value.returnTo.length > 4096 ||
    value.returnLabel.length === 0 ||
    value.returnLabel.length > 200
  ) {
    return fallback;
  }
  return {
    returnTo: value.returnTo,
    returnLabel: value.returnLabel,
    ...("returnState" in value ? { returnState: value.returnState } : {}),
  };
}
