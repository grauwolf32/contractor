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

/** Runtime configuration hub under Operations (RuntimeConfig versions, bindings, credentials). */
export const RUNTIME_CONFIGURATION_PATH = "/operations/configuration";

/** Exact RuntimeConfig version page under the Operations configuration hub. */
export function runtimeConfigVersionPath(ref: {
  name: string;
  version: string;
}): string {
  return `${RUNTIME_CONFIGURATION_PATH}/${encodeURIComponent(ref.name)}/${encodeURIComponent(ref.version)}`;
}

/**
 * Legacy `/runs/configuration…` locations map onto the Operations hub with the
 * rest of the path, the query and the fragment preserved.
 */
export function legacyRunConfigurationDestination(location: Location): string {
  const rest = location.pathname.replace(/^\/runs\/configuration(?=\/|$)/, "");
  return `${RUNTIME_CONFIGURATION_PATH}${rest}${location.search}${location.hash}`;
}
