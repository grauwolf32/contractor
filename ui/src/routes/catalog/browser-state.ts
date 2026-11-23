import { useLocation } from "react-router";
import { catalogReturnState } from "./navigation";

/** List context stays in the URL while an exact version is selected. */
export function useLibraryBrowser(base: string) {
  const location = useLocation();
  const params = new URLSearchParams(location.search);
  params.delete("usageCursor");
  params.delete("usagePage");
  const search = params.size ? `?${params}` : "";
  const listPath = `${base}${search}`;
  const label = base.endsWith("audit-presets")
    ? "All check types"
    : base.endsWith("agents")
      ? "All agents"
      : "All workflows";
  const fallback = { returnTo: listPath, returnLabel: label };
  const prior = catalogReturnState(location.state, fallback);
  const ownReturn =
    prior.returnTo.split("?")[0] === base && prior.returnLabel === label;
  const sourceState =
    ownReturn && prior !== fallback ? prior.returnState : location.state;
  const origin = catalogReturnState(sourceState, fallback);
  const context = origin.returnTo.split("?")[0] !== base ? origin : undefined;
  const current =
    typeof location.state === "object" && location.state !== null
      ? (location.state as Record<string, unknown>)
      : {};
  const rowState = {
    ...current,
    returnTo: listPath,
    returnLabel: label,
    returnState: sourceState,
  };
  const backState =
    current.catalogPagination === undefined
      ? sourceState
      : {
          ...(typeof sourceState === "object" && sourceState !== null
            ? sourceState
            : {}),
          catalogPagination: current.catalogPagination,
        };
  return { location, search, listPath, rowState, context, backState };
}
