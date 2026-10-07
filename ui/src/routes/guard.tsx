import "./login.css";
import { useIsFetching, useQueryClient } from "@tanstack/react-query";
import { Navigate, Outlet, useLocation } from "react-router";

import { APICompatibilityError } from "../api/error";
import { queryKeys } from "../api/query-keys";
import { useSession } from "../auth/session";
import { ErrorNotice } from "../app/error-notice";
import { SessionConnectionError } from "./session-error";

export function AuthenticatedRoute() {
  const { session, error, isLoading } = useSession();
  const location = useLocation();
  const queryClient = useQueryClient();
  const refreshing =
    useIsFetching({ queryKey: queryKeys.session, exact: true }) > 0;

  if (isLoading) {
    return (
      <main className="ops-state" aria-live="polite">
        <div className="ops-state-progress">
          <span className="ops-spinner" aria-hidden="true" />
          <p>Checking Server session…</p>
        </div>
      </main>
    );
  }
  // A failed background refresh keeps the cached session usable; only a
  // missing session or an incompatible Server blocks the application.
  if (
    error !== null &&
    (session === null ||
      session === undefined ||
      error instanceof APICompatibilityError)
  ) {
    return <SessionConnectionError error={error} />;
  }
  if (session === null || session === undefined) {
    return (
      <Navigate
        to="/login"
        replace
        state={{ from: `${location.pathname}${location.search}` }}
      />
    );
  }
  return (
    <>
      {error === null ? null : (
        <div className="ops-session-refresh">
          <ErrorNotice
            error={error}
            context="Could not refresh the Server session"
            onRetry={() =>
              void queryClient.refetchQueries({
                queryKey: queryKeys.session,
                exact: true,
              })
            }
            retryPending={refreshing}
          />
        </div>
      )}
      <Outlet />
    </>
  );
}
