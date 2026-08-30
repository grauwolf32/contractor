import { Navigate, Outlet, useLocation } from "react-router";

import { useSession } from "../auth/session";

export function AuthenticatedRoute() {
  const { session, error, isLoading } = useSession();
  const location = useLocation();

  if (isLoading) {
    return (
      <main className="centered-state" aria-live="polite">
        <span className="spinner" aria-hidden="true" />
        <p>Checking Server session…</p>
      </main>
    );
  }
  if (error !== null) {
    return (
      <main className="centered-state">
        <p className="eyebrow">Connection error</p>
        <h1>Contractor Server is not compatible or unavailable</h1>
        <p role="alert">{error.message}</p>
      </main>
    );
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
  return <Outlet />;
}
