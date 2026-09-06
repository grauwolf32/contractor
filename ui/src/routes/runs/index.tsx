import {
  Link,
  Navigate,
  Outlet,
  useLocation,
  useSearchParams,
} from "react-router";

import { TERMINAL_RUN_STATES } from "../../api/runs";
import { useSession } from "../../auth/session";
import { QueuePanel } from "../queue";
import { CompletedRunsPanel } from "./list";

export function RunsRoute() {
  const { pathname } = useLocation();
  const { session } = useSession();
  const configuration = pathname.startsWith("/runs/configuration");
  const [searchParams] = useSearchParams();
  const requestedView = searchParams.get("view");
  const requestedState = searchParams.get("state");
  const terminalStateDeepLink = TERMINAL_RUN_STATES.some(
    (state) => state === requestedState,
  );
  const completed =
    requestedView === "completed" ||
    (requestedView === null && terminalStateDeepLink);

  return (
    <section className="route-page runs-page">
      <header className="route-header-row runs-header">
        <div>
          <p className="eyebrow">Workflow execution</p>
          <h2>Runs</h2>
          <p className="lede">
            Follow active work and inspect completed results.
          </p>
        </div>
      </header>

      <nav className="operations-navigation" aria-label="Run views">
        <Link
          to="/runs"
          className={completed || configuration ? undefined : "active"}
          aria-current={completed || configuration ? undefined : "page"}
        >
          Queue
        </Link>
        <Link
          to="/runs?view=completed"
          className={completed && !configuration ? "active" : undefined}
          aria-current={completed && !configuration ? "page" : undefined}
        >
          Completed
        </Link>
        {session?.principal.capabilities.includes("operations") ? (
          <Link
            to="/runs/configuration"
            className={configuration ? "active" : undefined}
            aria-current={configuration ? "page" : undefined}
          >
            Configuration
          </Link>
        ) : null}
      </nav>

      {configuration ? (
        <Outlet />
      ) : completed ? (
        <CompletedRunsPanel />
      ) : (
        <QueuePanel />
      )}
    </section>
  );
}

export function LegacyQueueRedirect() {
  const location = useLocation();
  return (
    <Navigate
      replace
      to={{ pathname: "/runs", search: location.search, hash: location.hash }}
    />
  );
}
