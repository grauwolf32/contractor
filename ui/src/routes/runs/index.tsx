import { Link, Navigate, useLocation, useSearchParams } from "react-router";

import { TERMINAL_RUN_STATES } from "../../api/runs";
import { QueuePanel } from "../queue";
import { CompletedRunsPanel } from "./list";

export function RunsRoute() {
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

      <nav className="run-view-tabs" aria-label="Run views">
        <Link
          to="/runs"
          className={completed ? undefined : "active"}
          aria-current={completed ? undefined : "page"}
        >
          <strong>Queue</strong>
          <small>Active work</small>
        </Link>
        <Link
          to="/runs?view=completed"
          className={completed ? "active" : undefined}
          aria-current={completed ? "page" : undefined}
        >
          <strong>Completed</strong>
          <small>Results and history</small>
        </Link>
      </nav>

      {completed ? <CompletedRunsPanel /> : <QueuePanel />}
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
