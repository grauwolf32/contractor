import { Link, Navigate, useLocation, useSearchParams } from "react-router";

import { TERMINAL_RUN_STATES } from "../../api/runs";
import { legacyRunConfigurationDestination } from "../../app/navigation";
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

      <nav
        className="operations-navigation section-navigation"
        aria-label="Run views"
      >
        <Link
          to="/runs"
          className={completed ? undefined : "active"}
          aria-current={completed ? undefined : "page"}
        >
          Queue
        </Link>
        <Link
          to="/runs?view=completed"
          className={completed ? "active" : undefined}
          aria-current={completed ? "page" : undefined}
        >
          Completed
        </Link>
      </nav>

      {completed ? <CompletedRunsPanel /> : <QueuePanel />}
    </section>
  );
}

/**
 * `/runs/configuration…` moved to Operations → Configuration; old links and
 * bookmarks land on the same version with query and fragment preserved.
 */
export function LegacyRunConfigurationRedirect() {
  const location = useLocation();
  return <Navigate replace to={legacyRunConfigurationDestination(location)} />;
}
