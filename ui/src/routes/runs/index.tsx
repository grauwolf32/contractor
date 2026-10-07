import "./runs.css";
import { useId } from "react";
import { Link, useSearchParams } from "react-router";

import { TERMINAL_RUN_STATES } from "../../api/runs";
import { useDocumentTitle } from "../../app/document-title";
import { QueuePanel } from "../queue";
import { useQueueAdmission } from "./admission";
import { CompletedRunsPanel } from "./list";

/**
 * The one Runs destination (S18): Queue by default, Completed at
 * ?view=completed. A terminal ?state=… without a view opens Completed.
 */
export function RunsRoute() {
  const [searchParams] = useSearchParams();
  const titleId = useId();
  const requestedView = searchParams.get("view");
  const requestedState = searchParams.get("state");
  const terminalStateDeepLink = TERMINAL_RUN_STATES.some(
    (state) => state === requestedState,
  );
  const completed =
    requestedView === "completed" ||
    (requestedView === null && terminalStateDeepLink);
  useDocumentTitle(completed ? "Completed · Runs" : "Queue · Runs");
  const admission = useQueueAdmission();

  return (
    <div className="runs-page">
      <section className="runs-surface" aria-labelledby={titleId}>
        <header className="runs-head">
          <div className="runs-head-text">
            <h1 id={titleId} className="runs-title">
              Runs
            </h1>
            <p className="runs-subtitle">
              Workflow Runs: active work waits in the Queue, finished Runs and
              their results are in Completed.
            </p>
          </div>
          {admission.control}
        </header>
        {admission.notice === null ? null : (
          <div className="runs-head-notice">{admission.notice}</div>
        )}
        <nav className="runs-views" aria-label="Run views">
          <Link
            to="/runs"
            className="runs-view-link"
            aria-current={completed ? undefined : "page"}
          >
            Queue
          </Link>
          <Link
            to="/runs?view=completed"
            className="runs-view-link"
            aria-current={completed ? "page" : undefined}
          >
            Completed
          </Link>
        </nav>
        {completed ? <CompletedRunsPanel /> : <QueuePanel />}
      </section>
    </div>
  );
}
