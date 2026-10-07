import "./login.css";
import { Link } from "react-router";

import { useDocumentTitle } from "../app/document-title";

export function NotFoundRoute() {
  useDocumentTitle("Page not found");
  return (
    <main className="ops-state">
      <section className="ops-state-card" aria-labelledby="not-found-heading">
        <p className="ops-state-eyebrow">
          <span className="ops-state-code">404</span>
        </p>
        <h1 id="not-found-heading">Page not found</h1>
        <p>This address may be incomplete or the page may have moved.</p>
        <nav className="ops-state-actions" aria-label="Recovery">
          <Link className="ui-btn" data-variant="primary" to="/">
            Inbox
          </Link>
          <Link className="ui-btn" to="/projects">
            Projects
          </Link>
        </nav>
      </section>
    </main>
  );
}

/** Shown in place of the shell until the first route chunk has loaded. */
export function RouteChunkLoading() {
  return (
    <main className="ops-state" aria-live="polite">
      <div className="ops-state-progress">
        <span className="ops-spinner" aria-hidden="true" />
        <p>Loading workspace…</p>
      </div>
    </main>
  );
}
