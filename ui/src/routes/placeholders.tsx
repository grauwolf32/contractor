import { useDocumentTitle } from "../app/document-title";

export function NotFoundRoute() {
  useDocumentTitle("Page not found");
  return (
    <main className="centered-state">
      <p className="eyebrow">404</p>
      <h1>Page not found</h1>
      <p>This address may be incomplete or the page may have moved.</p>
      <nav className="action-row" aria-label="Recovery">
        <a href="/">Home</a>
        <a href="/projects">Projects</a>
      </nav>
    </main>
  );
}

/** Shown in place of the shell until the first route chunk has loaded. */
export function RouteChunkLoading() {
  return (
    <main className="centered-state" aria-live="polite">
      <span className="spinner" aria-hidden="true" />
      <p>Loading workspace…</p>
    </main>
  );
}
