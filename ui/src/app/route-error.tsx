import { Link, useLocation, useRouteError } from "react-router";

function isChunkLoadError(error: unknown): boolean {
  return (
    error instanceof Error &&
    /failed to fetch dynamically imported module|importing a module script failed|error loading dynamically imported module|failed to load module script/i.test(
      error.message,
    )
  );
}

export function RouteErrorPanel() {
  const error = useRouteError();
  const location = useLocation();
  const chunkFailed = isChunkLoadError(error);
  const reloadPath = `${location.pathname}${location.search}${location.hash}`;
  return (
    <section className="panel centered-state" role="alert">
      <h2>
        {chunkFailed ? "This page needs a reload" : "This page could not open"}
      </h2>
      <p>
        {chunkFailed
          ? "The page files may have changed. Reload to get the current version."
          : "An unexpected error prevented this page from opening."}
      </p>
      <div className="project-form-actions">
        <a className="primary-button" href={reloadPath}>
          Reload application
        </a>
        <Link className="secondary-button" to="/">
          Go to Inbox
        </Link>
      </div>
    </section>
  );
}
