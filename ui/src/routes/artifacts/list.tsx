import { useQuery } from "@tanstack/react-query";
import { type FormEvent, useState } from "react";
import { Link } from "react-router";

import {
  ARTIFACT_NAME_PATTERN,
  listArtifacts,
  type ArtifactWriteResponse,
} from "../../api/artifacts";
import { usePublicAPI } from "../../api/context";
import { queryKeys } from "../../api/query-keys";
import {
  ArtifactWriteForm,
  CursorControls,
  ErrorNotice,
  formatBytes,
  formatTimestamp,
} from "./common";

export function ArtifactListRoute() {
  const api = usePublicAPI();
  const [namespaceDraft, setNamespaceDraft] = useState("");
  const [namespace, setNamespace] = useState<string | undefined>();
  const [cursors, setCursors] = useState<Array<string | undefined>>([
    undefined,
  ]);
  const [filterError, setFilterError] = useState<string | null>(null);
  const [written, setWritten] = useState<ArtifactWriteResponse | null>(null);
  const cursor = cursors.at(-1);
  const query = useQuery({
    queryKey: queryKeys.artifacts.list(namespace, cursor),
    queryFn: () =>
      listArtifacts(api, {
        ...(namespace === undefined ? {} : { namespace }),
        ...(cursor === undefined ? {} : { cursor }),
      }),
  });

  function applyFilter(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    const candidate = namespaceDraft.trim();
    if (candidate !== "" && !ARTIFACT_NAME_PATTERN.test(candidate)) {
      setFilterError("Namespace filter is not a valid Artifact name.");
      return;
    }
    setFilterError(null);
    setNamespace(candidate === "" ? undefined : candidate);
    setCursors([undefined]);
  }

  return (
    <section className="route-page artifact-page">
      <header className="route-header-row">
        <div>
          <p className="eyebrow">Exact input library</p>
          <h2>Artifacts</h2>
          <p className="lede">
            User-scoped inputs stay mutable here. Workflow Runs bind one exact
            immutable revision.
          </p>
        </div>
        <button
          className="secondary-button"
          type="button"
          disabled={query.isFetching}
          onClick={() => void query.refetch()}
        >
          {query.isFetching ? "Refreshing…" : "Refresh"}
        </button>
      </header>

      <ArtifactWriteForm
        onWritten={(result) => {
          setWritten(result);
          setCursors([undefined]);
        }}
      />
      {written === null ? null : (
        <div className="notice notice-success" role="status">
          <strong>Artifact revision stored.</strong>
          <Link
            to={`/artifacts/${encodeURIComponent(written.artifact.namespace)}/${encodeURIComponent(written.artifact.name)}?revision=${encodeURIComponent(written.artifact.revision)}`}
          >
            Open {written.artifact.namespace}/{written.artifact.name}@
            {written.artifact.revision}
          </Link>
        </div>
      )}

      <div className="panel artifact-library">
        <div className="section-heading">
          <div>
            <p className="eyebrow">UserScope</p>
            <h3>Current bindings</h3>
          </div>
          <form className="inline-form" onSubmit={applyFilter}>
            <label>
              Namespace
              <input
                name="namespaceFilter"
                placeholder="all namespaces"
                value={namespaceDraft}
                onChange={(event) => setNamespaceDraft(event.target.value)}
              />
            </label>
            <button className="secondary-button" type="submit">
              Apply
            </button>
          </form>
        </div>
        {filterError === null ? null : (
          <p className="form-error" role="alert">
            {filterError}
          </p>
        )}
        {query.isPending ? (
          <p className="loading-copy" aria-live="polite">
            Loading Artifact bindings…
          </p>
        ) : query.error !== null ? (
          <ErrorNotice error={query.error} />
        ) : query.data.items.length === 0 ? (
          <div className="compact-empty">
            <strong>No Artifact bindings found.</strong>
            <p>Upload the first exact Workflow input above.</p>
          </div>
        ) : (
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Binding</th>
                  <th>Current revision</th>
                  <th>Media type</th>
                  <th>Size</th>
                  <th>Created</th>
                </tr>
              </thead>
              <tbody>
                {query.data.items.map((item) => (
                  <tr key={`${item.artifact.namespace}/${item.artifact.name}`}>
                    <td>
                      <Link
                        to={`/artifacts/${encodeURIComponent(item.artifact.namespace)}/${encodeURIComponent(item.artifact.name)}`}
                      >
                        {item.artifact.namespace}/{item.artifact.name}
                      </Link>
                    </td>
                    <td>
                      <code>{item.artifact.revision}</code>
                    </td>
                    <td>{item.mediaType}</td>
                    <td>{formatBytes(item.size)}</td>
                    <td>{formatTimestamp(item.createdAt)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <CursorControls
          label="Artifact binding pages"
          canGoBack={cursors.length > 1}
          {...(query.data?.page.hasMore === true &&
          query.data.page.nextCursor !== undefined
            ? { nextCursor: query.data.page.nextCursor }
            : {})}
          onBack={() =>
            setCursors((current) =>
              current.slice(0, Math.max(1, current.length - 1)),
            )
          }
          onNext={(next) => setCursors((current) => [...current, next])}
        />
      </div>
    </section>
  );
}
