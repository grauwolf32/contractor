import { useDocumentTitle } from "../../app/document-title";
import { ContextLink } from "../../app/context-navigation";
import { useQuery } from "@tanstack/react-query";
import { type FormEvent, useState } from "react";
import { Link, useSearchParams } from "react-router";

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
import { RefreshButton } from "../../app/refresh-button";

const EXCLUDED_SKILL_NAMESPACE = "skills";

export function ArtifactListRoute() {
  useDocumentTitle("Artifacts");
  const api = usePublicAPI();
  const [filters, setFilters] = useSearchParams();
  const namespace = filters.get("namespace") || undefined;
  const cursors: Array<string | undefined> = [
    undefined,
    ...filters.getAll("cursor"),
  ];
  function setCursors(
    update:
      | Array<string | undefined>
      | ((current: Array<string | undefined>) => Array<string | undefined>),
  ) {
    const nextCursors = typeof update === "function" ? update(cursors) : update;
    const next = new URLSearchParams(filters);
    next.delete("cursor");
    for (const cursor of nextCursors)
      if (cursor !== undefined) next.append("cursor", cursor);
    setFilters(next, { preventScrollReset: true });
  }
  const [filterError, setFilterError] = useState<string | null>(null);
  const [written, setWritten] = useState<ArtifactWriteResponse | null>(null);
  const [uploadOpen, setUploadOpen] = useState(false);
  const cursor = cursors.at(-1);
  const query = useQuery({
    queryKey: queryKeys.artifacts.list(
      namespace,
      cursor,
      EXCLUDED_SKILL_NAMESPACE,
    ),
    queryFn: () =>
      listArtifacts(api, {
        ...(namespace === undefined ? {} : { namespace }),
        excludeNamespace: EXCLUDED_SKILL_NAMESPACE,
        ...(cursor === undefined ? {} : { cursor }),
      }),
  });

  function applyFilter(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    const candidate = String(
      new FormData(event.currentTarget).get("namespaceFilter") ?? "",
    ).trim();
    if (candidate !== "" && !ARTIFACT_NAME_PATTERN.test(candidate)) {
      setFilterError("Namespace filter is not a valid Artifact name.");
      return;
    }
    setFilterError(null);
    const next = new URLSearchParams(filters);
    next.delete("cursor");
    if (candidate === "") next.delete("namespace");
    else next.set("namespace", candidate);
    setFilters(next, { preventScrollReset: true });
  }

  return (
    <section className="route-page artifact-page">
      <header className="route-header-row">
        <div>
          <h2>Artifacts</h2>
          <p className="lede">
            Inputs uploaded here can be updated; each Run pins one revision.
          </p>
        </div>
        <RefreshButton
          isFetching={query.isFetching}
          onRefresh={() => void query.refetch()}
          label="Refresh"
        />
      </header>

      <details
        className="artifact-create-disclosure"
        open={uploadOpen}
        onToggle={(event) => setUploadOpen(event.currentTarget.open)}
      >
        <summary>
          <span>Upload Artifact</span>
          <small>{uploadOpen ? "Close form" : "Create a new binding"}</small>
        </summary>
        <ArtifactWriteForm
          excludedNamespace={{
            namespace: EXCLUDED_SKILL_NAMESPACE,
            destination: "/catalog/skills",
            label: "Skills",
          }}
          onWritten={(result) => {
            setWritten(result);
            setCursors([undefined]);
            setUploadOpen(false);
          }}
        />
      </details>
      {written === null ? null : (
        <div className="notice notice-success" role="status">
          <strong>Artifact revision stored.</strong>
          <ContextLink
            returnLabel="Artifacts"
            to={`/artifacts/${encodeURIComponent(written.artifact.namespace)}/${encodeURIComponent(written.artifact.name)}?revision=${encodeURIComponent(written.artifact.revision)}`}
          >
            Open {written.artifact.namespace}/{written.artifact.name}@
            {written.artifact.revision}
          </ContextLink>
        </div>
      )}

      <div className="panel artifact-library">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Your library</p>
            <h3>Current bindings</h3>
            <small className="muted-copy">
              Skill packages live in the{" "}
              <Link to="/catalog/skills">Skills</Link> tab.
            </small>
          </div>
          <form className="inline-form" onSubmit={applyFilter}>
            <label>
              Namespace
              <input
                name="namespaceFilter"
                placeholder="all namespaces"
                key={namespace ?? ""}
                defaultValue={namespace ?? ""}
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
          <p className="loading-copy" role="status">
            Loading Artifact bindings…
          </p>
        ) : query.error !== null ? (
          <ErrorNotice
            error={query.error}
            context="Could not load Artifact bindings"
            onRetry={() => void query.refetch()}
            retryPending={query.isFetching}
          />
        ) : query.data.items.length === 0 ? (
          <div className="compact-empty">
            <strong>No Artifact bindings found.</strong>
            <p>Upload the first Workflow input above.</p>
          </div>
        ) : (
          <div className="table-scroll">
            <table className="responsive-table">
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
                    <td data-label="Binding">
                      <ContextLink
                        returnLabel="Artifacts"
                        to={`/artifacts/${encodeURIComponent(item.artifact.namespace)}/${encodeURIComponent(item.artifact.name)}`}
                      >
                        {item.artifact.namespace}/{item.artifact.name}
                      </ContextLink>
                    </td>
                    <td data-label="Current revision">
                      <code>{item.artifact.revision}</code>
                    </td>
                    <td data-label="Media type">{item.mediaType}</td>
                    <td data-label="Size">{formatBytes(item.size)}</td>
                    <td data-label="Created">
                      {formatTimestamp(item.createdAt)}
                    </td>
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
