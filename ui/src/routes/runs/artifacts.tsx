import { useMutation, useQuery } from "@tanstack/react-query";
import { type FormEvent, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router";

import {
  ARTIFACT_NAME_PATTERN,
  ARTIFACT_REVISION_PATTERN,
  type ArtifactMetadata,
  type DownloadedArtifact,
} from "../../api/artifacts";
import { usePublicAPI } from "../../api/context";
import { queryKeys } from "../../api/query-keys";
import {
  downloadRunArtifact,
  getRunArtifactLineage,
  getRunArtifactMetadata,
  listRunArtifacts,
  listRunArtifactVersions,
  previewRunArtifact,
  RUN_ID_PATTERN,
  type RunStatus,
} from "../../api/runs";
import { getWorkflow } from "../../api/workflows";
import {
  CursorControls,
  ErrorNotice,
  formatBytes,
  formatTimestamp,
} from "../artifacts/common";
import { ArtifactPreviewPanel } from "../artifacts/preview";
import {
  missingOutputCopy,
  organizeRunOutputs,
  outputRole,
  parseWorkflowIdentity,
  requireWorkflowOutputs,
  type OutputEntry,
} from "./output-model";
import "./outputs.css";

function triggerDownload(downloaded: DownloadedArtifact): void {
  const objectURL = URL.createObjectURL(downloaded.blob);
  const anchor = document.createElement("a");
  anchor.href = objectURL;
  anchor.download = downloaded.filename;
  anchor.hidden = true;
  document.body.append(anchor);
  try {
    anchor.click();
  } finally {
    anchor.remove();
    URL.revokeObjectURL(objectURL);
  }
}

function RunOutputPreview({
  runId,
  entry,
  runState,
}: {
  runId: string;
  entry: OutputEntry;
  runState: RunStatus["state"];
}) {
  const api = usePublicAPI();
  const [requested, setRequested] = useState(false);
  const artifact = entry.artifact;
  const metadata = useQuery({
    queryKey: queryKeys.runs.artifactMetadata(
      runId,
      artifact?.namespace ?? "missing",
      artifact?.name ?? entry.slot,
      artifact?.revision,
    ),
    queryFn: async () => {
      if (artifact === undefined) {
        throw new Error("Run output is unavailable");
      }
      const exact = await getRunArtifactMetadata(api, {
        runId,
        namespace: artifact.namespace,
        name: artifact.name,
        revision: artifact.revision,
      });
      if (
        exact.artifact.namespace !== artifact.namespace ||
        exact.artifact.name !== artifact.name ||
        exact.artifact.revision !== artifact.revision
      ) {
        throw new Error(
          "Artifact metadata did not match the selected exact Run output",
        );
      }
      return exact;
    },
    enabled: requested && artifact !== undefined,
  });
  if (artifact === undefined) {
    return (
      <article className={`run-result-card run-result-${entry.kind}`}>
        <div className="run-result-heading">
          <div>
            <span className="run-result-role">{outputRole(entry)}</span>
            <h4>{entry.slot}</h4>
          </div>
          <span className="run-result-state">Unavailable</span>
        </div>
        <p className="run-result-missing">
          {entry.declaration === undefined
            ? "Run output is unavailable."
            : missingOutputCopy(entry.declaration, runState)}
        </p>
      </article>
    );
  }
  const detailPath = `/runs/${encodeURIComponent(runId)}/artifacts/${encodeURIComponent(artifact.namespace)}/${encodeURIComponent(artifact.name)}?revision=${encodeURIComponent(artifact.revision)}`;

  return (
    <article className={`run-result-card run-result-${entry.kind}`}>
      <div className="run-result-heading">
        <div>
          <span className="run-result-role">{outputRole(entry)}</span>
          <h4>{entry.slot}</h4>
          <code>
            {artifact.namespace}/{artifact.name}@{artifact.revision}
          </code>
        </div>
        {entry.kind === "primary" ? (
          <span className="run-result-state">Primary</span>
        ) : null}
      </div>
      <div className="run-result-actions">
        {requested && metadata.error === null ? null : (
          <button
            type="button"
            disabled={metadata.isPending && requested}
            onClick={() => {
              if (requested) {
                void metadata.refetch();
              } else {
                setRequested(true);
              }
            }}
          >
            {metadata.isPending && requested
              ? "Loading result…"
              : metadata.error === null
                ? "Preview result"
                : "Retry preview"}
          </button>
        )}
        <Link className="run-output-detail-link" to={detailPath}>
          Open {artifact.namespace}/{artifact.name}@{artifact.revision} exact
          details →
        </Link>
      </div>
      {requested ? (
        <div className="run-output-preview-body">
          {!requested || metadata.isPending ? (
            <p className="loading-copy">Loading exact output metadata…</p>
          ) : metadata.error !== null ? (
            <ErrorNotice error={metadata.error} />
          ) : (
            <ArtifactPreviewPanel
              key={metadata.data.artifact.revision}
              metadata={metadata.data}
              unavailableCopy="Inline preview is unavailable; exact original bytes remain available from artifact details."
              loadPreview={() => previewRunArtifact(api, runId, metadata.data)}
              loadOnMountKey={[
                "run-output-preview",
                runId,
                metadata.data.artifact.namespace,
                metadata.data.artifact.name,
                metadata.data.artifact.revision,
              ]}
            />
          )}
        </div>
      ) : null}
    </article>
  );
}

export function RunOutputGallery({
  run,
}: {
  run: Pick<RunStatus, "runId" | "workflow" | "state" | "outputs">;
}) {
  const api = usePublicAPI();
  const identity = parseWorkflowIdentity(run.workflow);
  const contract = useQuery({
    queryKey:
      identity === undefined
        ? ["workflows", "detail", "invalid", run.workflow]
        : queryKeys.workflows.detail(identity.name, identity.version),
    queryFn: async ({ signal }) => {
      if (identity === undefined) {
        throw new Error("Run has an invalid exact Workflow selector");
      }
      return getWorkflow(api, identity.name, identity.version, signal);
    },
    select: (workflow) => {
      if (identity === undefined) {
        throw new Error("Run has an invalid exact Workflow selector");
      }
      return requireWorkflowOutputs(workflow, identity);
    },
    enabled: identity !== undefined,
    staleTime: Number.POSITIVE_INFINITY,
  });
  const contractError =
    identity === undefined
      ? new Error("Run has an invalid exact Workflow selector")
      : contract.error;
  const entries = organizeRunOutputs(run.outputs, contract.data);
  const declaredCount = contract.data
    ? Object.keys(contract.data).length
    : undefined;
  return (
    <section className="run-output-gallery" id="run-outputs">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Frozen Workflow outputs</p>
          <h3>Results</h3>
        </div>
        <span>
          {Object.keys(run.outputs).length} present
          {declaredCount === undefined ? "" : ` · ${declaredCount} declared`}
        </span>
      </div>
      {contract.isPending && identity !== undefined ? (
        <p className="loading-copy">Loading exact Workflow output contract…</p>
      ) : null}
      {contract.data === undefined ? null : (
        <p className="run-output-intent">
          Primary marks the Workflow&apos;s intended presentation order, not the
          quality or success of a result.
        </p>
      )}
      {contractError === null ? null : (
        <div className="run-output-contract-error">
          <p>
            Output roles are unavailable. Present artifacts remain accessible
            without primary classification.
          </p>
          <ErrorNotice error={contractError} />
        </div>
      )}
      {entries.length === 0 ? (
        <div className="compact-empty">
          {contract.data === undefined
            ? "No Run outputs are currently available."
            : "This Workflow declares no output slots."}
        </div>
      ) : (
        <div className="run-output-list">
          {entries.map((entry) => (
            <RunOutputPreview
              key={`${entry.slot}:${entry.artifact?.namespace ?? "missing"}:${entry.artifact?.name ?? "missing"}:${entry.artifact?.revision ?? "missing"}`}
              runId={run.runId}
              runState={run.state}
              entry={entry}
            />
          ))}
        </div>
      )}
    </section>
  );
}

export function RunArtifactLibrary({ runId }: { runId: string }) {
  const api = usePublicAPI();
  const [namespaceDraft, setNamespaceDraft] = useState("");
  const [namespace, setNamespace] = useState<string | undefined>();
  const [filterError, setFilterError] = useState<string | undefined>();
  const [cursors, setCursors] = useState<Array<string | undefined>>([
    undefined,
  ]);
  const cursor = cursors.at(-1);
  const query = useQuery({
    queryKey: queryKeys.runs.artifacts(runId, namespace, cursor),
    queryFn: () =>
      listRunArtifacts(api, {
        runId,
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
    setFilterError(undefined);
    setNamespace(candidate === "" ? undefined : candidate);
    setCursors([undefined]);
  }

  return (
    <div className="panel run-artifact-library" id="run-artifacts">
      <div className="section-heading">
        <div>
          <p className="eyebrow">RunScope</p>
          <h3>Inputs, intermediate Artifacts, and frozen outputs</h3>
        </div>
        <form className="inline-form" onSubmit={applyFilter}>
          <label>
            Namespace
            <input
              value={namespaceDraft}
              placeholder="all namespaces"
              onChange={(event) => setNamespaceDraft(event.target.value)}
            />
          </label>
          <button className="secondary-button" type="submit">
            Apply
          </button>
        </form>
      </div>
      {filterError === undefined ? null : (
        <p className="form-error" role="alert">
          {filterError}
        </p>
      )}
      {query.isPending ? (
        <p className="loading-copy">Loading RunScope Artifacts…</p>
      ) : query.error !== null ? (
        <ErrorNotice error={query.error} />
      ) : query.data.items.length === 0 ? (
        <div className="compact-empty">No bindings match this view.</div>
      ) : (
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Binding</th>
                <th>Exact revision</th>
                <th>Media type</th>
                <th>Size</th>
                <th>Frozen</th>
                <th>Created</th>
              </tr>
            </thead>
            <tbody>
              {query.data.items.map((metadata) => (
                <tr
                  key={`${metadata.artifact.namespace}/${metadata.artifact.name}`}
                >
                  <td>
                    <Link
                      to={`/runs/${encodeURIComponent(runId)}/artifacts/${encodeURIComponent(metadata.artifact.namespace)}/${encodeURIComponent(metadata.artifact.name)}?revision=${encodeURIComponent(metadata.artifact.revision)}`}
                    >
                      {metadata.artifact.namespace}/{metadata.artifact.name}
                    </Link>
                  </td>
                  <td>
                    <code>{metadata.artifact.revision}</code>
                  </td>
                  <td>{metadata.mediaType}</td>
                  <td>{formatBytes(metadata.size)}</td>
                  <td>{metadata.frozen ? "yes" : "no"}</td>
                  <td>{formatTimestamp(metadata.createdAt)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <CursorControls
        label="Run Artifact pages"
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
  );
}

function RunArtifactActions({
  runId,
  metadata,
}: {
  runId: string;
  metadata: ArtifactMetadata;
}) {
  const api = usePublicAPI();
  const download = useMutation({
    mutationFn: () => downloadRunArtifact(api, runId, metadata),
    onSuccess: triggerDownload,
  });
  return (
    <div className="artifact-actions-grid run-artifact-actions">
      <ArtifactPreviewPanel
        metadata={metadata}
        unavailableCopy="Inline preview is unavailable; exact original bytes remain downloadable."
        loadPreview={() => previewRunArtifact(api, runId, metadata)}
      />
      <div className="panel artifact-download-panel">
        <p className="eyebrow">Original bytes</p>
        <h3>Download</h3>
        <p className="muted-copy">
          Fetches exact revision <code>{metadata.artifact.revision}</code>
          directly from Go Server.
        </p>
        {download.error === null ? null : (
          <ErrorNotice error={download.error} />
        )}
        <button
          type="button"
          disabled={download.isPending}
          onClick={() => download.mutate()}
        >
          {download.isPending ? "Downloading…" : "Download exact revision"}
        </button>
      </div>
    </div>
  );
}

function RunArtifactHistory({
  runId,
  metadata,
}: {
  runId: string;
  metadata: ArtifactMetadata;
}) {
  const api = usePublicAPI();
  const identity = metadata.artifact;
  const [versionCursors, setVersionCursors] = useState<
    Array<string | undefined>
  >([undefined]);
  const [lineageCursors, setLineageCursors] = useState<
    Array<string | undefined>
  >([undefined]);
  const versionCursor = versionCursors.at(-1);
  const lineageCursor = lineageCursors.at(-1);
  const versions = useQuery({
    queryKey: queryKeys.runs.artifactVersions(
      runId,
      identity.namespace,
      identity.name,
      versionCursor,
    ),
    queryFn: () =>
      listRunArtifactVersions(api, {
        runId,
        namespace: identity.namespace,
        name: identity.name,
        ...(versionCursor === undefined ? {} : { cursor: versionCursor }),
      }),
  });
  const lineage = useQuery({
    queryKey: queryKeys.runs.artifactLineage(
      runId,
      identity.namespace,
      identity.name,
      identity.revision,
      lineageCursor,
    ),
    queryFn: () =>
      getRunArtifactLineage(api, {
        runId,
        namespace: identity.namespace,
        name: identity.name,
        revision: identity.revision,
        ...(lineageCursor === undefined ? {} : { cursor: lineageCursor }),
      }),
  });
  return (
    <div className="artifact-history-grid">
      <div className="panel">
        <p className="eyebrow">Immutable history</p>
        <h3>Versions</h3>
        {versions.isPending ? (
          <p className="loading-copy">Loading versions…</p>
        ) : versions.error !== null ? (
          <ErrorNotice error={versions.error} />
        ) : versions.data.items.length === 0 ? (
          <div className="compact-empty">No versions found.</div>
        ) : (
          <ul className="version-list">
            {versions.data.items.map((item) => (
              <li
                key={item.artifact.revision}
                className={
                  item.artifact.revision === identity.revision
                    ? "selected"
                    : undefined
                }
              >
                <Link
                  to={`?revision=${encodeURIComponent(item.artifact.revision)}`}
                >
                  <code>{item.artifact.revision}</code>
                  <span>{formatBytes(item.size)}</span>
                  <span>{formatTimestamp(item.createdAt)}</span>
                  {item.current ? <strong>current</strong> : null}
                </Link>
              </li>
            ))}
          </ul>
        )}
        <CursorControls
          label="Run Artifact version pages"
          canGoBack={versionCursors.length > 1}
          {...(versions.data?.page.hasMore === true &&
          versions.data.page.nextCursor !== undefined
            ? { nextCursor: versions.data.page.nextCursor }
            : {})}
          onBack={() =>
            setVersionCursors((current) =>
              current.slice(0, Math.max(1, current.length - 1)),
            )
          }
          onNext={(next) => setVersionCursors((current) => [...current, next])}
        />
      </div>
      <div className="panel">
        <p className="eyebrow">Provenance</p>
        <h3>Lineage</h3>
        {lineage.isPending ? (
          <p className="loading-copy">Loading lineage…</p>
        ) : lineage.error !== null ? (
          <ErrorNotice error={lineage.error} />
        ) : lineage.data.items.length === 0 ? (
          <div className="compact-empty">
            No lineage edges reference this exact revision.
          </div>
        ) : (
          <ol className="lineage-list">
            {lineage.data.items.map((edge, index) => (
              <li
                key={`${edge.kind}-${edge.createdAt}-${edge.source.revision}-${index}`}
              >
                <strong>{edge.kind.replaceAll("_", " ")}</strong>
                <span>
                  {edge.sourceScope}: {edge.source.namespace}/{edge.source.name}
                  @{edge.source.revision}
                </span>
                <span aria-hidden="true">→</span>
                <span>
                  {edge.targetScope}: {edge.target.namespace}/{edge.target.name}
                  @{edge.target.revision}
                </span>
                {edge.runId === undefined ? null : (
                  <small>Run {edge.runId}</small>
                )}
                {edge.stageExecutionId === undefined ? null : (
                  <small>Stage execution {edge.stageExecutionId}</small>
                )}
              </li>
            ))}
          </ol>
        )}
        <CursorControls
          label="Run Artifact lineage pages"
          canGoBack={lineageCursors.length > 1}
          {...(lineage.data?.page.hasMore === true &&
          lineage.data.page.nextCursor !== undefined
            ? { nextCursor: lineage.data.page.nextCursor }
            : {})}
          onBack={() =>
            setLineageCursors((current) =>
              current.slice(0, Math.max(1, current.length - 1)),
            )
          }
          onNext={(next) => setLineageCursors((current) => [...current, next])}
        />
      </div>
    </div>
  );
}

export function RunArtifactDetailRoute() {
  const api = usePublicAPI();
  const { runId = "", namespace = "", name = "" } = useParams();
  const [searchParams] = useSearchParams();
  const revision = searchParams.get("revision") ?? undefined;
  const valid =
    RUN_ID_PATTERN.test(runId) &&
    ARTIFACT_NAME_PATTERN.test(namespace) &&
    ARTIFACT_NAME_PATTERN.test(name) &&
    (revision === undefined || ARTIFACT_REVISION_PATTERN.test(revision));
  const query = useQuery({
    queryKey: queryKeys.runs.artifactMetadata(runId, namespace, name, revision),
    queryFn: () =>
      getRunArtifactMetadata(api, {
        runId,
        namespace,
        name,
        ...(revision === undefined ? {} : { revision }),
      }),
    enabled: valid,
  });
  if (!valid) {
    return (
      <section className="route-page">
        <ErrorNotice error={new Error("Run Artifact route is invalid")} />
        <Link to="/runs">Return to Runs</Link>
      </section>
    );
  }
  return (
    <section className="route-page artifact-page runs-page">
      <header className="route-header-row">
        <div>
          <Link className="back-link" to={`/runs/${encodeURIComponent(runId)}`}>
            ← Run {runId}
          </Link>
          <p className="eyebrow">Exact RunScope binding</p>
          <h2>
            {namespace}/{name}
          </h2>
          <p className="lede">
            {revision === undefined
              ? "Current authoritative binding"
              : "Selected immutable historical revision"}
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
      {query.isPending ? (
        <p className="loading-copy">Loading Run Artifact metadata…</p>
      ) : query.error !== null ? (
        <ErrorNotice error={query.error} />
      ) : (
        <>
          <dl className="metadata-grid panel">
            <div>
              <dt>Exact revision</dt>
              <dd>
                <code>{query.data.artifact.revision}</code>
              </dd>
            </div>
            <div>
              <dt>Status</dt>
              <dd>{query.data.current ? "current" : "historical"}</dd>
            </div>
            <div>
              <dt>Media type</dt>
              <dd>{query.data.mediaType}</dd>
            </div>
            <div>
              <dt>Size</dt>
              <dd>{formatBytes(query.data.size)}</dd>
            </div>
            <div>
              <dt>Created</dt>
              <dd>{formatTimestamp(query.data.createdAt)}</dd>
            </div>
            <div>
              <dt>Frozen</dt>
              <dd>{query.data.frozen ? "yes" : "no"}</dd>
            </div>
          </dl>
          <RunArtifactActions
            key={`actions-${query.data.artifact.revision}`}
            runId={runId}
            metadata={query.data}
          />
          <RunArtifactHistory
            key={`history-${query.data.artifact.revision}`}
            runId={runId}
            metadata={query.data}
          />
        </>
      )}
    </section>
  );
}
