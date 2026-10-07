import { useQuery } from "@tanstack/react-query";
import { useId } from "react";
import { Link, useLocation } from "react-router";

import type { ArtifactArchiveScope } from "../../api/artifact-archive";
import type { ArtifactMetadata, ExactArtifactRef } from "../../api/artifacts";
import { usePublicAPI } from "../../api/context";
import { artifactScopeKeys } from "../../api/query-keys";
import { scopedArtifactAPI } from "../../api/scoped-artifacts";
import { CursorControls } from "../../app/cursor-controls";
import { formatBytes } from "../../app/format";
import { useCursorStack } from "../../app/pagination";
import { QueryView } from "../../app/query-view";
import { RecordedTime } from "../../app/recorded-time";
import "./materials.css";

function ExactRef({
  scope,
  value,
}: {
  scope: string;
  value: ExactArtifactRef;
}) {
  return (
    <span className="materials-mono materials-wrap">
      {scope}: {value.namespace}/{value.name}@{value.revision}
    </span>
  );
}

/**
 * Every version of a binding (newest first, paged) and the lineage of the
 * shown revision: where it came from and where Runs took it.
 */
export function ArtifactHistory({
  scope,
  metadata,
}: {
  scope: ArtifactArchiveScope;
  metadata: ArtifactMetadata;
}) {
  const location = useLocation();
  const api = usePublicAPI();
  const artifacts = scopedArtifactAPI(api, scope);
  const keys = artifactScopeKeys(scope);
  const versionsHeading = useId();
  const lineageHeading = useId();
  const versionPages = useCursorStack();
  const lineagePages = useCursorStack();
  const identity = metadata.artifact;
  const versions = useQuery({
    queryKey: keys.versions(
      identity.namespace,
      identity.name,
      versionPages.cursor,
    ),
    queryFn: () =>
      artifacts.versions({
        namespace: identity.namespace,
        name: identity.name,
        ...(versionPages.cursor === undefined
          ? {}
          : { cursor: versionPages.cursor }),
      }),
  });
  const lineage = useQuery({
    queryKey: keys.lineage(
      identity.namespace,
      identity.name,
      identity.revision,
      lineagePages.cursor,
    ),
    queryFn: () =>
      artifacts.lineage({
        namespace: identity.namespace,
        name: identity.name,
        revision: identity.revision,
        ...(lineagePages.cursor === undefined
          ? {}
          : { cursor: lineagePages.cursor }),
      }),
  });

  return (
    <div className="materials-history">
      <section aria-labelledby={versionsHeading}>
        <h3 id={versionsHeading} className="materials-subtitle">
          Versions
        </h3>
        <QueryView
          query={versions}
          loading={
            <p className="materials-loading" role="status">
              Loading versions…
            </p>
          }
          onRetry={() => void versions.refetch()}
          isEmpty={(page) => page.items.length === 0}
          empty={<p className="materials-quiet">No versions found.</p>}
        >
          {(page) => (
            <ol className="materials-versions">
              {page.items.map((item) => {
                const shown = item.artifact.revision === identity.revision;
                return (
                  <li key={item.artifact.revision}>
                    <Link
                      to={`?revision=${encodeURIComponent(item.artifact.revision)}`}
                      state={location.state}
                      aria-current={shown ? "page" : undefined}
                    >
                      <code>{item.artifact.revision}</code>
                      <span>{formatBytes(item.size)}</span>
                      <RecordedTime value={item.createdAt} />
                      {item.current ? (
                        <span className="materials-badge">current</span>
                      ) : null}
                    </Link>
                  </li>
                );
              })}
            </ol>
          )}
        </QueryView>
        <CursorControls
          label="Version pages"
          {...versionPages.controls(versions.data?.page)}
        />
      </section>

      <section aria-labelledby={lineageHeading}>
        <h3 id={lineageHeading} className="materials-subtitle">
          Lineage
        </h3>
        <QueryView
          query={lineage}
          loading={
            <p className="materials-loading" role="status">
              Loading lineage…
            </p>
          }
          onRetry={() => void lineage.refetch()}
          isEmpty={(page) => page.items.length === 0}
          empty={
            <p className="materials-quiet">
              No lineage edges reference this revision.
            </p>
          }
        >
          {(page) => (
            <ol className="materials-lineage">
              {page.items.map((edge, index) => (
                <li
                  key={`${edge.kind}-${edge.createdAt}-${edge.source.revision}-${index}`}
                >
                  <strong>{edge.kind.replaceAll("_", " ")}</strong>
                  <span className="materials-lineage-path">
                    <ExactRef scope={edge.sourceScope} value={edge.source} />
                    <span aria-hidden="true">→</span>
                    <span className="ui-visually-hidden">to</span>
                    <ExactRef scope={edge.targetScope} value={edge.target} />
                  </span>
                  <span className="materials-lineage-meta">
                    {edge.runId === undefined ? null : (
                      <Link to={`/runs/${encodeURIComponent(edge.runId)}`}>
                        Run {edge.runId}
                      </Link>
                    )}
                    {edge.stageExecutionId === undefined ? null : (
                      <small>Stage execution {edge.stageExecutionId}</small>
                    )}
                    <RecordedTime value={edge.createdAt} />
                  </span>
                </li>
              ))}
            </ol>
          )}
        </QueryView>
        <CursorControls
          label="Lineage pages"
          {...lineagePages.controls(lineage.data?.page)}
        />
      </section>
    </div>
  );
}
