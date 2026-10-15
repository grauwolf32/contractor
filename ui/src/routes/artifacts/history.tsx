import { useQuery, type QueryKey } from "@tanstack/react-query";
import { useId, type ReactNode } from "react";
import { Link, useLocation } from "react-router";

import type { ArtifactArchiveScope } from "../../api/artifact-archive";
import type { ArtifactMetadata, ExactArtifactRef } from "../../api/artifacts";
import { usePublicAPI } from "../../api/context";
import { artifactScopeKeys } from "../../api/query-keys";
import { scopedArtifactAPI } from "../../api/scoped-artifacts";
import { CursorControls } from "../../app/cursor-controls";
import { formatBytes } from "../../app/format";
import { useCursorStack, type PageContinuation } from "../../app/pagination";
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
    <span className="ui-mono">
      {scope}: {value.namespace}/{value.name}@{value.revision}
    </span>
  );
}

/**
 * One paged list of the history: a heading, the list in its loading and
 * empty states, and its own page cursor.
 */
function HistorySection<T>({
  title,
  titleAs: Title,
  queryKey,
  load,
  empty,
  pagesLabel,
  children,
}: {
  title: string;
  titleAs: "h2" | "h3";
  queryKey: (cursor: string | undefined) => QueryKey;
  load: (page: {
    cursor?: string;
  }) => Promise<{ items: T[]; page: PageContinuation }>;
  empty: string;
  pagesLabel: string;
  children: (items: T[]) => ReactNode;
}) {
  const heading = useId();
  const pages = useCursorStack();
  const query = useQuery({
    queryKey: queryKey(pages.cursor),
    queryFn: () =>
      load(pages.cursor === undefined ? {} : { cursor: pages.cursor }),
  });
  return (
    <section aria-labelledby={heading}>
      <Title id={heading} className="materials-subtitle">
        {title}
      </Title>
      <QueryView
        query={query}
        loading={
          <p className="materials-loading" role="status">
            {`Loading ${title.toLowerCase()}…`}
          </p>
        }
        isEmpty={(page) => page.items.length === 0}
        empty={<p className="ui-caption">{empty}</p>}
      >
        {(page) => children(page.items)}
      </QueryView>
      <CursorControls
        label={pagesLabel}
        {...pages.controls(query.data?.page)}
      />
    </section>
  );
}

/**
 * Every version of a binding (newest first, paged) and the lineage of the
 * shown revision: where it came from and where Runs took it.
 */
export function ArtifactHistory({
  scope,
  metadata,
  titleAs = "h3",
}: {
  scope: ArtifactArchiveScope;
  metadata: ArtifactMetadata;
  /** Heading level of Versions and Lineage. Default "h3". */
  titleAs?: "h2" | "h3" | undefined;
}) {
  const location = useLocation();
  const api = usePublicAPI();
  const artifacts = scopedArtifactAPI(api, scope);
  const keys = artifactScopeKeys(scope);
  const { namespace, name, revision } = metadata.artifact;
  return (
    <div className="materials-history">
      <HistorySection
        title="Versions"
        titleAs={titleAs}
        queryKey={(cursor) => keys.versions(namespace, name, cursor)}
        load={(page) => artifacts.versions({ namespace, name, ...page })}
        empty="No versions found."
        pagesLabel="Version pages"
      >
        {(items) => (
          <ol className="materials-versions">
            {items.map((item) => (
              <li key={item.artifact.revision}>
                <Link
                  to={`?revision=${encodeURIComponent(item.artifact.revision)}`}
                  state={location.state}
                  aria-current={
                    item.artifact.revision === revision ? "page" : undefined
                  }
                >
                  <code>{item.artifact.revision}</code>
                  <span>{formatBytes(item.size)}</span>
                  <RecordedTime value={item.createdAt} />
                  {item.current ? (
                    <span className="materials-badge">current</span>
                  ) : null}
                </Link>
              </li>
            ))}
          </ol>
        )}
      </HistorySection>
      <HistorySection
        title="Lineage"
        titleAs={titleAs}
        queryKey={(cursor) => keys.lineage(namespace, name, revision, cursor)}
        load={(page) =>
          artifacts.lineage({ namespace, name, revision, ...page })
        }
        empty="No lineage edges reference this revision."
        pagesLabel="Lineage pages"
      >
        {(items) => (
          <ol className="materials-lineage">
            {items.map((edge, index) => (
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
      </HistorySection>
    </div>
  );
}
