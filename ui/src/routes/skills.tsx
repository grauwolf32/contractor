import { useQuery } from "@tanstack/react-query";
import { useId, useState } from "react";

import { listArtifacts, type ArtifactWriteResponse } from "../api/artifacts";
import { usePublicAPI } from "../api/context";
import { queryKeys } from "../api/query-keys";
import { ContextLink } from "../app/context-navigation";
import { CursorControls } from "../app/cursor-controls";
import { formatBytes } from "../app/format";
import { useCursorStack } from "../app/pagination";
import { QueryView } from "../app/query-view";
import { RecordedTime } from "../app/recorded-time";
import { RefreshButton } from "../app/refresh-button";
import { EmptyState } from "../ui";
import { ArtifactStoredNotice } from "./artifacts/bindings";
import { artifactDetailPath } from "./artifacts/paths";
import { LibrarySectionHeader } from "./catalog/library-parts";
import { SkillDescription } from "./skill-description";
import { SkillUploadDialog } from "./skill-upload-dialog";

import "./skills.css";

const SKILL_NAMESPACE = "skills";

/** Library → Skills: global Skill packages and their upload. */
export function SkillsRoute() {
  const api = usePublicAPI();
  const heading = useId();
  const listHeading = useId();
  const pages = useCursorStack();
  const [uploadOpen, setUploadOpen] = useState(false);
  const [written, setWritten] = useState<ArtifactWriteResponse | null>(null);
  const cursor = pages.cursor;
  const query = useQuery({
    queryKey: queryKeys.artifacts.list(SKILL_NAMESPACE, cursor),
    queryFn: () =>
      listArtifacts(api, {
        namespace: SKILL_NAMESPACE,
        ...(cursor === undefined ? {} : { cursor }),
      }),
  });

  return (
    <section className="library-section skills-page" aria-labelledby={heading}>
      <LibrarySectionHeader
        id={heading}
        title="Skills"
        description="Reusable instructions and files for your agents across Projects."
        actions={
          <>
            <RefreshButton
              className="ui-btn"
              isFetching={query.isFetching}
              onRefresh={() => void query.refetch()}
              label="Refresh"
            />
            <button
              type="button"
              className="ui-btn"
              data-variant="primary"
              onClick={() => setUploadOpen(true)}
            >
              Upload Skills
            </button>
          </>
        }
      />

      {uploadOpen ? (
        <SkillUploadDialog
          onClose={() => setUploadOpen(false)}
          onWritten={(result) => {
            setWritten(result);
            pages.reset();
            setUploadOpen(false);
          }}
        />
      ) : null}

      {written === null ? null : (
        <ArtifactStoredNotice
          title="Global Skill revision stored."
          artifact={written.artifact}
          returnLabel="Skills"
          to={artifactDetailPath({ kind: "user" }, written.artifact)}
        />
      )}

      <section
        className="library-panel skill-library"
        aria-labelledby={listHeading}
      >
        <header className="library-panel-header">
          <div>
            <h3 id={listHeading} className="library-block-title">
              Current packages
            </h3>
            <p className="library-muted">
              Each package holds a SKILL.md and its files. Open one to read it
              or to see earlier revisions.
            </p>
          </div>
          <span className="library-tag">Global</span>
        </header>
        <QueryView
          query={query}
          loading={
            <p className="library-muted" aria-live="polite">
              Loading Skills…
            </p>
          }
          errorContext="Could not load Skills"
          onRetry={() => void query.refetch()}
          isEmpty={(queryData) => queryData.items.length === 0}
          empty={
            <EmptyState title="No global Skill packages found.">
              Bundled initialization or an ordinary User Artifact upload can
              create the first package.
            </EmptyState>
          }
        >
          {(queryData) => (
            <ul className="skill-list">
              {queryData.items.map((item) => (
                <li className="skill-row" key={item.artifact.name}>
                  <div className="skill-row-main">
                    <ContextLink
                      className="skill-row-name"
                      returnLabel="Skills"
                      to={`/artifacts/skills/${encodeURIComponent(item.artifact.name)}`}
                    >
                      {item.artifact.name}
                    </ContextLink>
                    <SkillDescription metadata={item} />
                  </div>
                  <div className="skill-row-side">
                    <span className="skill-package-size">
                      {formatBytes(item.size)} · Skill package
                    </span>
                    <details className="skill-revision">
                      <summary>Current revision</summary>
                      <code>{item.artifact.revision}</code>
                      <small>
                        <RecordedTime value={item.createdAt} />
                      </small>
                    </details>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </QueryView>
        <CursorControls
          label="Skill package pages"
          {...pages.controls(query.data?.page)}
        />
      </section>
    </section>
  );
}
