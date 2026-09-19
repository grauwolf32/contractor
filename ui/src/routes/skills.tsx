import { ContextLink } from "../app/context-navigation";
import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import { listArtifacts, type ArtifactWriteResponse } from "../api/artifacts";
import { usePublicAPI } from "../api/context";
import { queryKeys } from "../api/query-keys";
import {
  CursorControls,
  ErrorNotice,
  formatBytes,
  formatTimestamp,
} from "./artifacts/common";
import { SkillUploadDialog } from "./skill-upload-dialog";
import { SkillDescription } from "./skill-description";

import "./skills.css";

const SKILL_NAMESPACE = "skills";

export function SkillsRoute() {
  const api = usePublicAPI();
  const [cursors, setCursors] = useState<Array<string | undefined>>([
    undefined,
  ]);
  const [uploadOpen, setUploadOpen] = useState(false);
  const [written, setWritten] = useState<ArtifactWriteResponse | null>(null);
  const cursor = cursors.at(-1);
  const query = useQuery({
    queryKey: queryKeys.artifacts.list(SKILL_NAMESPACE, cursor),
    queryFn: () =>
      listArtifacts(api, {
        namespace: SKILL_NAMESPACE,
        ...(cursor === undefined ? {} : { cursor }),
      }),
  });

  return (
    <section className="route-page skills-page">
      <header className="route-header-row">
        <div>
          <h2>Skills</h2>
          <p className="lede">
            Reusable instructions and files for your agents across Projects.
          </p>
        </div>
        <div className="skills-actions">
          <button type="button" onClick={() => setUploadOpen(true)}>
            Upload Skills
          </button>
          <button
            className="secondary-button"
            type="button"
            disabled={query.isFetching}
            onClick={() => void query.refetch()}
          >
            {query.isFetching ? "Refreshing…" : "Refresh"}
          </button>
        </div>
      </header>

      {uploadOpen ? (
        <SkillUploadDialog
          onClose={() => setUploadOpen(false)}
          onWritten={(result) => {
            setWritten(result);
            setCursors([undefined]);
            setUploadOpen(false);
          }}
        />
      ) : null}

      {written === null ? null : (
        <div className="notice notice-success" role="status">
          <strong>Global Skill revision stored.</strong>
          <ContextLink
            returnLabel="Skills"
            to={`/artifacts/skills/${encodeURIComponent(written.artifact.name)}?revision=${encodeURIComponent(written.artifact.revision)}`}
          >
            Open skills/{written.artifact.name}@{written.artifact.revision}
          </ContextLink>
        </div>
      )}

      <div className="panel skill-library">
        <div className="section-heading">
          <div>
            <p className="eyebrow">UserScope / skills</p>
            <h3>Current packages</h3>
          </div>
          <span className="skill-scope-badge">Global</span>
        </div>
        {query.isPending ? (
          <p className="loading-copy" aria-live="polite">
            Loading Skills…
          </p>
        ) : query.error !== null ? (
          <ErrorNotice error={query.error} />
        ) : query.data.items.length === 0 ? (
          <div className="compact-empty">
            <strong>No global Skill packages found.</strong>
            <p>
              Bundled initialization or an ordinary User Artifact upload can
              create the first package.
            </p>
          </div>
        ) : (
          <div className="table-scroll">
            <table className="responsive-table skill-table">
              <thead>
                <tr>
                  <th>Skill</th>
                  <th>Purpose</th>
                  <th>Version</th>
                </tr>
              </thead>
              <tbody>
                {query.data.items.map((item) => (
                  <tr key={item.artifact.name}>
                    <td data-label="Skill">
                      <ContextLink
                        returnLabel="Skills"
                        to={`/artifacts/skills/${encodeURIComponent(item.artifact.name)}`}
                      >
                        {item.artifact.name}
                      </ContextLink>
                      <small className="skill-package-size">
                        {formatBytes(item.size)} · Skill package
                      </small>
                    </td>
                    <td data-label="Purpose">
                      <SkillDescription metadata={item} />
                    </td>
                    <td data-label="Version">
                      <details>
                        <summary>Current revision</summary>
                        <code>{item.artifact.revision}</code>
                        <small>{formatTimestamp(item.createdAt)}</small>
                      </details>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <CursorControls
          label="Skill package pages"
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
