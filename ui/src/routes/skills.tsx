import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Link } from "react-router";

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
          <p className="eyebrow">Global owner guidance</p>
          <h2>Skills</h2>
          <p className="lede">
            One UserScope package and revision history can be selected by
            Workers across any Project. Skills never become Project-owned copies
            or execution permissions.
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
          <Link
            to={`/artifacts/skills/${encodeURIComponent(written.artifact.name)}?revision=${encodeURIComponent(written.artifact.revision)}`}
          >
            Open skills/{written.artifact.name}@{written.artifact.revision}
          </Link>
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
                  <th>Current revision</th>
                  <th>Media type</th>
                  <th>Size</th>
                  <th>Created</th>
                </tr>
              </thead>
              <tbody>
                {query.data.items.map((item) => (
                  <tr key={item.artifact.name}>
                    <td data-label="Skill">
                      <Link
                        to={`/artifacts/skills/${encodeURIComponent(item.artifact.name)}`}
                      >
                        {item.artifact.name}
                      </Link>
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
