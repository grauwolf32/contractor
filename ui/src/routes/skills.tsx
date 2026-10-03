import { ContextLink } from "../app/context-navigation";
import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import { listArtifacts, type ArtifactWriteResponse } from "../api/artifacts";
import { usePublicAPI } from "../api/context";
import { queryKeys } from "../api/query-keys";
import { CursorControls } from "../app/cursor-controls";
import { useCursorStack } from "../app/pagination";
import { formatBytes } from "../app/format";
import { SkillUploadDialog } from "./skill-upload-dialog";
import { SkillDescription } from "./skill-description";

import "./skills.css";
import { RefreshButton } from "../app/refresh-button";
import { RecordedTime } from "../app/recorded-time";
import { QueryView } from "../app/query-view";
import { ArtifactStoredNotice } from "./artifacts/bindings";
import { artifactDetailPath } from "./artifacts/paths";

const SKILL_NAMESPACE = "skills";

export function SkillsRoute() {
  const api = usePublicAPI();
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
          <RefreshButton
            isFetching={query.isFetching}
            onRefresh={() => void query.refetch()}
            label="Refresh"
          />
        </div>
      </header>

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

      <div className="panel skill-library">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Skills</p>
            <h3>Current packages</h3>
          </div>
          <span className="skill-scope-badge">Global</span>
        </div>
        <QueryView
          query={query}
          loading={
            <p className="loading-copy" aria-live="polite">
              Loading Skills…
            </p>
          }
          errorContext="Could not load Skills"
          onRetry={() => void query.refetch()}
          isEmpty={(queryData) => queryData.items.length === 0}
          empty={
            <div className="compact-empty">
              <strong>No global Skill packages found.</strong>
              <p>
                Bundled initialization or an ordinary User Artifact upload can
                create the first package.
              </p>
            </div>
          }
        >
          {(queryData) => (
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
                  {queryData.items.map((item) => (
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
                          <small>
                            <RecordedTime value={item.createdAt} />
                          </small>
                        </details>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </QueryView>
        <CursorControls
          label="Skill package pages"
          {...pages.controls(query.data?.page)}
        />
      </div>
    </section>
  );
}
