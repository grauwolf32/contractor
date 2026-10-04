import { useDocumentTitle } from "../../app/document-title";
import { useQuery } from "@tanstack/react-query";
import { useId, useState } from "react";
import { Link, useSearchParams } from "react-router";

import { listArtifacts, type ArtifactWriteResponse } from "../../api/artifacts";
import { usePublicAPI } from "../../api/context";
import { queryKeys } from "../../api/query-keys";
import { ArtifactWriteForm } from "./common";
import { CursorControls } from "../../app/cursor-controls";
import { useURLCursorStack } from "../../app/pagination";
import { Dialog, DialogHeader } from "../../app/dialog";
import { RefreshButton } from "../../app/refresh-button";
import { QueryView } from "../../app/query-view";
import { ArtifactBindingsTable, ArtifactStoredNotice } from "./bindings";
import { useNamespaceFilter } from "./namespace-filter";
import { artifactDetailPath } from "./paths";

const EXCLUDED_SKILL_NAMESPACE = "skills";
const USER_SCOPE = { kind: "user" } as const;

export function ArtifactListRoute() {
  useDocumentTitle("Artifacts");
  const api = usePublicAPI();
  const [filters, setFilters] = useSearchParams();
  const namespace = filters.get("namespace") || undefined;
  const pages = useURLCursorStack();
  const [written, setWritten] = useState<ArtifactWriteResponse | null>(null);
  const [uploadOpen, setUploadOpen] = useState(false);
  const uploadHeading = useId();
  const cursor = pages.cursor;
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

  const namespaceFilter = useNamespaceFilter({
    value: namespace,
    onApply: (candidate) => {
      const next = new URLSearchParams(filters);
      next.delete("cursor");
      if (candidate === undefined) next.delete("namespace");
      else next.set("namespace", candidate);
      setFilters(next, { preventScrollReset: true });
    },
  });

  return (
    <section className="route-page artifact-page">
      <header className="route-header-row">
        <div>
          <h2>Artifacts</h2>
          <p className="lede">
            Inputs uploaded here can be updated; each Run pins one revision.
          </p>
        </div>
        <div className="inline-actions">
          <button type="button" onClick={() => setUploadOpen(true)}>
            Upload Artifact
          </button>
          <RefreshButton
            isFetching={query.isFetching}
            onRefresh={() => void query.refetch()}
            label="Refresh"
          />
        </div>
      </header>

      {uploadOpen ? (
        <Dialog
          className="project-dialog panel"
          labelledBy={uploadHeading}
          onRequestClose={() => setUploadOpen(false)}
        >
          <DialogHeader
            id={uploadHeading}
            eyebrow="New Artifact"
            title="Upload Artifact"
            close={{
              label: "Close Upload Artifact form",
              onClose: () => setUploadOpen(false),
            }}
          />
          <ArtifactWriteForm
            excludedNamespace={{
              namespace: EXCLUDED_SKILL_NAMESPACE,
              destination: "/catalog/skills",
              label: "Skills",
            }}
            headingId={uploadHeading}
            onCancel={() => setUploadOpen(false)}
            onWritten={(result) => {
              setWritten(result);
              pages.reset();
              setUploadOpen(false);
            }}
          />
        </Dialog>
      ) : null}
      {written === null ? null : (
        <ArtifactStoredNotice
          title="Artifact revision stored."
          artifact={written.artifact}
          returnLabel="Artifacts"
          to={artifactDetailPath(USER_SCOPE, written.artifact)}
        />
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
          {namespaceFilter.form}
        </div>
        {namespaceFilter.error}
        <QueryView
          query={query}
          loading={
            <p className="loading-copy" role="status">
              Loading Artifact bindings…
            </p>
          }
          errorContext="Could not load Artifact bindings"
          onRetry={() => void query.refetch()}
          isEmpty={(queryData) => queryData.items.length === 0}
          empty={
            <div className="compact-empty">
              <strong>No Artifact bindings found.</strong>
              <p>Upload the first Workflow input above.</p>
            </div>
          }
        >
          {(queryData) => (
            <ArtifactBindingsTable
              items={queryData.items}
              returnLabel="Artifacts"
              detailPath={(item) =>
                artifactDetailPath(USER_SCOPE, {
                  namespace: item.artifact.namespace,
                  name: item.artifact.name,
                })
              }
            />
          )}
        </QueryView>
        <CursorControls
          label="Artifact binding pages"
          {...pages.controls(query.data?.page)}
        />
      </div>
    </section>
  );
}
