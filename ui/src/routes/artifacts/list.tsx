import { useQuery } from "@tanstack/react-query";
import { useCallback, useEffect, useId, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router";

import { listArtifacts, type ArtifactWriteResponse } from "../../api/artifacts";
import { usePublicAPI } from "../../api/context";
import { queryKeys } from "../../api/query-keys";
import { CursorControls } from "../../app/cursor-controls";
import { Dialog, DialogHeader } from "../../app/dialog";
import { useDocumentTitle } from "../../app/document-title";
import { useURLCursorStack } from "../../app/pagination";
import { QueryView } from "../../app/query-view";
import { RefreshButton } from "../../app/refresh-button";
import { EmptyState, ListSection } from "../../ui";
import { LibraryTabs } from "../catalog/library-tabs";
import { ArtifactWriteForm } from "./common";
import { MaterialIcon } from "./icons";
import { MaterialRow } from "./material-row";
import { useNamespaceFilter } from "./namespace-filter";
import { FileStoredNotice } from "./notices";
import { artifactDetailPath } from "./paths";
import "./materials.css";

/** Skill packages have their own Library tab (S06:383-397). */
const EXCLUDED_SKILL_NAMESPACE = "skills";
const SKILLS_PATH = "/catalog/skills";
const USER_SCOPE = { kind: "user" } as const;

/**
 * Library → Files (`/artifacts`): the personal files Runs take as inputs,
 * without Skill packages, with the namespace filter and page cursor in the
 * URL (`?namespace=&cursor=`) and an upload dialog.
 */
export function ArtifactListRoute() {
  useDocumentTitle("Files · Library");
  const api = usePublicAPI();
  const [filters, setFilters] = useSearchParams();
  const namespace = filters.get("namespace") || undefined;
  const pages = useURLCursorStack();
  const [written, setWritten] = useState<ArtifactWriteResponse | null>(null);
  const [uploadOpen, setUploadOpen] = useState(false);
  const titleId = useId();
  const notice = useRef<HTMLDivElement>(null);
  const uploadButton = useRef<HTMLButtonElement>(null);
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

  // The upload dialog has closed by now: move focus to the outcome.
  useEffect(() => {
    if (written === null) return undefined;
    const timer = window.setTimeout(() => notice.current?.focus(), 0);
    return () => window.clearTimeout(timer);
  }, [written]);

  function applyNamespace(candidate: string | undefined): void {
    const next = new URLSearchParams(filters);
    next.delete("cursor");
    if (candidate === undefined) next.delete("namespace");
    else next.set("namespace", candidate);
    setFilters(next, { preventScrollReset: true });
  }

  const namespaceFilter = useNamespaceFilter({
    value: namespace,
    onApply: applyNamespace,
  });

  return (
    <div className="route-page materials-page materials-files">
      <header className="materials-library-head">
        <h1 className="materials-library-title">Library</h1>
        <LibraryTabs />
      </header>

      <section className="materials-panel" aria-labelledby={titleId}>
        <div className="materials-panel-title">
          <div>
            <h2 id={titleId}>Files</h2>
            <p className="materials-quiet">
              Your own inputs for Workflow Runs. A Run reads the exact version
              you select. Skill packages live in{" "}
              <Link to={SKILLS_PATH}>Skills</Link>.
            </p>
          </div>
          <div className="materials-actions">
            <button
              ref={uploadButton}
              type="button"
              className="ui-btn"
              data-variant="primary"
              onClick={() => setUploadOpen(true)}
            >
              <MaterialIcon name="upload" />
              Upload file
            </button>
            <RefreshButton
              className="ui-btn"
              isFetching={query.isFetching}
              onRefresh={() => void query.refetch()}
              label="Refresh"
            />
          </div>
        </div>

        {written === null ? null : (
          <div className="materials-panel-notice">
            <FileStoredNotice
              ref={notice}
              result={written}
              onDismiss={() => {
                setWritten(null);
                uploadButton.current?.focus();
              }}
            />
          </div>
        )}

        <div className="materials-panel-head">
          {namespaceFilter.form}
          {namespace === undefined ? null : (
            <button
              type="button"
              className="ui-btn"
              data-variant="ghost"
              data-size="sm"
              onClick={() => applyNamespace(undefined)}
            >
              Show all namespaces
            </button>
          )}
        </div>
        {namespaceFilter.error}
        <QueryView
          query={query}
          loading={
            <p className="materials-loading" role="status">
              Loading files…
            </p>
          }
          errorContext="Could not load files"
          onRetry={() => void query.refetch()}
          isEmpty={(page) => page.items.length === 0}
          empty={
            namespace === undefined ? (
              <EmptyState title="No files yet">
                Upload a file to use it as a Workflow input.
              </EmptyState>
            ) : (
              <EmptyState title={`Nothing in the ${namespace} namespace`}>
                Show all namespaces to see every file.
              </EmptyState>
            )
          }
        >
          {(page) => (
            <ListSection>
              {page.items.map((item) => (
                <MaterialRow
                  key={`${item.artifact.namespace}/${item.artifact.name}`}
                  item={item}
                  returnLabel="Files"
                  // Rows open the current binding, not the listed revision.
                  to={artifactDetailPath(USER_SCOPE, {
                    namespace: item.artifact.namespace,
                    name: item.artifact.name,
                  })}
                />
              ))}
            </ListSection>
          )}
        </QueryView>
        <div className="materials-pager">
          <CursorControls
            label="File pages"
            {...pages.controls(query.data?.page)}
          />
        </div>
      </section>

      {uploadOpen ? (
        <FileUploadDialog
          onClose={() => setUploadOpen(false)}
          onWritten={(result) => {
            setWritten(result);
            pages.reset();
            setUploadOpen(false);
          }}
        />
      ) : null}
    </div>
  );
}

/**
 * Upload of one file to the personal library, as Materials uploads one
 * material: closing the dialog (or "Cancel upload") aborts a running
 * upload, which the Server may still have stored, so the dialog says to
 * look before retrying; a refused or lost write is explained and never
 * retried.
 */
function FileUploadDialog({
  onClose,
  onWritten,
}: {
  onClose: () => void;
  onWritten: (result: ArtifactWriteResponse) => void;
}) {
  const heading = useId();
  const cancelNote = useId();
  const operation = useRef<AbortController | null>(null);
  useEffect(() => () => operation.current?.abort(), []);
  const startOperation = useCallback(() => {
    const controller = new AbortController();
    operation.current = controller;
    return controller.signal;
  }, []);
  function close() {
    operation.current?.abort();
    onClose();
  }

  return (
    <Dialog
      className="project-dialog panel materials-sheet"
      labelledBy={heading}
      describedBy={cancelNote}
      onRequestClose={close}
    >
      <DialogHeader
        id={heading}
        eyebrow="Library"
        title="Upload file"
        close={{ label: "Close file upload", onClose: close }}
      />
      <p id={cancelNote} className="materials-quiet">
        Closing this dialog cancels a running upload. If you cancel it or lose
        the response, look for the file in the list before trying again: it may
        already be stored.
      </p>
      <ArtifactWriteForm
        excludedNamespace={{
          namespace: EXCLUDED_SKILL_NAMESPACE,
          destination: SKILLS_PATH,
          label: "Skills",
        }}
        headingId={heading}
        startOperation={startOperation}
        onCancel={close}
        onWritten={onWritten}
      />
    </Dialog>
  );
}
