import { useQuery } from "@tanstack/react-query";
import { useEffect, useId, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router";

import { listArtifacts, type ArtifactWriteResponse } from "../../api/artifacts";
import { usePublicAPI } from "../../api/context";
import { queryKeys } from "../../api/query-keys";
import { useDocumentTitle } from "../../app/document-title";
import { useURLCursorStack } from "../../app/pagination";
import { RefreshButton } from "../../app/refresh-button";
import { Button, ListSection } from "../../ui";
import { LibraryTabs } from "../catalog/library-tabs";
import { ArtifactWriteForm } from "./common";
import { MaterialIcon } from "./icons";
import { MaterialList } from "./material-list";
import { MaterialRow } from "./material-row";
import { FileStoredNotice } from "./notices";
import { artifactDetailPath } from "./paths";
import { UploadDialog } from "./upload-dialog";
import { uploadCancelNote } from "./upload-operation";
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

  return (
    <div className="route-page materials-page">
      <header className="materials-library-head">
        <h1 className="materials-library-title">Library</h1>
        <LibraryTabs />
      </header>

      <section className="materials-panel" aria-labelledby={titleId}>
        <div className="materials-panel-title">
          <div>
            <h2 id={titleId}>Files</h2>
            <p className="ui-caption">
              Your own inputs for Workflow Runs. A Run reads the exact version
              you select. Skill packages live in{" "}
              <Link to={SKILLS_PATH}>Skills</Link>.
            </p>
          </div>
          <div className="ui-toolbar">
            <Button
              ref={uploadButton}
              variant="primary"
              icon={<MaterialIcon name="upload" />}
              onClick={() => setUploadOpen(true)}
            >
              Upload file
            </Button>
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

        <MaterialList
          query={query}
          namespace={namespace}
          onNamespace={applyNamespace}
          noun="files"
          empty={{
            title: "No files yet",
            text: "Upload a file to use it as a Workflow input.",
          }}
          namespaceEmptyText="Show all namespaces to see every file."
          pagesLabel="File pages"
          pages={pages}
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
        </MaterialList>
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
 * material (UploadDialog): closing the dialog aborts a running upload.
 */
function FileUploadDialog({
  onClose,
  onWritten,
}: {
  onClose: () => void;
  onWritten: (result: ArtifactWriteResponse) => void;
}) {
  return (
    <UploadDialog
      eyebrow="Library"
      title="Upload file"
      closeLabel="Close file upload"
      note={uploadCancelNote("file")}
      onClose={onClose}
    >
      {(upload) => (
        <ArtifactWriteForm
          excludedNamespace={{
            namespace: EXCLUDED_SKILL_NAMESPACE,
            destination: SKILLS_PATH,
            label: "Skills",
          }}
          headingId={upload.headingId}
          startOperation={upload.start}
          onCancel={upload.close}
          onWritten={onWritten}
        />
      )}
    </UploadDialog>
  );
}
