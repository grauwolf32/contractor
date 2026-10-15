import { useQuery } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { useLocation, useSearchParams } from "react-router";

import type { ArtifactWriteResponse } from "../../api/artifacts";
import { usePublicAPI } from "../../api/context";
import type { GitImportResult } from "../../api/git-artifacts";
import { listProjectArtifacts } from "../../api/project-artifacts";
import { queryKeys } from "../../api/query-keys";
import { usePageLocationNow } from "../../app/page-location";
import { useURLCursorStack } from "../../app/pagination";
import { RefreshButton } from "../../app/refresh-button";
import { Button, ListSection } from "../../ui";
import { AddMaterialSheet } from "../artifacts/add-material";
import { MaterialIcon } from "../artifacts/icons";
import { groupMaterialsByKind } from "../artifacts/kinds";
import { MaterialList } from "../artifacts/material-list";
import { MaterialRow } from "../artifacts/material-row";
import { MaterialAddedNotice } from "../artifacts/notices";
import { artifactDetailPath } from "../artifacts/paths";
import "../artifacts/materials.css";

/** `?add=artifact` opens the "Add material" sheet (a deep link from Overview). */
const ADD_PARAM = "add";
const ADD_VALUE = "artifact";
const NAMESPACE_PARAM = "artifactsNamespace";
const CURSOR_PARAM = "artifactsCursor";

/**
 * Project → Materials: the project's materials grouped by kind, with the
 * namespace filter and page cursor in the URL, and the "Add material" sheet
 * (upload or Git import).
 */
export function ProjectArtifactRegion({ projectId }: { projectId: string }) {
  const api = usePublicAPI();
  const [filters, setFilters] = useSearchParams();
  const location = useLocation();
  const pageNow = usePageLocationNow();
  const region = useRef<HTMLElement>(null);
  const addButton = useRef<HTMLButtonElement>(null);
  const notice = useRef<HTMLDivElement>(null);
  const addOpen = filters.get(ADD_PARAM) === ADD_VALUE;
  const namespace = filters.get(NAMESPACE_PARAM) || undefined;
  const pages = useURLCursorStack({
    param: CURSOR_PARAM,
    navigateOptions: { preventScrollReset: true, state: location.state },
  });
  const [written, setWritten] = useState<
    ArtifactWriteResponse | GitImportResult | null
  >(null);
  const cursor = pages.cursor;
  const query = useQuery({
    queryKey: queryKeys.projects.artifacts.list(projectId, namespace, cursor),
    queryFn: () =>
      listProjectArtifacts(api, {
        projectId,
        ...(namespace === undefined ? {} : { namespace }),
        ...(cursor === undefined ? {} : { cursor }),
      }),
  });

  // Move focus to the outcome of a write once the dialog that made it has
  // closed. The outcome (state) and the closed sheet (the URL) commit
  // separately; while the sheet is open the rest of the page is inert, and on
  // closing it restores focus itself, so focusing earlier would be lost.
  const focusOutcome = useRef(false);
  useEffect(() => {
    if (written === null || addOpen || !focusOutcome.current) return undefined;
    const timer = window.setTimeout(() => {
      focusOutcome.current = false;
      notice.current?.focus();
    }, 0);
    return () => window.clearTimeout(timer);
  }, [written, addOpen]);

  function changeFilters(
    change: (next: URLSearchParams) => void,
    replace = false,
  ): void {
    const next = new URLSearchParams(filters);
    change(next);
    setFilters(next, {
      preventScrollReset: true,
      state: location.state,
      ...(replace ? { replace: true } : {}),
    });
  }

  function setNamespaceFilter(value: string | undefined): void {
    changeFilters((next) => {
      next.delete(CURSOR_PARAM);
      if (value === undefined) next.delete(NAMESPACE_PARAM);
      else next.set(NAMESPACE_PARAM, value);
    });
  }

  function openAdd(): void {
    changeFilters((next) => next.set(ADD_PARAM, ADD_VALUE), true);
  }

  function closeAdd(): void {
    changeFilters((next) => next.delete(ADD_PARAM), true);
    // A sheet opened by a link has no trigger to return focus to.
    window.setTimeout(() => {
      const button = addButton.current;
      const active = document.activeElement;
      if (
        button !== null &&
        button.isConnected &&
        (active === null || !(region.current?.contains(active) ?? false))
      )
        button.focus();
    }, 0);
  }

  function finishAdd(result: ArtifactWriteResponse | GitImportResult): void {
    focusOutcome.current = true;
    setWritten(result);
    // The write answered after an await: change the URL only while the
    // user is still on this page.
    if (pageNow() === undefined) return;
    // Show the new material: back to the first page of every namespace.
    changeFilters((next) => {
      next.delete(ADD_PARAM);
      next.delete(NAMESPACE_PARAM);
      next.delete(CURSOR_PARAM);
    }, true);
  }

  return (
    <section
      ref={region}
      className="materials-region"
      id="project-artifacts"
      aria-label="Materials"
    >
      <div className="materials-region-head">
        <p className="materials-region-intro">
          Source code, API specs, architecture models and docs for checks and
          Runs. Each check and Run reads one exact version, so a new version
          never changes earlier results.
        </p>
        <div className="ui-toolbar">
          <Button
            ref={addButton}
            variant="primary"
            icon={<MaterialIcon name="plus" />}
            onClick={openAdd}
          >
            Add material
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
        <MaterialAddedNotice
          ref={notice}
          projectId={projectId}
          result={written}
          onDismiss={() => {
            setWritten(null);
            addButton.current?.focus();
          }}
        />
      )}

      <div className="materials-panel">
        <MaterialList
          query={query}
          namespace={namespace}
          onNamespace={setNamespaceFilter}
          noun="materials"
          empty={{
            title: "No materials yet",
            text: "Add source code, an API spec, an architecture model or docs so checks have something to read.",
          }}
          namespaceEmptyText="Show all namespaces to see every material of this project."
          pagesLabel="Material pages"
          pages={pages}
        >
          {(data) => {
            // Groups come from one page: their counts are totals only when
            // every material fits on it.
            const onePage = cursor === undefined && !data.page.hasMore;
            return (
              <>
                {onePage ? null : (
                  <p className="ui-caption materials-page-note">
                    Grouped by kind within this page. Other pages can hold more
                    materials of each kind.
                  </p>
                )}
                {groupMaterialsByKind(data.items).map((group) => (
                  <ListSection
                    key={group.kind}
                    title={group.label}
                    count={group.items.length}
                    aside={onePage ? undefined : "on this page"}
                    titleAs="h3"
                  >
                    {group.items.map((item) => (
                      <MaterialRow
                        key={`${item.artifact.namespace}/${item.artifact.name}`}
                        item={item}
                        returnLabel="Materials"
                        to={artifactDetailPath(
                          { kind: "project", id: projectId },
                          {
                            namespace: item.artifact.namespace,
                            name: item.artifact.name,
                          },
                        )}
                      />
                    ))}
                  </ListSection>
                ))}
              </>
            );
          }}
        </MaterialList>
      </div>

      {addOpen ? (
        <AddMaterialSheet
          projectId={projectId}
          onClose={closeAdd}
          onAdded={finishAdd}
        />
      ) : null}
    </section>
  );
}
