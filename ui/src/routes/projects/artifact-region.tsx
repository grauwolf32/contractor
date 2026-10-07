import { useQuery } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { useLocation, useSearchParams } from "react-router";

import type { ArtifactWriteResponse } from "../../api/artifacts";
import { usePublicAPI } from "../../api/context";
import type { GitImportResult } from "../../api/git-artifacts";
import { listProjectArtifacts } from "../../api/project-artifacts";
import { queryKeys } from "../../api/query-keys";
import { CursorControls } from "../../app/cursor-controls";
import { useURLCursorStack } from "../../app/pagination";
import { QueryView } from "../../app/query-view";
import { RefreshButton } from "../../app/refresh-button";
import { EmptyState, ListSection } from "../../ui";
import { AddMaterialSheet } from "../artifacts/add-material";
import { MaterialIcon } from "../artifacts/icons";
import { groupMaterialsByKind } from "../artifacts/kinds";
import { MaterialRow } from "../artifacts/material-row";
import { useNamespaceFilter } from "../artifacts/namespace-filter";
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

  // The dialog that made a write has closed by now: move focus to its outcome.
  useEffect(() => {
    if (written === null) return undefined;
    const timer = window.setTimeout(() => notice.current?.focus(), 0);
    return () => window.clearTimeout(timer);
  }, [written]);

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
    setWritten(result);
    // Show the new material: back to the first page of every namespace.
    changeFilters((next) => {
      next.delete(ADD_PARAM);
      next.delete(NAMESPACE_PARAM);
      next.delete(CURSOR_PARAM);
    }, true);
  }

  const namespaceFilter = useNamespaceFilter({
    value: namespace,
    onApply: setNamespaceFilter,
  });

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
        <div className="materials-actions">
          <button
            ref={addButton}
            type="button"
            className="ui-btn"
            data-variant="primary"
            onClick={openAdd}
          >
            <MaterialIcon name="plus" />
            Add material
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
        <div className="materials-panel-head">
          {namespaceFilter.form}
          {namespace === undefined ? null : (
            <button
              type="button"
              className="ui-btn"
              data-variant="ghost"
              data-size="sm"
              onClick={() => setNamespaceFilter(undefined)}
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
              Loading materials…
            </p>
          }
          errorContext="Could not load materials"
          onRetry={() => void query.refetch()}
          isEmpty={(page) => page.items.length === 0}
          empty={
            namespace === undefined ? (
              <EmptyState title="No materials yet">
                Add source code, an API spec, an architecture model or docs so
                checks have something to read.
              </EmptyState>
            ) : (
              <EmptyState title={`Nothing in the ${namespace} namespace`}>
                Show all namespaces to see every material of this project.
              </EmptyState>
            )
          }
        >
          {(page) =>
            groupMaterialsByKind(page.items).map((group) => (
              <ListSection
                key={group.kind}
                title={group.label}
                count={group.items.length}
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
            ))
          }
        </QueryView>
        <div className="materials-pager">
          <CursorControls
            label="Material pages"
            {...pages.controls(query.data?.page)}
          />
        </div>
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
