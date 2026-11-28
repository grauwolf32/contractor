import { useQuery } from "@tanstack/react-query";
import { useContext, useEffect, useRef, useState } from "react";
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
import { useMaterialListScroll } from "../artifacts/material-list-scroll";
import {
  matchingMaterials,
  materialCategory,
} from "../artifacts/material-library-filter";
import {
  MaterialLibraryFilters,
  NoMatchingMaterials,
} from "../artifacts/material-library";
import { MaterialAddedNotice } from "../artifacts/notices";
import { artifactDetailPath } from "../../app/artifact-path";
import { ProjectSectionActionsContext } from "./section-actions-context";
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
  const workspaceActions = useContext(ProjectSectionActionsContext);
  const [filters, setFilters] = useSearchParams();
  const location = useLocation();
  const pageNow = usePageLocationNow();
  const region = useRef<HTMLElement>(null);
  const addButton = useRef<HTMLButtonElement>(null);
  const notice = useRef<HTMLDivElement>(null);
  const addOpen = filters.get(ADD_PARAM) === ADD_VALUE;
  const namespace = filters.get(NAMESPACE_PARAM) || undefined;
  const category = materialCategory(filters.get("materialsKind"));
  const search = filters.get("materialsSearch") ?? "";
  const pages = useURLCursorStack({
    param: CURSOR_PARAM,
    navigateOptions: { preventScrollReset: true, state: location.state },
  });
  const [written, setWritten] = useState<
    ArtifactWriteResponse | GitImportResult | null
  >(null);
  const cursor = pages.cursor;
  const query = useQuery({
    queryKey: queryKeys.projects.artifacts.list(
      projectId,
      namespace,
      cursor,
      "material",
    ),
    queryFn: () =>
      listProjectArtifacts(api, {
        projectId,
        order: "material",
        ...(namespace === undefined ? {} : { namespace }),
        ...(cursor === undefined ? {} : { cursor }),
      }),
  });
  const rememberReadingPosition = useMaterialListScroll(
    region,
    query.data !== undefined,
  );

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
      next.delete("materialsKind");
      next.delete("materialsSearch");
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
      next.delete("materialsKind");
      next.delete("materialsSearch");
    }, true);
  }

  return (
    <section
      ref={region}
      className="materials-region"
      id="project-artifacts"
      aria-label="Materials"
      onClickCapture={rememberReadingPosition}
    >
      <div className="materials-region-head">
        <div className="materials-library-heading">
          <h2>Materials</h2>
          <p className="ui-caption">
            Source code, API specs and supporting files for your checks.
          </p>
        </div>
        <div className="ui-toolbar">
          <Button
            ref={addButton}
            variant="primary"
            icon={<MaterialIcon name="plus" />}
            onClick={openAdd}
          >
            Add material
          </Button>
          {workspaceActions !== undefined ? null : (
            <RefreshButton
              isFetching={query.isFetching}
              onRefresh={() => void query.refetch()}
              label="Refresh"
            />
          )}
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

      <div className="materials-panel materials-library">
        <MaterialList
          query={query}
          namespace={namespace}
          onNamespace={setNamespaceFilter}
          noun="materials"
          advancedFilter
          controls={
            <MaterialLibraryFilters
              items={query.data?.items ?? []}
              category={category}
              search={search}
              partial={
                cursor !== undefined || query.data?.page.hasMore === true
              }
              onCategory={(value) =>
                changeFilters((next) => {
                  if (value === "all") next.delete("materialsKind");
                  else next.set("materialsKind", value);
                }, true)
              }
              onSearch={(value) =>
                changeFilters((next) => {
                  if (value === "") next.delete("materialsSearch");
                  else next.set("materialsSearch", value);
                }, true)
              }
            />
          }
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
            const matches = matchingMaterials(data.items, category, search);
            return (
              <>
                {onePage ? null : (
                  <p className="ui-caption materials-page-note">
                    Search, categories and counts apply to this page. Use Next
                    to browse more materials, ordered by kind.
                  </p>
                )}
                {matches.length === 0 ? (
                  <NoMatchingMaterials
                    onClear={() =>
                      changeFilters((next) => {
                        next.delete("materialsKind");
                        next.delete("materialsSearch");
                      }, true)
                    }
                  />
                ) : null}
                {groupMaterialsByKind(matches).map((group) => (
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
                        previewOnOpen
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
