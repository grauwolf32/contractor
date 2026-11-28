import { useId, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router";
import { useDocumentTitle } from "../../../app/document-title";
import { ErrorNotice } from "../../../app/error-notice";
import { Icon } from "../../../app/icon";
import { useListReadingPosition } from "../../../app/list-reading-position";
import { RecordedTime } from "../../../app/recorded-time";
import { findingStateLabel } from "../../../app/vocabulary";
import {
  Button,
  EmptyState,
  ListRow,
  ListSection,
  PaneLayout,
  useListNavigation,
} from "../../../ui";
import { IssueDetail } from "../../issues/issue-detail";
import { issuePath } from "../../issues/links";
import { ProjectSectionActions } from "../navigation";
import { DECISION_OPTIONS } from "./finding-options";
import { useProjectFindingView } from "./project-finding-view";
import "./project-facts.css";

export function ProjectFactsRoute() {
  const { projectId = "", findingId } = useParams();
  const navigate = useNavigate();
  const {
    params,
    findings,
    filters,
    verdict,
    visible,
    chips,
    change,
    refresh,
    total,
  } = useProjectFindingView(projectId, true);
  const criteria = new URLSearchParams(params);
  criteria.delete("from");
  criteria.delete("tab");
  const listPath = `/projects/${encodeURIComponent(projectId)}/facts${criteria.size ? "?" + criteria : ""}`;
  const detailParams = new URLSearchParams(criteria);
  detailParams.set("from", "facts");
  const toFact = (id: string) => `${issuePath(projectId, id)}?${detailParams}`;
  const position = visible.findIndex((row) => row.findingId === findingId);
  const openAt = (index: number) => {
    const row = visible[index];
    if (row)
      void navigate(toFact(row.findingId), {
        replace: findingId !== undefined,
      });
  };
  const nav = useListNavigation({
    count: visible.length,
    index: position,
    onMove: openAt,
    onOpen: openAt,
    enabled: findingId !== undefined,
  });
  const ref = useRef<HTMLElement>(null);
  const remember = useListReadingPosition(
    ref,
    findings.data !== undefined && findingId === undefined,
    ".fact-rows a",
  );
  const [filtersOpen, setFiltersOpen] = useState(false);
  const active =
    Number(filters.state !== "all") + Number(verdict !== undefined);
  const optionsId = useId();
  useDocumentTitle(visible[position]?.title ?? "Project facts");
  const list = (
    <section
      className="route-page project-facts"
      ref={ref}
      onClickCapture={findingId ? undefined : remember}
    >
      {!findingId && (
        <ProjectSectionActions>
          <Button size="sm" pending={findings.isFetching} onClick={refresh}>
            Refresh
          </Button>
        </ProjectSectionActions>
      )}
      <header className="facts-heading">
        <h2>Facts</h2>
      </header>
      <div className="facts-toolbar">
        <label className="facts-search">
          <Icon name="search" size={16} />
          <input
            type="search"
            aria-label="Search facts"
            placeholder="Search facts"
            value={params.get("q") ?? ""}
            onChange={(e) => change("q", e.target.value)}
          />
        </label>
        <Button
          size="sm"
          aria-expanded={filtersOpen}
          aria-controls={optionsId}
          onClick={() => setFiltersOpen(!filtersOpen)}
        >
          Filters{active > 0 ? ` · ${active}` : ""}
        </Button>
        <div className="facts-count">
          {findings.data && (
            <span role="status">
              {visible.length} {visible.length === 1 ? "fact" : "facts"} shown ·{" "}
              {total} records loaded
            </span>
          )}
          {criteria.size > 0 && (
            <Button
              size="xs"
              variant="ghost"
              onClick={() =>
                void navigate(
                  findingId
                    ? issuePath(projectId, findingId) + "?from=facts"
                    : `/projects/${encodeURIComponent(projectId)}/facts`,
                  { replace: true },
                )
              }
            >
              Clear filters
            </Button>
          )}
        </div>
      </div>
      <div className="facts-options" id={optionsId} hidden={!filtersOpen}>
        <label>
          State
          <select
            value={filters.state}
            onChange={(e) => change("state", e.target.value)}
          >
            {chips.map((c) => (
              <option key={c.value} value={c.value}>
                {c.label} ({c.count})
              </option>
            ))}
          </select>
        </label>
        <label>
          Decision
          <select
            value={verdict ?? ""}
            onChange={(e) => change("verdict", e.target.value)}
          >
            {DECISION_OPTIONS.map((c) => (
              <option key={c.value} value={c.value}>
                {c.label}
              </option>
            ))}
          </select>
        </label>
      </div>
      {findings.isPending && <p role="status">Loading facts…</p>}
      {findings.error && (
        <ErrorNotice
          error={findings.error}
          onRetry={
            findings.isFetchNextPageError
              ? () => void findings.fetchNextPage()
              : refresh
          }
        />
      )}
      {findings.isSuccess && !visible.length && (
        <EmptyState title="No facts in the loaded records">
          {findings.hasNextPage
            ? "Load more records to extend this view."
            : criteria.size
              ? "Try another search or clear the filters."
              : "Informational observations from this project appear here."}
        </EmptyState>
      )}
      <div className="fact-rows" {...nav.containerProps}>
        {visible.length > 0 && (
          <ListSection aria-label="Facts">
            {visible.map((row) => (
              <ListRow
                key={row.findingId}
                id={`fact-${row.findingId}`}
                to={toFact(row.findingId)}
                selected={findingId === row.findingId}
                glyph={<Icon name="docs" size={18} />}
                title={row.title}
                meta={[
                  row.subjectKey || null,
                  row.analystVerdict
                    ? findingStateLabel(row.state).label
                    : null,
                ]}
                trailing={
                  <span className="fact-date">
                    <RecordedTime value={row.createdAt} />
                    <Icon name="chevron-right" size={15} />
                  </span>
                }
              />
            ))}
          </ListSection>
        )}
      </div>
      {findings.hasNextPage && !findings.isFetchNextPageError && (
        <Button
          size="sm"
          pending={findings.isFetchingNextPage}
          onClick={() => void findings.fetchNextPage()}
        >
          Load more facts
        </Button>
      )}
    </section>
  );
  if (!findingId) return list;
  return (
    <PaneLayout
      className="issues-workspace project-fact-panes"
      showDetail
      listLabel="Project facts"
      detailLabel="Selected observation"
      list={list}
      backLink={{ label: "Back to facts", to: listPath }}
      detail={
        <IssueDetail
          key={findingId}
          projectId={projectId}
          findingId={findingId}
          projects={[]}
          factsReturn={listPath}
          position={
            position < 0
              ? undefined
              : { index: position, count: visible.length }
          }
          listPending={findings.isPending}
          onPrevious={position > 0 ? () => openAt(position - 1) : undefined}
          onNext={
            position >= 0 && position + 1 < visible.length
              ? () => openAt(position + 1)
              : undefined
          }
        />
      }
    />
  );
}
