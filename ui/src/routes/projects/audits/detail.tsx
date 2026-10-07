import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, type ReactNode } from "react";
import {
  Link,
  Navigate,
  useLocation,
  useNavigate,
  useParams,
  useSearchParams,
  type To,
} from "react-router";

import {
  AUDIT_ID_PATTERN,
  auditPollInterval,
  getAudit,
  type Audit,
} from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { PublicAPIError } from "../../../api/error";
import { getProject, PROJECT_ID_PATTERN } from "../../../api/projects";
import { queryKeys } from "../../../api/query-keys";
import { useDocumentTitle } from "../../../app/document-title";
import { ErrorNotice } from "../../../app/error-notice";
import { catalogReturnState } from "../../../app/navigation";
import {
  checkItemKind,
  checkStateLabel,
  itemNoun,
  type ItemKind,
} from "../../../app/vocabulary";
import {
  DetailPane,
  EmptyState,
  PaneLayout,
  TechnicalDetails,
  useListNavigation,
} from "../../../ui";
import { AuditFindings } from "./audit-findings";
import { itemHeading, parseGroup } from "./assessments";
import { CheckActivity } from "./check-activity";
import {
  useCheckFindings,
  useCheckReport,
  useCheckWorkspace,
  useWaitingDecisions,
} from "./check-data";
import { CheckItemView } from "./check-item";
import {
  CHECK_SECTIONS,
  checkLinks,
  hashedItem,
  parseSection,
  pinsItemList,
  type CheckLinks,
  type CheckSection,
} from "./check-links";
import { CheckListPane } from "./check-list";
import {
  buildEntries,
  commonPathPrefix,
  endpointArea,
  entryName,
  filterEntries,
  groupByKind,
  type CheckEntry,
} from "./check-model";
import { isOpaqueSubject } from "./check-title";
import { useAuditCoverage } from "./coverage-data";
import { AuditRuns } from "./executions";
import { useAuditItems } from "./items-data";
import { auditProfileLabel } from "./labels";
import { CheckProgress } from "./progress";
import { AuditReportView } from "./report";
import { AuditReviews } from "./reviews";

import "./styles.css";
import "./checks.css";

function sectionLabel(section: CheckSection, kind: ItemKind): string {
  switch (section) {
    case "overview":
      return "Overview";
    case "coverage":
      return itemHeading(kind);
    case "findings":
      return "Possible issues";
    case "reviews":
      return "Decisions";
    case "runs":
      return "Runs";
    case "report":
      return "Report";
  }
}

/** "a requirement", "an endpoint", "an item". */
function withArticle(noun: string): string {
  return /^[aeiou]/i.test(noun) ? `an ${noun}` : `a ${noun}`;
}

/** The return link a ContextLink left in the history state, if any. */
function ReturnContext() {
  const { state } = useLocation();
  const target = catalogReturnState(state, { returnTo: "", returnLabel: "" });
  if (target.returnTo === "") return null;
  return (
    <Link
      className="checks-return"
      to={target.returnTo}
      state={target.returnState}
    >
      ← {target.returnLabel}
    </Link>
  );
}

function SectionTabs({
  section,
  kind,
  links,
}: {
  section: CheckSection;
  kind: ItemKind;
  links: CheckLinks;
}) {
  const location = useLocation();
  return (
    <nav className="checks-tabs" aria-label="Check sections">
      <ul role="list">
        {CHECK_SECTIONS.map((candidate) => (
          <li key={candidate}>
            <Link
              to={links.section(candidate)}
              state={location.state}
              aria-current={candidate === section ? "page" : undefined}
            >
              {sectionLabel(candidate, kind)}
            </Link>
          </li>
        ))}
      </ul>
    </nav>
  );
}

/** Names of the check's items for decisions: listed rows, then any item. */
function useSubjects(
  entries: readonly CheckEntry[],
  items: ReturnType<typeof useAuditItems>["items"],
): ReadonlyMap<string, string> {
  return useMemo(() => {
    const names = new Map<string, string>();
    for (const item of items)
      if (!isOpaqueSubject(item.subjectKey))
        names.set(item.itemId, item.subjectKey);
    for (const entry of entries) names.set(entry.row.itemId, entryName(entry));
    return names;
  }, [entries, items]);
}

function CheckPage({
  audit,
  projectName,
  projectError,
  section,
}: {
  audit: Audit;
  projectName: string | undefined;
  projectError: Error | null;
  section: CheckSection;
}) {
  const api = usePublicAPI();
  const location = useLocation();
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const pin = pinsItemList(section) ? params.get("auditRevision") : null;
  const coverage = useAuditCoverage(audit, pin);
  const items = useAuditItems(audit, api, audit.state !== "draft");
  const findings = useCheckFindings(audit);
  const workspace = useCheckWorkspace(audit);
  const overview = section === "overview";
  const waiting = useWaitingDecisions(audit, overview);
  const report = useCheckReport(audit, overview);
  const fallbackKind = checkItemKind(audit);
  const entries = useMemo(
    () =>
      buildEntries(coverage.items, items.items, findings.items, fallbackKind),
    [coverage.items, fallbackKind, findings.items, items.items],
  );
  const kind = entries[0]?.kind ?? fallbackKind;
  const group = parseGroup(params.get("result"));
  const search = params.get("q") ?? "";
  // The list shows the matching items grouped by kind; J / K, "Endpoint N
  // of M" and previous / next follow that same order.
  const groups = useMemo(
    () => groupByKind(filterEntries(entries, group, search)),
    [entries, group, search],
  );
  const visible = useMemo(
    () => groups.flatMap((candidate) => candidate.entries),
    [groups],
  );
  const links = useMemo(
    () => checkLinks(audit.projectId, audit.auditId, params, section),
    [audit.auditId, audit.projectId, params, section],
  );
  const subjects = useSubjects(entries, items.items);
  const selectedId =
    section === "coverage" ? hashedItem(location.hash) : undefined;
  const selected =
    selectedId === undefined
      ? undefined
      : entries.find((entry) => entry.row.itemId === selectedId);
  const visibleIndex =
    selectedId === undefined
      ? -1
      : visible.findIndex((entry) => entry.row.itemId === selectedId);

  function go(to: To, replace = false) {
    void navigate(to, {
      state: location.state,
      preventScrollReset: true,
      replace,
    });
  }
  // J / K walk "All activity" and then the shown items.
  const { containerProps } = useListNavigation({
    count: visible.length + 1,
    index: overview ? 0 : visibleIndex >= 0 ? visibleIndex + 1 : -1,
    enabled: overview || section === "coverage",
    onMove: (next) => {
      // On the item list without a shown selection, J and K start at the
      // first item rather than leaving the list.
      const target =
        next === 0 && section === "coverage" && visibleIndex < 0 ? 1 : next;
      const entry = visible[target - 1];
      if (target === 0) go(links.overview);
      else if (entry !== undefined) go(links.item(entry.row.itemId));
    },
  });

  // A deep link (#check-<itemId>) shows its row in the list.
  const listReady = !coverage.isPending;
  useEffect(() => {
    if (selectedId === undefined || !listReady) return;
    const row = document.getElementById(`check-${selectedId}`);
    if (row !== null && typeof row.scrollIntoView === "function")
      row.scrollIntoView({ block: "nearest" });
  }, [listReady, selectedId]);
  const technicalRequested = overview && location.hash === "#technical-details";
  useEffect(() => {
    if (!technicalRequested) return;
    const target = document.getElementById("technical-details");
    if (target !== null && typeof target.scrollIntoView === "function")
      target.scrollIntoView({ block: "start" });
  }, [technicalRequested]);

  function setListParam(key: "result" | "q", value: string) {
    const next = new URLSearchParams(params);
    if (value === "" || value === "all") next.delete(key);
    else next.set(key, value);
    go(
      {
        pathname: location.pathname,
        search: next.toString() === "" ? "" : `?${next.toString()}`,
        hash: location.hash,
      },
      true,
    );
  }
  function clearFilters() {
    const next = new URLSearchParams(params);
    next.delete("result");
    next.delete("q");
    go(
      {
        pathname: location.pathname,
        search: next.toString() === "" ? "" : `?${next.toString()}`,
        hash: location.hash,
      },
      true,
    );
  }
  function refreshList() {
    if (pin === null) {
      void coverage.refetch();
      if (audit.state !== "draft") void items.refetch();
      void findings.refetch();
      return;
    }
    // Refresh starts a new context: drop the pin, keep the filters.
    const next = new URLSearchParams(params);
    next.delete("auditRevision");
    go(
      {
        pathname: location.pathname,
        search: next.toString() === "" ? "" : `?${next.toString()}`,
        hash: location.hash,
      },
      true,
    );
  }

  const singular = itemNoun(kind, 1);
  const plural = itemNoun(kind, 2);
  const state = checkStateLabel(audit.state);
  let content: ReactNode;
  switch (section) {
    case "overview":
      content = (
        <CheckActivity
          audit={audit}
          kind={kind}
          entries={entries}
          findings={findings}
          workspace={workspace.data}
          waiting={waiting}
          report={report}
          subjects={subjects}
          links={links}
          technicalOpen={technicalRequested}
        />
      );
      break;
    case "coverage": {
      if (selectedId === undefined)
        content = (
          <EmptyState title={`Choose ${withArticle(singular)}`}>
            Pick {withArticle(singular)} in the list to read its result, the
            possible issues found on it and what the AI looked at.
          </EmptyState>
        );
      else if (selected === undefined)
        content = coverage.isPending ? (
          <p className="checks-quiet" role="status">
            Loading {plural}…
          </p>
        ) : (
          <EmptyState title={`This ${singular} is not in the list`}>
            It may belong to another round, or it is beyond the {plural} loaded
            so far.
          </EmptyState>
        );
      else {
        const previous =
          visibleIndex > 0 ? visible[visibleIndex - 1] : undefined;
        const next = visibleIndex >= 0 ? visible[visibleIndex + 1] : undefined;
        const prefix =
          selected.operation === undefined
            ? ""
            : commonPathPrefix(
                entries.flatMap((entry) =>
                  entry.operation === undefined ? [] : [entry.operation.path],
                ),
              );
        const standard = selected.item?.origin.standard;
        const area =
          selected.operation === undefined
            ? standard === undefined
              ? undefined
              : (audit.baseline?.standards.find(
                  (candidate) =>
                    candidate.reference.scheme === standard.scheme &&
                    candidate.reference.version === standard.version,
                )?.title ?? `${standard.scheme} ${standard.version}`)
            : (() => {
                const name = endpointArea(selected.operation.path, prefix);
                return name === undefined ? undefined : `${name} area`;
              })();
        content = (
          <CheckItemView
            key={selected.row.itemId}
            audit={audit}
            entry={selected}
            items={items}
            place={
              visibleIndex < 0
                ? undefined
                : { index: visibleIndex, count: visible.length }
            }
            area={area}
            previous={
              previous === undefined
                ? undefined
                : links.item(previous.row.itemId)
            }
            next={next === undefined ? undefined : links.item(next.row.itemId)}
          />
        );
      }
      break;
    }
    case "findings":
      content = (
        <div className="audit-page checks-legacy">
          <AuditFindings audit={audit} />
        </div>
      );
      break;
    case "reviews":
      content = (
        <AuditReviews
          audit={audit}
          kind={kind}
          subjects={subjects}
          items={items}
        />
      );
      break;
    case "runs":
      content = (
        <TechnicalDetails
          summary="Runs"
          description="The workflow runs this check started, one per attempt, for admins and debugging."
          defaultOpen
        >
          <AuditRuns
            audit={audit}
            kind={kind}
            items={items}
            entries={entries}
          />
        </TechnicalDetails>
      );
      break;
    case "report":
      content = (
        <div className="audit-page checks-legacy">
          <AuditReportView audit={audit} api={api} />
        </div>
      );
      break;
  }
  const detailLabel =
    section === "coverage"
      ? `Selected ${singular}`
      : section === "overview"
        ? "All activity"
        : sectionLabel(section, kind);
  return (
    <PaneLayout
      listLabel={`Check and ${plural}`}
      detailLabel={detailLabel}
      showDetail={section !== "coverage" || selectedId !== undefined}
      backLink={{ to: links.list, label: `Back to ${plural}` }}
      list={
        <CheckListPane
          audit={audit}
          projectName={projectName}
          kind={kind}
          entries={entries}
          groups={groups}
          shown={visible.length}
          group={group}
          search={search}
          coverage={coverage}
          pin={pin}
          activitySelected={overview}
          selectedItemId={selectedId}
          links={links}
          navigation={containerProps}
          onFilter={setListParam}
          onClearFilters={clearFilters}
          onRefresh={refreshList}
        />
      }
      detail={
        <DetailPane
          header={
            <div className="checks-detail-header">
              <p className="checks-narrow-identity">
                {auditProfileLabel(audit)} · {state.label}
              </p>
              <ReturnContext />
              {projectError === null ? null : (
                <ErrorNotice error={projectError} />
              )}
              <CheckProgress
                audit={audit}
                projectName={projectName}
                projectId={audit.projectId}
                kind={kind}
                entries={entries}
                partialList={coverage.truncated}
                pin={pin}
                workspace={workspace}
                links={links}
              />
              <SectionTabs section={section} kind={kind} links={links} />
            </div>
          }
        >
          {content}
        </DetailPane>
      }
    />
  );
}

/** A message in place of the check page (loading, errors). */
function CheckMessage({ children }: { children: ReactNode }) {
  return <section className="checks-page-message">{children}</section>;
}

/**
 * The check page, /projects/:projectId/audits/:auditId[/:section]: the
 * check's items in the list pane, and in the detail pane its progress and
 * controls, then the section (Overview, the items, Possible issues,
 * Decisions, Runs, Report). The old `checks` section redirects to the items.
 */
export function ProjectAuditDetailRoute() {
  const location = useLocation();
  const api = usePublicAPI();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { projectId = "", auditId = "", section: rawSection } = useParams();
  const validRoute =
    PROJECT_ID_PATTERN.test(projectId) && AUDIT_ID_PATTERN.test(auditId);
  const section = parseSection(rawSection);
  const project = useQuery({
    queryKey: queryKeys.projects.detail(projectId),
    queryFn: () => getProject(api, projectId),
    enabled: validRoute,
  });
  const audit = useQuery({
    queryKey: queryKeys.audits.detail(auditId),
    queryFn: () => getAudit(api, auditId),
    enabled: validRoute,
    refetchInterval: (query) =>
      auditPollInterval(
        query.state.data === undefined ? [] : [query.state.data],
        1_000,
      ),
    refetchOnReconnect: true,
    refetchOnWindowFocus: true,
    retry: (count, error) =>
      !(error instanceof PublicAPIError && error.status === 404) && count < 2,
  });
  useEffect(() => {
    if (audit.error instanceof PublicAPIError && audit.error.status === 404) {
      void queryClient.invalidateQueries({
        queryKey: queryKeys.projects.audits.all(projectId),
      });
      void navigate(`/projects/${encodeURIComponent(projectId)}/audits`, {
        replace: true,
      });
    }
  }, [audit.error, navigate, projectId, queryClient]);
  useDocumentTitle(
    audit.data === undefined
      ? "Check"
      : `${auditProfileLabel(audit.data)}${project.data === undefined ? "" : ` · ${project.data.name}`}`,
  );

  // The Checks tab was folded into the item list; keep old links working.
  if (rawSection === "checks" && validRoute)
    return (
      <Navigate
        to={{
          pathname: `/projects/${encodeURIComponent(projectId)}/audits/${encodeURIComponent(auditId)}/coverage`,
          search: location.search,
          hash: location.hash,
        }}
        replace
        state={location.state}
      />
    );

  if (!validRoute)
    return (
      <CheckMessage>
        <ErrorNotice error={new Error("This check address is invalid.")} />
        <Link to="/checks">All checks</Link>
      </CheckMessage>
    );
  if (audit.isPending)
    return (
      <CheckMessage>
        <p className="checks-quiet" role="status">
          Loading check…
        </p>
      </CheckMessage>
    );
  if (audit.error !== null)
    return (
      <CheckMessage>
        <ErrorNotice
          error={audit.error}
          onRetry={() => void audit.refetch()}
          retryPending={audit.isFetching}
        />
      </CheckMessage>
    );
  if (audit.data.projectId !== projectId)
    return (
      <CheckMessage>
        <ErrorNotice
          error={new Error("This check does not belong to this project.")}
        />
      </CheckMessage>
    );
  return (
    <CheckPage
      audit={audit.data}
      projectName={project.data?.name}
      projectError={project.error}
      section={section}
    />
  );
}
