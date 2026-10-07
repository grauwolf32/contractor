import { useMemo } from "react";
import { useNavigate, useSearchParams } from "react-router";

import {
  useAllChecks,
  useProjectsIndex,
  type CrossProjectCheck,
} from "../../api/cross-project";
import { useDocumentTitle } from "../../app/document-title";
import {
  DetailPane,
  EmptyState,
  PaneLayout,
  useListNavigation,
} from "../../ui";
import { useCheckWorkspaces } from "../projects/audits/check-data";
import { checkPath } from "../projects/audits/check-links";
import { CheckSummary } from "./list/check-summary";
import { ChecksListPane } from "./list/checks-list";
import {
  CHECK_FILTERS,
  checksHref,
  inFilter,
  parseCheck,
  parseCheckFilter,
  parseProject,
  type CheckFilter,
} from "./list/filters";

import "../projects/audits/checks.css";

/** Rows whose progress line is read: the first ones the filters show. */
const PROGRESS_ROWS = 30;

/**
 * Checks: every check across the first page of projects
 * (/checks?state=<filter>&project=<projectId>&check=<auditId>). The list
 * filters by state and project; the selected check shows its progress, what
 * waits for the user and its controls, and opens the check page.
 */
export function ChecksRoute() {
  useDocumentTitle("Checks");
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const filter = parseCheckFilter(params.get("state"));
  const projectId = parseProject(params.get("project"));
  const checkId = parseCheck(params.get("check"));
  const checks = useAllChecks();
  const index = useProjectsIndex();
  const inProject = useMemo(
    () =>
      projectId === undefined
        ? checks.checks
        : checks.checks.filter(
            (check) => check.project.projectId === projectId,
          ),
    [checks.checks, projectId],
  );
  const counts = useMemo(() => {
    const result = {} as Record<CheckFilter, number>;
    for (const option of CHECK_FILTERS)
      result[option.value] = inProject.filter((check) =>
        inFilter(option.value, check.audit.state),
      ).length;
    return result;
  }, [inProject]);
  const rows = useMemo(
    () => inProject.filter((check) => inFilter(filter, check.audit.state)),
    [filter, inProject],
  );
  const progressAudits = useMemo(
    () =>
      rows
        .filter(
          (check) =>
            check.audit.state !== "draft" && check.audit.state !== "deleting",
        )
        .slice(0, PROGRESS_ROWS)
        .map((check) => check.audit),
    [rows],
  );
  const workspaces = useCheckWorkspaces(progressAudits);
  const listed: CrossProjectCheck | undefined =
    checkId === undefined
      ? undefined
      : checks.checks.find((check) => check.audit.auditId === checkId);
  const href = (next: {
    filter?: CheckFilter;
    projectId?: string | undefined;
    checkId?: string | undefined;
  }) =>
    checksHref({
      filter: next.filter ?? filter,
      projectId: "projectId" in next ? next.projectId : projectId,
      checkId: "checkId" in next ? next.checkId : checkId,
    });
  const selectedIndex =
    checkId === undefined
      ? -1
      : rows.findIndex((check) => check.audit.auditId === checkId);
  const { containerProps } = useListNavigation({
    count: rows.length,
    index: selectedIndex,
    onMove: (next) => {
      const check = rows[next];
      if (check !== undefined)
        void navigate(href({ checkId: check.audit.auditId }));
    },
    onOpen: (position) => {
      const check = rows[position];
      if (check !== undefined)
        void navigate(checkPath(check.project.projectId, check.audit.auditId));
    },
  });
  const start =
    projectId === undefined
      ? "/checks/new"
      : `/checks/new?${new URLSearchParams({ project: projectId }).toString()}`;
  return (
    <PaneLayout
      listLabel="Checks"
      detailLabel="Selected check"
      showDetail={checkId !== undefined}
      backLink={{ to: href({ checkId: undefined }), label: "Back to checks" }}
      list={
        <ChecksListPane
          checks={checks}
          projects={index.projects}
          rows={rows}
          counts={counts}
          filter={filter}
          projectId={projectId}
          selectedId={checkId}
          workspaces={workspaces}
          rowHref={(auditId) => href({ checkId: auditId })}
          startHref={start}
          navigation={containerProps}
          onFilter={(value) =>
            void navigate(href({ filter: value }), { replace: true })
          }
          onProject={(value) =>
            void navigate(href({ projectId: value }), { replace: true })
          }
        />
      }
      detail={
        checkId === undefined ? (
          <DetailPane>
            <EmptyState title="Choose a check">
              Pick a check to see how far it got, what waits for you and its
              controls.
            </EmptyState>
          </DetailPane>
        ) : (
          <CheckSummary key={checkId} auditId={checkId} listed={listed} />
        )
      }
    />
  );
}
