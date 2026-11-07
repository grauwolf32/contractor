import type { components } from "../../../api/generated/public";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link, useLocation, useParams, useSearchParams } from "react-router";
import {
  auditPollInterval,
  listProjectAudits,
  type Audit,
} from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { conformsTo } from "../../../api/contract";
import { getProject } from "../../../api/projects";
import { queryKeys } from "../../../api/query-keys";
import { ContextLink, ReturnLink } from "../../../app/context-navigation";
import { CursorControls } from "../../../app/cursor-controls";
import { useDocumentTitle } from "../../../app/document-title";
import { formatNumber } from "../../../app/format";
import { Icon } from "../../../app/icon";
import { useURLCursorStack } from "../../../app/pagination";
import { pollInterval } from "../../../app/polling";
import { QueryView } from "../../../app/query-view";
import { RecordedTime } from "../../../app/recorded-time";
import { checkStateLabel } from "../../../app/vocabulary";
import {
  ButtonLink,
  DetailHeader,
  DetailPane,
  EmptyState,
  Facts,
  IdChip,
  ListPane,
  ListRow,
  ListSection,
  PaneLayout,
  StatusChip,
  StatusGlyph,
  TechnicalDetails,
} from "../../../ui";
import { useCheckWorkspace } from "./check-data";
import { timeLimitText } from "./check-format";
import { checkPath } from "./check-links";
import { AuditControls } from "./controls";
import { AuditFindings } from "./audit-findings";
import { AuditHypotheses } from "./hypotheses";
import { AuditActivity } from "./activity";
import { auditProfileLabel } from "./labels";
import { ResearchMaterialCard } from "./material-card";
import { StopReasonBanner } from "./progress";
import {
  useResearchHypotheses,
  useResearchMaterials,
  useResearchTasks,
  useResearchProgram,
} from "./research-data";
import { ResearchTaskTimeline, ResearchWork } from "./research-work";
import "./checks.css";
import "./research.css";

const SECTIONS = [
  ["overview", "Overview"],
  ["work", "Work"],
  ["hypotheses", "Hypotheses"],
  ["findings", "Possible issues"],
  ["materials", "Materials"],
  ["activity", "Activity"],
] as const;
type Section = (typeof SECTIONS)[number][0];
function href(audit: Audit, section: Section) {
  return (
    checkPath(audit.projectId, audit.auditId) +
    (section === "overview" ? "" : `/${section}`)
  );
}

function CheckContext({
  audit,
  projectName,
}: {
  audit: Audit;
  projectName: string;
}) {
  const api = usePublicAPI();
  const location = useLocation();
  const pages = useURLCursorStack({
    param: "contextCursor",
    navigateOptions: { state: location.state },
  });
  const query = useQuery({
    queryKey: queryKeys.projects.audits.list(audit.projectId, pages.cursor),
    queryFn: () =>
      listProjectAudits(api, {
        projectId: audit.projectId,
        ...(pages.cursor ? { cursor: pages.cursor } : {}),
      }),
    refetchInterval: pollInterval<components["schemas"]["AuditPage"]>({
      interval: 5000,
      settled: (page) => auditPollInterval(page.items) === false,
    }),
  });
  return (
    <ListPane
      title="Checks"
      subtitle={projectName}
      actions={
        <ButtonLink
          size="sm"
          variant="primary"
          to={`/checks/new?project=${encodeURIComponent(audit.projectId)}`}
        >
          Start a check
        </ButtonLink>
      }
      toolbar={
        <div className="research-context-links">
          <ReturnLink
            to={`/checks?project=${encodeURIComponent(audit.projectId)}`}
            label="All checks"
          />
          <Link to={`/projects/${encodeURIComponent(audit.projectId)}`}>
            Open Project →
          </Link>
        </div>
      }
    >
      <QueryView query={query}>
        {(data) => (
          <>
            <ListSection>
              {data.items.map((check) => {
                const state = checkStateLabel(check.state);
                return (
                  <ListRow
                    key={check.auditId}
                    to={checkPath(check.projectId, check.auditId)}
                    state={location.state}
                    selected={check.auditId === audit.auditId}
                    title={auditProfileLabel(check)}
                    glyph={<StatusGlyph tone={state.tone} />}
                    meta={[
                      <StatusChip key="state" tone={state.tone} size="sm">
                        {state.label}
                      </StatusChip>,
                      check.scope.objective,
                    ]}
                  />
                );
              })}
            </ListSection>
            <div className="research-context-footer">
              <CursorControls
                label="Project check pages"
                {...pages.controls(data.page)}
              />
            </div>
          </>
        )}
      </QueryView>
    </ListPane>
  );
}

function Materials({ audit }: { audit: Audit }) {
  const { query, pages } = useResearchMaterials(audit);
  const [params] = useSearchParams();
  let selected: unknown;
  try {
    selected = JSON.parse(params.get("source") ?? "null");
  } catch {
    selected = undefined;
  }
  const source = conformsTo("WorkflowMaterialDescriptor", selected)
    ? (selected as {
        ref: { namespace: string; name: string; revision: string };
        digest: string;
      })
    : undefined;
  return (
    <section className="research-section">
      <header className="research-section-head">
        <div>
          <p className="eyebrow">Retained exact versions</p>
          <h2>Materials</h2>
          <p className="ui-caption">
            Accepted inputs, grounds and outputs. Preview reads only the
            selected exact version.
          </p>
        </div>
      </header>
      <QueryView query={query}>
        {(data) => (
          <>
            {source &&
            !data.items.some(
              (e) =>
                e.material.ref.namespace === source.ref.namespace &&
                e.material.ref.name === source.ref.name &&
                e.material.ref.revision === source.ref.revision &&
                e.material.digest === source.digest,
            ) ? (
              <div className="checks-notice" role="status">
                <strong>
                  Selected ground: {source.ref.name}@{source.ref.revision}
                </strong>
                <p>
                  This exact material is not on the loaded page. Use the
                  material pages to locate its retained receipt.
                </p>
              </div>
            ) : null}
            {data.items.length === 0 ? (
              <EmptyState title="No accepted materials yet">
                Outputs appear after collection. A saved proposal in an active
                Run is not a registered hypothesis.
              </EmptyState>
            ) : (
              data.items.map((entry) => (
                <ResearchMaterialCard
                  key={entry.key}
                  audit={audit}
                  entry={entry}
                  selected={
                    source?.ref.revision === entry.material.ref.revision &&
                    source.ref.name === entry.material.ref.name &&
                    source.ref.namespace === entry.material.ref.namespace &&
                    source.digest === entry.material.digest
                  }
                />
              ))
            )}
            <CursorControls
              label="Material pages"
              {...pages.controls(data.page)}
            />
          </>
        )}
      </QueryView>
    </section>
  );
}

function ProgramDetails({ audit }: { audit: Audit }) {
  const query = useResearchProgram(audit);
  return (
    <QueryView query={query}>
      {(profile) => (
        <>
          <p>{auditProfileLabel(audit)}</p>
          <p className="ui-caption">
            {profile.program?.continuation
              ? `Optional continuation: ${profile.program.continuation.workflow}. It can propose hypotheses and work after newly collected results, within the pinned budget.`
              : "No automatic continuation is configured."}
          </p>
          <p className="ui-caption">
            Hypothesis registration and checking work are separate. A saved
            proposal is not a registered hypothesis.
          </p>
          <Facts
            size="sm"
            layout="list"
            items={Object.entries(audit.inputs).map(([term, value]) => ({
              term,
              value: (
                <code>
                  {value.ref.name}@{value.ref.revision}
                </code>
              ),
            }))}
          />
        </>
      )}
    </QueryView>
  );
}

export function ResearchAudit({ audit }: { audit: Audit }) {
  const { section: rawSection } = useParams();
  const section = rawSection ?? "overview";
  const api = usePublicAPI();
  const [programOpen, setProgramOpen] = useState(false);
  const project = useQuery({
    queryKey: queryKeys.projects.detail(audit.projectId),
    queryFn: () => getProject(api, audit.projectId),
  });
  const workspace = useCheckWorkspace(audit, audit.state !== "draft");
  const tasks = useResearchTasks(audit);
  const hypotheses = useResearchHypotheses(
    audit,
    section === "overview" || section === "hypotheses",
  );
  const state = checkStateLabel(audit.state);
  const title = auditProfileLabel(audit);
  useDocumentTitle(title);
  const current =
    tasks.query.data?.items.filter((task) => task.state !== "settled") ?? [];
  const collected =
    tasks.query.data?.items.filter((task) => task.state === "settled") ?? [];
  const valid = SECTIONS.some(([key]) => key === section);
  return (
    <PaneLayout
      className="research-panes"
      listLabel="Project checks"
      detailLabel="Check research"
      showDetail
      backLink={{
        to: `/checks?project=${encodeURIComponent(audit.projectId)}`,
        label: "Back to checks",
      }}
      list={
        <CheckContext
          audit={audit}
          projectName={project.data?.name ?? audit.projectId}
        />
      }
      detail={
        <DetailPane
          header={
            <DetailHeader
              titleAs="h1"
              title={title}
              eyebrow="Check research"
              breadcrumb={[
                {
                  label: project.data?.name ?? audit.projectId,
                  to: `/projects/${encodeURIComponent(audit.projectId)}`,
                },
                {
                  label: "Checks",
                  to: `/checks?project=${encodeURIComponent(audit.projectId)}`,
                },
              ]}
              status={<StatusChip tone={state.tone}>{state.label}</StatusChip>}
              meta={
                <>
                  <span>
                    {audit.startedAt ? "Started" : "Created"}{" "}
                    <RecordedTime value={audit.startedAt ?? audit.createdAt} />
                  </span>
                  {timeLimitText(audit) ? (
                    <span>{timeLimitText(audit)}</span>
                  ) : null}
                  <ContextLink
                    returnLabel="Check"
                    to={`/reports/${encodeURIComponent(audit.auditId)}`}
                  >
                    Report →
                  </ContextLink>
                </>
              }
              actions={
                <AuditControls audit={audit} projectName={project.data?.name} />
              }
            />
          }
        >
          <div className="research-body">
            <StopReasonBanner audit={audit} />
            {audit.state === "paused" ? (
              <p className="research-pause-note">
                New Audit Runs are paused. Admitted work can finish and collect;
                accepted tasks remain for Continue.
              </p>
            ) : null}
            {audit.scope.objective ? (
              <section
                className="research-goal"
                aria-label="Research objective"
              >
                <span className="research-symbol">
                  <Icon name="evals" />
                </span>
                <div>
                  <p className="eyebrow">Research objective</p>
                  <p>{audit.scope.objective}</p>
                  {audit.scope.target ? (
                    <p className="ui-caption">{audit.scope.target}</p>
                  ) : null}
                </div>
                <TechnicalDetails
                  summary="Scope"
                  defaultOpen={audit.state === "draft"}
                >
                  <Facts
                    layout="list"
                    items={[
                      { term: "Target", value: audit.scope.target },
                      {
                        term: "Authorization",
                        value: audit.scope.authorizationScope,
                      },
                    ]}
                  />
                </TechnicalDetails>
              </section>
            ) : null}
            {audit.state !== "draft" ? (
              <QueryView
                query={workspace}
                loading={<p role="status">Loading progress…</p>}
              >
                {(snapshot) => (
                  <section
                    className="research-progress"
                    aria-label="Task collection progress"
                  >
                    <header>
                      <h2>Work progress</h2>
                      <span>
                        {formatNumber(snapshot.totalTasks)} accepted tasks
                      </span>
                    </header>
                    <div
                      className="research-progress-track"
                      role="img"
                      aria-label={`${snapshot.completedTasks} completed, ${snapshot.gaps} technical gaps, ${snapshot.pendingTasks} pending`}
                    >
                      {[
                        ["done", snapshot.completedTasks],
                        ["gap", snapshot.gaps],
                        ["pending", snapshot.pendingTasks],
                      ].map(([key, count]) => (
                        <span
                          key={key}
                          data-kind={key}
                          style={{
                            width: `${snapshot.totalTasks === 0 ? 0 : (Number(count) * 100) / snapshot.totalTasks}%`,
                          }}
                        />
                      ))}
                    </div>
                    <div className="research-progress-legend">
                      <Link to={href(audit, "work")}>
                        <StatusGlyph tone="info" />
                        {snapshot.completedTasks} completed
                      </Link>
                      <Link to={`${href(audit, "work")}?work=gap`}>
                        <StatusGlyph tone="warning" />
                        {snapshot.gaps} technical gaps
                      </Link>
                      <Link to={`${href(audit, "work")}?work=pending`}>
                        <StatusGlyph tone="idle" />
                        {snapshot.pendingTasks} pending
                      </Link>
                      <span>{snapshot.outstandingRuns} Runs in progress</span>
                    </div>
                    {snapshot.auditRevision < audit.revision ? (
                      <p className="ui-caption" role="status">
                        Progress is from an earlier snapshot; refreshing…
                      </p>
                    ) : null}
                  </section>
                )}
              </QueryView>
            ) : (
              <p className="ui-caption">
                Draft check · work is created after Start.
              </p>
            )}
            <nav
              className="research-tabs section-navigation"
              aria-label="Check sections"
            >
              {SECTIONS.map(([key, label]) => (
                <Link
                  key={key}
                  to={href(audit, key)}
                  aria-current={section === key ? "page" : undefined}
                >
                  {label}
                  {key === "work" ? <span>{audit.taskCount}</span> : null}
                </Link>
              ))}
            </nav>
            {!valid ? (
              <EmptyState title="Unknown check section">
                <Link to={href(audit, "overview")}>Overview</Link>
              </EmptyState>
            ) : null}
            {section === "overview" ? (
              <div className="research-overview">
                <div>
                  <QueryView
                    query={tasks.query}
                    loading={<p role="status">Loading tasks…</p>}
                  >
                    {(data) => (
                      <>
                        <section className="research-section">
                          <header className="research-section-head">
                            <div>
                              <p className="eyebrow">In progress</p>
                              <h2>Current work</h2>
                            </div>
                            <Link to={href(audit, "work")}>All tasks →</Link>
                          </header>
                          {current.length ? (
                            <ResearchTaskTimeline
                              audit={audit}
                              tasks={current.slice(0, 3)}
                            />
                          ) : (
                            <p className="ui-caption">
                              No pending work on this page.
                            </p>
                          )}
                        </section>
                        <section className="research-section">
                          <header className="research-section-head">
                            <div>
                              <p className="eyebrow">Collected work</p>
                              <h2>Results &amp; observations</h2>
                            </div>
                          </header>
                          <ResearchTaskTimeline
                            audit={audit}
                            tasks={collected.slice(0, 4)}
                          />
                          {collected.length === 0 ? (
                            <p className="ui-caption">
                              No collected results on this page.
                            </p>
                          ) : null}
                        </section>
                        <p className="ui-caption">
                          Showing the current task page
                          {data.page.hasMore
                            ? "; more pages are available"
                            : ""}
                          . Collection progress is separate from a security
                          verdict.
                        </p>
                        <CursorControls
                          label="Task pages"
                          {...tasks.pages.controls(data.page)}
                        />
                      </>
                    )}
                  </QueryView>
                </div>
                <aside className="research-support">
                  <section className="research-coverage">
                    <p className="eyebrow">Fixed checklist</p>
                    <QueryView query={hypotheses.query}>
                      {(page) => (
                        <>
                          <h2>
                            {page.fixed_denominator} original requirements
                          </h2>
                          <p>
                            {page.pending_fixed_count
                              ? `${page.pending_fixed_count} awaiting admission; they remain unchecked.`
                              : "The original denominator is retained."}
                          </p>
                          <p className="ui-caption">
                            Generated and manual hypotheses are additional.
                            Per-hypothesis checks show their recorded results
                            and tested conditions.
                          </p>
                          <Link to={href(audit, "hypotheses")}>
                            View hypotheses →
                          </Link>
                        </>
                      )}
                    </QueryView>
                  </section>
                  <section className="research-finding-summary">
                    <span className="research-symbol" data-tone="warning">
                      <Icon name="issues" />
                    </span>
                    <div>
                      <p className="eyebrow">Possible issues</p>
                      <h2>
                        {workspace.data
                          ? `${workspace.data.unreviewedFindings} need review`
                          : "Project findings"}
                      </h2>
                      <p className="ui-caption">
                        Independent origins and analyst decisions stay with the
                        Project issue.
                      </p>
                      <Link to={href(audit, "findings")}>Inspect issues →</Link>
                    </div>
                  </section>
                  {workspace.data?.gaps ? (
                    <section className="research-gaps">
                      <h3>What is still unknown</h3>
                      <p>
                        {workspace.data.gaps} tasks have technical gaps. Inspect
                        their recorded causes and any retained results.
                      </p>
                      <Link to={`${href(audit, "work")}?work=gap`}>
                        Inspect gaps →
                      </Link>
                    </section>
                  ) : null}
                  <TechnicalDetails
                    summary="Program & selected materials"
                    onToggle={setProgramOpen}
                  >
                    {programOpen ? <ProgramDetails audit={audit} /> : null}
                  </TechnicalDetails>
                </aside>
              </div>
            ) : null}
            {section === "work" ? <ResearchWork audit={audit} /> : null}
            <div hidden={section !== "hypotheses"}>
              <AuditHypotheses
                audit={audit}
                expanded
                active={section === "hypotheses"}
              />
            </div>
            {section === "findings" ? <AuditFindings audit={audit} /> : null}
            {section === "materials" ? <Materials audit={audit} /> : null}
            {section === "activity" ? (
              <AuditActivity audit={audit} expanded />
            ) : null}
            <TechnicalDetails description="Exact check identity, baseline and configuration.">
              <Facts
                size="sm"
                layout="list"
                items={[
                  {
                    term: "Check ID",
                    value: <IdChip value={audit.auditId} label="check ID" />,
                  },
                  {
                    term: "Version",
                    value: `${audit.profile.name}@${audit.profile.version}`,
                  },
                  {
                    term: "Digest",
                    value: <code>{audit.profile.digest}</code>,
                  },
                  { term: "Submitted Runs", value: audit.submittedRunCount },
                  { term: "Iteration", value: audit.iteration },
                  { term: "Revision", value: audit.revision },
                ]}
              />
            </TechnicalDetails>
            <Link
              className="research-project-issues"
              to={`/projects/${encodeURIComponent(audit.projectId)}/findings`}
            >
              Project possible issues
            </Link>
          </div>
        </DetailPane>
      }
    />
  );
}
