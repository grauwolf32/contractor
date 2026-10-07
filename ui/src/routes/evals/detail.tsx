import { useQuery } from "@tanstack/react-query";
import { useId, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router";
import { usePublicAPI } from "../../api/context";
import {
  listEvalMembers,
  type EvalExperiment,
  type EvalMemberQuery,
} from "../../api/evals";
import { queryKeys } from "../../api/query-keys";
import { useDocumentTitle } from "../../app/document-title";
import { MobileSectionPicker } from "../../app/mobile-section-picker";
import { FilterChips, IdChip } from "../../ui";
import { ArmKey, EvalError, EvalFrame, EvalStateChip } from "./common";
import { EvalComparison } from "./comparison";
import { EvalControls } from "./controls";
import { EvalDiagnostics } from "./diagnostics";
import {
  ATTEMPT_FILTERS,
  controlModeLabel,
  EVAL_SECTIONS,
  executionKindLabel,
  expectedMembersText,
  experimentListPath,
  experimentPath,
  variantArm,
} from "./labels";
import { MemberExecutions, MemberSummary } from "./member";
import { EvalOverviewCharts, EvalOverviewSummary } from "./overview";
import { useEvalExperiment } from "./queries";
import { EvalReadiness } from "./readiness";
import { EvalSetupForm } from "./setup";
import { useEvalViewRefresh } from "./view-refresh";

function Attempts({ experiment }: { experiment: EvalExperiment }) {
  const api = usePublicAPI();
  const heading = useId(),
    evidence = useId();
  const [params, setParams] = useSearchParams();
  const refresh = useEvalViewRefresh(experiment, "attempts");
  const snapshot = params.get("viewSnapshot") ?? experiment.viewSnapshot;
  const cursor = params.get("cursor");
  const filter =
    ATTEMPT_FILTERS.find((f) => f.value === params.get("filter"))?.value ??
    "all";
  const query: EvalMemberQuery = {
    filter,
    ...(snapshot ? { viewSnapshot: snapshot } : {}),
    ...(cursor ? { cursor } : {}),
  };
  const members = useQuery({
    queryKey: queryKeys.evals.members(experiment.experimentId, query),
    enabled: !!snapshot,
    queryFn: ({ signal }) =>
      listEvalMembers(api, experiment.experimentId, query, signal),
  });
  const [expanded, setExpanded] = useState<string | null>(null);
  function update(next: Record<string, string>) {
    setParams({
      filter,
      ...(snapshot ? { viewSnapshot: snapshot } : {}),
      ...next,
    });
    setExpanded(null);
  }
  return (
    <section className="eval-section" aria-labelledby={heading}>
      <h2 id={heading}>Expected attempts</h2>
      <FilterChips
        label="Attempt filter"
        options={ATTEMPT_FILTERS}
        value={filter}
        // Pressing the selected chip keeps the page and the open evidence.
        onChange={(value) => {
          if (value !== filter) update({ filter: value });
        }}
      />
      <EvalError error={members.error} reload={() => void refresh()} />
      {members.data ? (
        <>
          <p className="eval-muted">
            {members.data.filteredCount} matching attempts; the experiment
            retains all {experiment.expectedMembers} expected members.
          </p>
          {members.data.items.length ? (
            <ul className="eval-records" role="list">
              {members.data.items.map((member) => {
                const id = member.member.memberId;
                const arm = variantArm(experiment, member.member.variantId);
                const open = expanded === id;
                return (
                  <li className="eval-record eval-attempt" key={id}>
                    <h3 className="eval-attempt-title">
                      {arm ? (
                        <ArmKey arm={arm}>{arm.toUpperCase()}</ArmKey>
                      ) : (
                        <span className="eval-variant-id">
                          {member.member.variantId}
                        </span>
                      )}{" "}
                      · {member.member.caseId} / sample {member.member.sample}
                    </h3>
                    <MemberSummary member={member} />
                    <div className="eval-actions">
                      <button
                        type="button"
                        className="ui-btn"
                        data-size="sm"
                        aria-expanded={open}
                        aria-controls={open ? `${evidence}-${id}` : undefined}
                        onClick={() => setExpanded(open ? null : id)}
                      >
                        Execution evidence
                      </button>
                    </div>
                    {open ? (
                      <div id={`${evidence}-${id}`}>
                        <MemberExecutions
                          id={experiment.experimentId}
                          member={member}
                        />
                      </div>
                    ) : null}
                  </li>
                );
              })}
            </ul>
          ) : (
            <p className="eval-muted">
              No attempts match this filter. Choose All to inspect every
              expected attempt.
            </p>
          )}
          <div className="eval-actions">
            <button
              type="button"
              className="ui-btn"
              data-size="sm"
              disabled={!cursor}
              onClick={() => update({})}
            >
              First attempts
            </button>
            <button
              type="button"
              className="ui-btn"
              data-size="sm"
              disabled={!members.data.page.hasMore}
              onClick={() => update({ cursor: members.data!.page.nextCursor! })}
            >
              Next attempts
            </button>
          </div>
        </>
      ) : !snapshot ? (
        <p className="eval-muted">Attempts are available after preparation.</p>
      ) : members.isPending ? (
        <p role="status">Loading attempts…</p>
      ) : null}
    </section>
  );
}

function DraftSetup({ experiment }: { experiment: EvalExperiment }) {
  const [dirty, setDirty] = useState(false);
  return (
    <EvalSetupForm
      experiment={experiment}
      onDirtyChange={setDirty}
      actions={<EvalControls experiment={experiment} disabled={dirty} />}
    />
  );
}

function SectionTabs({
  experimentId,
  section,
}: {
  experimentId: string;
  section: string;
}) {
  return (
    <div className="eval-tabs-row">
      <nav className="eval-tabs" aria-label="Experiment sections">
        {EVAL_SECTIONS.map((tab) => (
          <Link
            key={tab.value}
            aria-current={tab.value === section ? "page" : undefined}
            to={experimentPath(experimentId, tab.value)}
          >
            {tab.label}
          </Link>
        ))}
      </nav>
      <MobileSectionPicker
        label="Experiment section"
        value={`/evals/experiments/${encodeURIComponent(experimentId)}/${section}`}
        options={EVAL_SECTIONS.map((tab) => ({
          to: experimentPath(experimentId, tab.value),
          label: tab.label,
        }))}
      />
    </div>
  );
}

function Overview({ experiment }: { experiment: EvalExperiment }) {
  const heading = useId();
  return (
    <section className="eval-section" aria-labelledby={heading}>
      <h2 id={heading}>Overview</h2>
      <p className="eval-muted">
        Execution completion and assessment coverage are separate.
      </p>
      <EvalOverviewSummary experiment={experiment} />
      {experiment.deadlineAt ? (
        <p className="eval-muted">
          Original deadline: {new Date(experiment.deadlineAt).toLocaleString()}.
          Pausing does not extend it.
        </p>
      ) : null}
      {experiment.viewSnapshot ? (
        <EvalOverviewCharts
          experiment={experiment}
          snapshot={experiment.viewSnapshot}
        />
      ) : null}
      <p>
        <Link to={experimentPath(experiment.experimentId, "setup")}>
          Review setup and readiness
        </Link>
      </p>
    </section>
  );
}

export function EvalDetailRoute() {
  const { experimentId = "", section = "overview" } = useParams();
  const experiment = useEvalExperiment(experimentId);
  const data = experiment.data;
  useDocumentTitle(data?.name ?? "Experiment");
  const name = data?.name ?? "Experiment";
  return (
    <EvalFrame
      title={name}
      breadcrumb={[
        { label: "Experiments", to: experimentListPath(experimentId) },
        { label: name },
      ]}
      status={data ? <EvalStateChip state={data.state} /> : undefined}
      description={
        data ? (
          <>
            <span>
              {executionKindLabel(data.executionKind)} ·{" "}
              {controlModeLabel(data.controlMode)} ·{" "}
              {expectedMembersText(data.expectedMembers)}
            </span>
            <IdChip value={data.experimentId} label="experiment ID" />
          </>
        ) : undefined
      }
      tabs={
        data ? (
          <SectionTabs experimentId={experimentId} section={section} />
        ) : undefined
      }
    >
      <EvalError
        error={experiment.error}
        reload={() => void experiment.refetch()}
      />
      {data ? (
        <>
          <EvalDiagnostics experiment={data} />
          {section === "setup" ? (
            data.state === "draft" && data.draft ? (
              <DraftSetup
                key={`${data.experimentId}:${data.revision}`}
                experiment={data}
              />
            ) : (
              <>
                <EvalControls key={data.experimentId} experiment={data} />
                {data.state === "preparing" ? (
                  <p className="eval-callout" role="status">
                    Preparing the experiment. Compatibility checks are running;
                    this page updates automatically.
                  </p>
                ) : null}
                <EvalReadiness experiment={data} />
              </>
            )
          ) : (
            <>
              <EvalControls key={data.experimentId} experiment={data} />
              {section === "comparison" ? (
                <EvalComparison experiment={data} />
              ) : section === "attempts" ? (
                <Attempts experiment={data} />
              ) : (
                <Overview experiment={data} />
              )}
            </>
          )}
        </>
      ) : experiment.isPending ? (
        <p role="status">Loading experiment…</p>
      ) : null}
    </EvalFrame>
  );
}
