import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useParams, useSearchParams } from "react-router";
import { usePublicAPI } from "../../api/context";
import {
  listEvalMembers,
  type EvalExperiment,
  type EvalMemberQuery,
} from "../../api/evals";
import { EvalError, EvalField, EvalFrame } from "./common";
import { EvalComparison } from "./comparison";
import { EvalControls } from "./controls";
import { MemberSummary, MemberExecutions } from "./member";
import { EvalOverviewCharts, EvalOverviewSummary } from "./overview";
import { useEvalExperiment } from "./queries";
import { EvalReadiness } from "./readiness";
import { EvalSetupForm } from "./setup";
import { EvalDiagnostics } from "./diagnostics";
import { MobileSectionPicker } from "../../app/mobile-section-picker";
import { StateBadge } from "../runs/components";
import { useEvalViewRefresh } from "./view-refresh";

function Attempts({ experiment }: { experiment: EvalExperiment }) {
  const api = usePublicAPI();
  const [params, setParams] = useSearchParams();
  const refresh = useEvalViewRefresh(experiment, "attempts");
  const snapshot = params.get("viewSnapshot") ?? experiment.viewSnapshot;
  const cursor = params.get("cursor");
  const filters = [
    "all",
    "unresolved",
    "failed",
    "unscored",
    "unsupported",
    "blocked",
    "conflicting",
  ] as const;
  const filter = filters.find((f) => f === params.get("filter")) ?? "all";
  const query: EvalMemberQuery = {
    filter,
    ...(snapshot ? { viewSnapshot: snapshot } : {}),
    ...(cursor ? { cursor } : {}),
  };
  const members = useQuery({
    queryKey: ["evals", "members", experiment.experimentId, query],
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
    <>
      <h2>Expected attempts</h2>
      <EvalField label="Attempt filter">
        <select
          value={filter}
          onChange={(e) => update({ filter: e.target.value })}
        >
          {filters.map((f) => (
            <option key={f}>{f}</option>
          ))}
        </select>
      </EvalField>
      <EvalError error={members.error} reload={() => void refresh()} />
      {members.data ? (
        <>
          <p>
            {members.data.filteredCount} matching attempts; the experiment
            retains all {experiment.expectedMembers} expected members.
          </p>
          <div className="eval-attempts">
            {members.data.items.map((member) => (
              <article
                className="panel eval-panel"
                key={member.member.memberId}
              >
                <h3>
                  {member.member.variantId} · {member.member.caseId} / sample{" "}
                  {member.member.sample}
                </h3>
                <MemberSummary member={member} />
                <button
                  type="button"
                  className="secondary-button"
                  onClick={() =>
                    setExpanded(
                      expanded === member.member.memberId
                        ? null
                        : member.member.memberId,
                    )
                  }
                >
                  Execution evidence
                </button>
                {expanded === member.member.memberId ? (
                  <MemberExecutions
                    id={experiment.experimentId}
                    member={member}
                  />
                ) : null}
              </article>
            ))}
          </div>
          <div className="eval-actions">
            <button
              type="button"
              className="secondary-button"
              disabled={!cursor}
              onClick={() => update({})}
            >
              First attempts
            </button>
            <button
              type="button"
              className="secondary-button"
              disabled={!members.data.page.hasMore}
              onClick={() => update({ cursor: members.data!.page.nextCursor! })}
            >
              Next attempts
            </button>
          </div>
        </>
      ) : !snapshot ? (
        <p>Attempts are available after preparation.</p>
      ) : members.isPending ? (
        <p role="status">Loading attempts…</p>
      ) : null}
    </>
  );
}

function DraftSetup({ experiment }: { experiment: EvalExperiment }) {
  const [dirty, setDirty] = useState(false);
  return (
    <>
      <EvalSetupForm
        experiment={experiment}
        onDirtyChange={setDirty}
        actions={<EvalControls experiment={experiment} disabled={dirty} />}
      />
    </>
  );
}

export function EvalDetailRoute() {
  const { experimentId = "", section = "overview" } = useParams();
  const experiment = useEvalExperiment(experimentId);
  const data = experiment.data;
  return (
    <EvalFrame title={data?.name ?? "Experiment"}>
      <EvalError
        error={experiment.error}
        reload={() => void experiment.refetch()}
      />
      {data ? (
        <>
          <div className="eval-identity">
            <StateBadge state={data.state} />
            <span>
              {data.executionKind === "audit" ? "Audit" : "Workflow"} ·{" "}
              {data.controlMode === "server"
                ? "Server controlled"
                : "External producer"}
            </span>
            <span>{data.expectedMembers} expected members</span>
          </div>
          <EvalDiagnostics experiment={data} />
          <nav
            className="section-navigation eval-tabs"
            aria-label="Experiment sections"
          >
            {["overview", "comparison", "attempts", "setup"].map((tab) => (
              <Link
                key={tab}
                aria-current={tab === section ? "page" : undefined}
                to={`/evals/experiments/${encodeURIComponent(experimentId)}/${tab}`}
              >
                {tab[0]!.toUpperCase() + tab.slice(1)}
              </Link>
            ))}
          </nav>
          <MobileSectionPicker
            label="Experiment section"
            value={`/evals/experiments/${encodeURIComponent(experimentId)}/${section}`}
            options={["overview", "comparison", "attempts", "setup"].map(
              (tab) => ({
                to: `/evals/experiments/${encodeURIComponent(experimentId)}/${tab}`,
                label: tab[0]!.toUpperCase() + tab.slice(1),
              }),
            )}
          />
          {section === "setup" ? (
            <>
              {data.state === "draft" && data.draft ? (
                <DraftSetup
                  key={`${data.experimentId}:${data.revision}`}
                  experiment={data}
                />
              ) : (
                <>
                  <EvalControls key={data.experimentId} experiment={data} />
                  {data.state === "preparing" ? (
                    <p role="status">
                      Preparing the experiment. Compatibility checks are
                      running; this page updates automatically.
                    </p>
                  ) : null}
                  <EvalReadiness experiment={data} />
                </>
              )}
            </>
          ) : (
            <>
              <EvalControls key={data.experimentId} experiment={data} />
              {section === "comparison" ? (
                <EvalComparison experiment={data} />
              ) : section === "attempts" ? (
                <Attempts experiment={data} />
              ) : (
                <>
                  <h2>Overview</h2>
                  <p>
                    Execution completion and assessment coverage are separate.
                  </p>
                  <EvalOverviewSummary experiment={data} />
                  {data.deadlineAt ? (
                    <p>
                      Original deadline:{" "}
                      {new Date(data.deadlineAt).toLocaleString()}. Pausing does
                      not extend it.
                    </p>
                  ) : null}
                  {data.viewSnapshot ? (
                    <EvalOverviewCharts
                      experiment={data}
                      snapshot={data.viewSnapshot}
                    />
                  ) : null}
                  <Link
                    to={`/evals/experiments/${encodeURIComponent(experimentId)}/setup`}
                  >
                    Review exact setup and readiness
                  </Link>
                </>
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
