import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useParams, useSearchParams } from "react-router";
import { usePublicAPI } from "../../api/context";
import {
  listEvalMembers,
  type EvalExperiment,
  type EvalMemberQuery,
} from "../../api/evals";
import { EvalChartPanel } from "./charts";
import { EvalError, EvalField, EvalFrame } from "./common";
import { EvalComparison } from "./comparison";
import { EvalControls } from "./controls";
import { MemberSummary, MemberExecutions } from "./member";
import { useEvalExperiment } from "./queries";
import { EvalReadiness } from "./readiness";
import { EvalSetupForm } from "./setup";

function Attempts({ experiment }: { experiment: EvalExperiment }) {
  const api = usePublicAPI();
  const [params, setParams] = useSearchParams();
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
      <EvalError error={members.error} reload={() => setParams({ filter })} />
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
                  className="secondary"
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
              className="secondary"
              disabled={!cursor}
              onClick={() => update({})}
            >
              First attempts
            </button>
            <button
              type="button"
              className="secondary"
              disabled={!members.data.page.hasMore}
              onClick={() => update({ cursor: members.data!.page.nextCursor! })}
            >
              Next attempts
            </button>
          </div>
        </>
      ) : (
        <p>Attempts are available after preparation.</p>
      )}
    </>
  );
}

function DraftSetup({ experiment }: { experiment: EvalExperiment }) {
  const [dirty, setDirty] = useState(false);
  return (
    <>
      <EvalSetupForm experiment={experiment} onDirtyChange={setDirty} />
      {dirty ? <p>Save your changes before preparing this draft.</p> : null}
      <EvalControls experiment={experiment} disabled={dirty} />
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
          <p>
            {data.state} · {data.executionKind} · {data.controlMode} ·{" "}
            {data.expectedMembers} expected members
          </p>
          <nav className="eval-tabs" aria-label="Experiment sections">
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
          {section === "setup" ? (
            <>
              {data.state === "draft" && data.draft ? (
                <DraftSetup
                  key={`${data.experimentId}:${data.revision}`}
                  experiment={data}
                />
              ) : (
                <>
                  <EvalReadiness experiment={data} />
                  <EvalControls key={data.experimentId} experiment={data} />
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
                  {data.summary ? (
                    <>
                      <p>
                        <strong>Conclusion: {data.summary.conclusion}</strong> ·{" "}
                        {data.freshness}
                      </p>
                      {Object.entries(data.summary.counts).map(
                        ([arm, counts]) => (
                          <p key={arm}>
                            {arm}: {counts.terminal}/{counts.expected} terminal
                            · {counts.scored}/{counts.expected} scored ·{" "}
                            {counts.endToEndPassed}/{counts.expected} end-to-end
                            passed
                          </p>
                        ),
                      )}
                    </>
                  ) : (
                    <p>Comparison is waiting for collected observations.</p>
                  )}
                  {data.deadlineAt ? (
                    <p>
                      Original deadline:{" "}
                      {new Date(data.deadlineAt).toLocaleString()}. Pausing does
                      not extend it.
                    </p>
                  ) : null}
                  {data.viewSnapshot ? (
                    <div className="eval-overview-charts">
                      <EvalChartPanel
                        experiment={data}
                        chart="quality"
                        snapshot={data.viewSnapshot}
                      />
                      <EvalChartPanel
                        experiment={data}
                        chart="progress"
                        snapshot={data.viewSnapshot}
                      />
                    </div>
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
      ) : (
        <p role="status">Loading experiment…</p>
      )}
    </EvalFrame>
  );
}
