import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { usePublicAPI } from "../../api/context";
import {
  EVAL_POLL_MS,
  listEvalExecutions,
  type EvalMember,
} from "../../api/evals";
import { queryKeys } from "../../api/query-keys";
import { ContextLink } from "../../app/context-navigation";
import { StatusChip, type StatusTone } from "../../ui";
import { EvalError } from "./common";
import { executionRefLabel } from "./labels";

function capitalized(value: string): string {
  const words = value.replaceAll("_", " ");
  return words.charAt(0).toUpperCase() + words.slice(1);
}

/**
 * What is known about one attempt's quality. Execution success, collection
 * and assessment stay distinct: a successful run without an assessment is not
 * a pass, and a missing result is not a failure of the candidate.
 */
function memberQuality(member: EvalMember): {
  label: string;
  tone: StatusTone;
} {
  if (member.conflicting)
    return { label: "Conflicting evidence", tone: "blocked" };
  if (member.member.eligibility !== "eligible")
    return { label: capitalized(member.member.eligibility), tone: "warning" };
  if (!member.execution || member.execution.state === "not_submitted")
    return { label: "Not submitted", tone: "idle" };
  if (member.execution.state === "failed")
    return { label: "Failed execution", tone: "blocked" };
  if (member.execution.state === "cancelled")
    return { label: "Cancelled execution", tone: "neutral" };
  if (member.assessment === "fail")
    return { label: "Failed criteria", tone: "blocked" };
  if (member.assessment === "unscored")
    return member.resultSha256
      ? { label: "Awaiting assessment / review", tone: "review" }
      : { label: "Results not collected", tone: "warning" };
  return (
    {
      pass: { label: "Passed criteria", tone: "success" },
      error: { label: "Assessment error", tone: "blocked" },
      incomplete: { label: "Incomplete assessment", tone: "warning" },
    } as const
  )[member.assessment];
}

export function MemberSummary({ member }: { member: EvalMember }) {
  const usage = member.usage;
  const quality = memberQuality(member);
  return (
    <div className="eval-member">
      <p className="eval-member-quality">
        <StatusChip tone={quality.tone} size="sm">
          {quality.label}
        </StatusChip>
        <span className="eval-muted">
          Execution:{" "}
          {member.execution?.state.replaceAll("_", " ") ?? "not submitted"}
        </span>
      </p>
      {member.member.reason ? <p>{member.member.reason}</p> : null}
      <dl className="eval-facts">
        <dt>Tokens</dt>
        <dd>
          {usage?.totalTokens.value ?? "Unavailable"} ·{" "}
          {usage?.totalTokens.completeness ?? "unavailable"}
        </dd>
        <dt>Duration</dt>
        <dd>
          {usage?.wallMs.value === null || usage?.wallMs.value === undefined
            ? "Unavailable"
            : `${usage.wallMs.value} ms`}{" "}
          · {usage?.wallMs.completeness ?? "unavailable"}
        </dd>
      </dl>
      {member.execution?.ref?.kind === "run" ? (
        <ContextLink
          returnLabel="Evaluation evidence"
          to={`/runs/${encodeURIComponent(member.execution.ref.id)}`}
        >
          Open Run
        </ContextLink>
      ) : null}
    </div>
  );
}

export function MemberExecutions({
  id,
  member,
}: {
  id: string;
  member: EvalMember;
}) {
  const api = usePublicAPI();
  const [cursor, setCursor] = useState<string | undefined>();
  const inventory = useQuery({
    queryKey: queryKeys.evals.inventory(id, member.member.memberId, cursor),
    queryFn: () => listEvalExecutions(api, id, member.member.memberId, cursor),
    refetchInterval: (query) =>
      !cursor && !query.state.error && !query.state.data?.inventoryComplete
        ? EVAL_POLL_MS
        : false,
  });
  return (
    <section className="eval-executions">
      <h4>Owned executions</h4>
      <EvalError
        error={inventory.error}
        reload={() => {
          if (cursor) setCursor(undefined);
          else void inventory.refetch();
        }}
      />
      {inventory.data ? (
        <>
          <p className="eval-muted">
            {inventory.data.inventoryComplete
              ? "Complete inventory"
              : "Incomplete inventory"}
            ; child Runs are evidence, not additional evaluation samples.
          </p>
          {inventory.data.gaps.map((gap) => (
            <p className="eval-muted" key={gap}>
              {gap}
            </p>
          ))}
          <ul className="eval-execution-list" role="list">
            {inventory.data.items.map((entry, index) => {
              const ref = entry.execution;
              const href =
                !entry.available || !ref
                  ? null
                  : ref.kind === "run"
                    ? `/runs/${encodeURIComponent(ref.id)}`
                    : entry.projectId
                      ? `/projects/${encodeURIComponent(entry.projectId)}/audits/${encodeURIComponent(ref.id)}`
                      : null;
              return (
                <li key={entry.intentId ?? ref?.id ?? index}>
                  <span className="eval-muted">
                    {entry.role ?? "Parent"}
                    {entry.round === null
                      ? ""
                      : ` · round ${entry.round}`} · {entry.state}
                  </span>{" "}
                  {href && ref ? (
                    <ContextLink returnLabel="Evaluation evidence" to={href}>
                      {executionRefLabel(ref.kind)} {ref.id}
                    </ContextLink>
                  ) : (
                    <span>
                      {ref
                        ? `${executionRefLabel(ref.kind)} ${ref.id}`
                        : "Unresolved submission"}{" "}
                      · unavailable
                    </span>
                  )}
                </li>
              );
            })}
          </ul>
          <div className="eval-actions">
            <button
              type="button"
              className="ui-btn"
              data-size="xs"
              disabled={!cursor}
              onClick={() => setCursor(undefined)}
            >
              First executions
            </button>
            {inventory.data.page.hasMore ? (
              <button
                type="button"
                className="ui-btn"
                data-size="xs"
                onClick={() =>
                  setCursor(inventory.data?.page.nextCursor ?? undefined)
                }
              >
                More executions
              </button>
            ) : null}
          </div>
        </>
      ) : inventory.isPending ? (
        <p role="status">Loading execution evidence…</p>
      ) : null}
    </section>
  );
}
