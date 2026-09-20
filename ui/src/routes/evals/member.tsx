import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { usePublicAPI } from "../../api/context";
import {
  EVAL_POLL_MS,
  listEvalExecutions,
  type EvalMember,
} from "../../api/evals";
import { ContextLink } from "../../app/context-navigation";
import { EvalError } from "./common";

function memberQuality(member: EvalMember): string {
  if (member.conflicting) return "Conflicting evidence";
  if (member.member.eligibility !== "eligible")
    return member.member.eligibility;
  if (!member.execution || member.execution.state === "not_submitted")
    return "Not submitted";
  if (member.execution.state === "failed") return "Failed execution";
  if (member.execution.state === "cancelled") return "Cancelled execution";
  if (member.assessment === "fail") return "Failed check";
  if (member.assessment === "unscored")
    return member.resultSha256
      ? "Awaiting assessment / review"
      : "Results not collected";
  return {
    pass: "Passed checks",
    error: "Assessment error",
    incomplete: "Incomplete assessment",
  }[member.assessment];
}

export function MemberSummary({ member }: { member: EvalMember }) {
  const usage = member.usage;
  return (
    <>
      <strong>{memberQuality(member)}</strong>
      <p>Execution: {member.execution?.state ?? "not submitted"}</p>
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
    </>
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
    queryKey: ["evals", "inventory", id, member.member.memberId, cursor],
    queryFn: () => listEvalExecutions(api, id, member.member.memberId, cursor),
    refetchInterval: (query) =>
      !cursor && !query.state.error && !query.state.data?.inventoryComplete
        ? EVAL_POLL_MS
        : false,
  });
  return (
    <section>
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
          <p>
            {inventory.data.inventoryComplete
              ? "Complete inventory"
              : "Incomplete inventory"}
            ; child Runs are evidence, not additional evaluation samples.
          </p>
          {inventory.data.gaps.map((gap) => (
            <p key={gap}>{gap}</p>
          ))}
          <ul>
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
                  {entry.role ?? "Parent"}
                  {entry.round === null ? "" : ` · round ${entry.round}`} ·{" "}
                  {entry.state} ·{" "}
                  {href ? (
                    <ContextLink returnLabel="Evaluation evidence" to={href}>
                      {ref?.kind} {ref?.id}
                    </ContextLink>
                  ) : (
                    <span>
                      {ref?.id ?? "Unresolved submission"} · unavailable
                    </span>
                  )}
                </li>
              );
            })}
          </ul>
          <div className="eval-actions">
            <button
              type="button"
              className="secondary-button"
              disabled={!cursor}
              onClick={() => setCursor(undefined)}
            >
              First executions
            </button>
            {inventory.data.page.hasMore ? (
              <button
                type="button"
                className="secondary-button"
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
