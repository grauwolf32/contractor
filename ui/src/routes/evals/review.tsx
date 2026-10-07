import { artifactHref } from "./links";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Fragment, useEffect, useId, useRef, useState } from "react";
import { usePublicAPI } from "../../api/context";
import {
  getEvalReview,
  selectEvalAssessment,
  submitEvalAssessment,
  type EvalAssessment,
  type EvalExperiment,
  type EvalMember,
} from "../../api/evals";
import { ContextLink } from "../../app/context-navigation";
import { IdChip, TechnicalDetails } from "../../ui";
import { EvalError, EvalField } from "./common";
import { useEvalOwner } from "./queries";
import { finishMutation, recoverableMutation } from "./recovery";
import { queryKeys } from "../../api/query-keys";

type Decision = EvalAssessment["checks"][number]["status"];

export function EvalHumanReview({
  experiment,
  member,
  onClose,
  onSaved,
}: {
  experiment: EvalExperiment;
  member: EvalMember;
  onClose: () => void;
  onSaved: () => void;
}) {
  const api = usePublicAPI(),
    owner = useEvalOwner(),
    cache = useQueryClient();
  const heading = useId();
  const headingRef = useRef<HTMLHeadingElement>(null);
  // The review opens below the pair; take the reader there.
  useEffect(() => {
    headingRef.current?.focus();
  }, []);
  const reviewKey = queryKeys.evals.review(
    experiment.experimentId,
    member.member.memberId,
    member.resultSha256,
  );
  const review = useQuery({
    queryKey: reviewKey,
    queryFn: () =>
      getEvalReview(api, experiment.experimentId, member.member.memberId),
    gcTime: 0,
    staleTime: Infinity,
  });
  const [decisions, setDecisions] = useState<
    Record<
      string,
      { status: Decision | ""; reason: string; evidence: string[] }
    >
  >({});
  function update(
    id: string,
    patch: Partial<{
      status: Decision | "";
      reason: string;
      evidence: string[];
    }>,
  ) {
    setDecisions((current) => ({
      ...current,
      [id]: { status: "", reason: "", evidence: [], ...current[id], ...patch },
    }));
  }
  async function readCurrentReview() {
    const context = await getEvalReview(
      api,
      experiment.experimentId,
      member.member.memberId,
    );
    cache.setQueryData(reviewKey, context);
    if (context.resultSha256 !== member.resultSha256)
      throw new Error(
        "The selected evidence changed. Reload the pair and review it again.",
      );
    return context;
  }
  const save = useMutation({
    mutationFn: async () => {
      // The panel can stay open while unrelated members advance the Eval.
      // Check its evidence again before recording an immutable assessment.
      const context = await readCurrentReview();
      if (!context.revision || !experiment.planSha256)
        throw new Error(
          "The selected evidence changed. Reload the pair and review it again.",
        );
      const checks: EvalAssessment["checks"] = context.checks.map((rubric) => {
        const decision = decisions[rubric.id],
          policy = context.policy?.find((c) => c.id === rubric.id);
        if (
          !decision?.status ||
          !decision.reason.trim() ||
          !policy?.implementationSha256
        )
          throw new Error(
            "Choose an explicit decision and explain it for every pinned rubric.",
          );
        return {
          id: rubric.id,
          evaluator: policy.evaluator,
          implementationSha256: policy.implementationSha256,
          status: decision.status,
          reason: decision.reason.trim(),
          evidenceRefs: decision.evidence,
        };
      });
      if (!checks.length)
        throw new Error(
          "No pinned human review rubric is available for this result.",
        );
      const body: EvalAssessment = {
        schemaVersion: "contractor.eval-assessment-input/v1",
        source: { kind: "human" },
        resultSha256: context.resultSha256,
        checks,
        previousAssessmentSha256: member.assessmentSha256,
      };
      const operation = `review:${experiment.experimentId}:${member.member.memberId}`;
      // Keep the assessment key until selection succeeds so a retry replays it.
      const receipt = await recoverableMutation(
        owner,
        operation,
        body,
        (key) =>
          submitEvalAssessment(
            api,
            experiment.experimentId,
            member.member.memberId,
            body,
            key,
          ),
        { finish: false },
      );
      const selection = {
        planSha256: experiment.planSha256,
        selections: [
          {
            memberId: member.member.memberId,
            resultSha256: context.resultSha256,
            assessmentSha256: receipt.recordSha256,
          },
        ],
      };
      // Assessment creation is not CAS protected. Refresh immediately before
      // selecting so coordinator progress cannot strand the recorded result.
      const latest = await readCurrentReview();
      if (!latest.revision)
        throw new Error("The current review revision is unavailable.");
      const { revision } = latest;
      const correlation = { selection, revision };
      await recoverableMutation(
        owner,
        operation + ":select",
        correlation,
        (key) =>
          selectEvalAssessment(
            api,
            experiment.experimentId,
            selection,
            key,
            revision,
          ),
      );
      await finishMutation(owner, operation, body);
    },
    onSuccess: async () => {
      await cache.invalidateQueries({ queryKey: queryKeys.evals.all });
      onSaved();
    },
  });
  const changed =
    review.data && review.data.resultSha256 !== member.resultSha256;
  return (
    <section className="eval-panel eval-review" aria-labelledby={heading}>
      <div className="eval-step-heading">
        <h2 id={heading} ref={headingRef} tabIndex={-1}>
          Review result
        </h2>
        <p className="eval-muted">
          This decision is recorded under your account and cannot be changed
          later.
        </p>
      </div>
      <EvalError
        error={review.error ?? save.error}
        reload={
          review.error || save.error
            ? () => {
                void review.refetch().then(() => save.reset());
              }
            : undefined
        }
      />
      {changed ? (
        <p className="eval-callout" role="alert">
          A newer result is selected. Close this review and refresh the pair
          before judging it.
        </p>
      ) : null}
      {review.data && !changed ? (
        <>
          {review.data.gaps.map((gap) => (
            <p className="eval-muted" key={gap}>
              {gap}
            </p>
          ))}
          {review.data.checks.map((rubric) => (
            <fieldset
              className="eval-rubric-review"
              key={rubric.id}
              disabled={save.isPending}
            >
              <legend>
                <code>{rubric.id}</code> · rubric {rubric.revision}
              </legend>
              <p className="eval-rubric">{rubric.rubric}</p>
              {Object.keys(rubric.expected).length ? (
                <dl className="eval-facts">
                  {Object.entries(rubric.expected).map(([key, value]) => (
                    <Fragment key={key}>
                      <dt>{key}</dt>
                      <dd>{value}</dd>
                    </Fragment>
                  ))}
                </dl>
              ) : null}
              <div className="eval-grid">
                <EvalField label={`Decision for ${rubric.id}`}>
                  <select
                    value={decisions[rubric.id]?.status ?? ""}
                    onChange={(e) =>
                      update(rubric.id, {
                        status: e.target.value as Decision | "",
                      })
                    }
                  >
                    <option value="">Choose after review</option>
                    <option value="pass">Pass</option>
                    <option value="fail">Fail</option>
                    <option value="incomplete">Incomplete evidence</option>
                    {review.data?.policy?.find((c) => c.id === rubric.id)
                      ?.allowNotApplicable ? (
                      <option value="not_applicable">Not applicable</option>
                    ) : null}
                  </select>
                </EvalField>
              </div>
              <EvalField label={`Reason for ${rubric.id}`}>
                <textarea
                  rows={3}
                  value={decisions[rubric.id]?.reason ?? ""}
                  onChange={(e) =>
                    update(rubric.id, { reason: e.target.value })
                  }
                />
              </EvalField>
              <fieldset className="eval-subfieldset">
                <legend>Cited evidence</legend>
                {review.data?.evidence.length ? (
                  review.data.evidence.map((evidence) => (
                    <label className="eval-checkbox" key={evidence.id}>
                      <input
                        type="checkbox"
                        checked={
                          decisions[rubric.id]?.evidence.includes(
                            evidence.id,
                          ) ?? false
                        }
                        onChange={(e) =>
                          update(rubric.id, {
                            evidence: e.target.checked
                              ? [
                                  ...(decisions[rubric.id]?.evidence ?? []),
                                  evidence.id,
                                ]
                              : (decisions[rubric.id]?.evidence ?? []).filter(
                                  (id) => id !== evidence.id,
                                ),
                          })
                        }
                      />
                      <ContextLink
                        to={artifactHref(evidence.artifact)}
                        returnLabel="Human review"
                      >
                        {evidence.id}
                      </ContextLink>
                    </label>
                  ))
                ) : (
                  <p className="eval-muted">No evidence files to cite.</p>
                )}
              </fieldset>
            </fieldset>
          ))}
          <TechnicalDetails summary="Result digest">
            <IdChip value={review.data.resultSha256} label="result digest" />
          </TechnicalDetails>
        </>
      ) : null}
      <div className="eval-actions">
        <button type="button" className="ui-btn" onClick={onClose}>
          Close review
        </button>
        <button
          type="button"
          className="ui-btn"
          data-variant="primary"
          disabled={!review.data?.checks.length || !!changed || save.isPending}
          onClick={() => save.mutate()}
        >
          {save.isPending ? "Saving review…" : "Save and select assessment"}
        </button>
      </div>
    </section>
  );
}
