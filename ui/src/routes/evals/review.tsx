import { artifactHref } from "./links";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
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
import { EvalError, EvalField } from "./common";
import { useEvalOwner } from "./queries";
import { finishMutation, recoverableMutation } from "./recovery";

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
  const review = useQuery({
    queryKey: [
      "evals",
      "review",
      experiment.experimentId,
      member.member.memberId,
      member.resultSha256,
    ],
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
  const save = useMutation({
    mutationFn: async () => {
      const context = review.data;
      if (
        !context ||
        context.resultSha256 !== member.resultSha256 ||
        !context.revision ||
        !experiment.planSha256
      )
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
      const { revision } = context;
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
      await cache.invalidateQueries({ queryKey: ["evals"] });
      onSaved();
    },
  });
  const changed =
    review.data && review.data.resultSha256 !== member.resultSha256;
  return (
    <section className="panel eval-panel">
      <h3>Review result</h3>
      <p>
        This decision is recorded under your account and cannot be changed
        later.
      </p>
      <EvalError error={review.error ?? save.error} />
      {changed ? (
        <p role="alert">
          A newer result is selected. Close this review and refresh the pair
          before judging it.
        </p>
      ) : null}
      {review.data && !changed ? (
        <>
          <p className="eval-digest">
            Result: <code>{review.data.resultSha256}</code>
          </p>
          {review.data.gaps.map((gap) => (
            <p key={gap}>{gap}</p>
          ))}
          {review.data.checks.map((rubric) => (
            <fieldset key={rubric.id} disabled={save.isPending}>
              <legend>
                {rubric.id} · rubric {rubric.revision}
              </legend>
              <p className="eval-rubric">{rubric.rubric}</p>
              <dl>
                {Object.entries(rubric.expected).map(([key, value]) => (
                  <div key={key}>
                    <dt>{key}</dt>
                    <dd>{value}</dd>
                  </div>
                ))}
              </dl>
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
              <EvalField label={`Reason for ${rubric.id}`}>
                <textarea
                  rows={3}
                  value={decisions[rubric.id]?.reason ?? ""}
                  onChange={(e) =>
                    update(rubric.id, { reason: e.target.value })
                  }
                />
              </EvalField>
              <fieldset>
                <legend>Cited evidence</legend>
                {review.data?.evidence.map((evidence) => (
                  <label key={evidence.id}>
                    <input
                      type="checkbox"
                      checked={
                        decisions[rubric.id]?.evidence.includes(evidence.id) ??
                        false
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
                ))}
              </fieldset>
            </fieldset>
          ))}
        </>
      ) : null}
      <div className="eval-actions">
        <button type="button" className="secondary-button" onClick={onClose}>
          Close review
        </button>
        <button
          type="button"
          disabled={!review.data?.checks.length || !!changed || save.isPending}
          onClick={() => save.mutate()}
        >
          {save.isPending ? "Saving review…" : "Save and select assessment"}
        </button>
      </div>
    </section>
  );
}
