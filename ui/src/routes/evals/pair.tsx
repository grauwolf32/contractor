import { artifactHref } from "./links";
import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { useLocation, useParams, useSearchParams } from "react-router";
import { usePublicAPI } from "../../api/context";
import { getEvalPair, type EvalMember } from "../../api/evals";
import { ContextLink, ReturnLink } from "../../app/context-navigation";
import { EvalError, EvalFrame } from "./common";
import { MemberExecutions, MemberSummary } from "./member";
import { useEvalExperiment } from "./queries";
import { EvalHumanReview } from "./review";

export function EvalPairRoute() {
  const { experimentId = "", pairId = "" } = useParams();
  const [params, setParams] = useSearchParams();
  const location = useLocation();
  const snapshot = params.get("viewSnapshot") ?? undefined;
  const api = usePublicAPI(),
    experiment = useEvalExperiment(experimentId);
  const pair = useQuery({
    queryKey: ["evals", "pair", experimentId, pairId, snapshot],
    queryFn: () => getEvalPair(api, experimentId, pairId, snapshot),
  });
  const [review, setReview] = useState<EvalMember | null>(null);
  function refresh() {
    setReview(null);
    setParams({}, { replace: true, state: location.state });
    void experiment.refetch();
    void pair.refetch();
  }
  const canReview =
    experiment.data?.setup?.checks.some(
      (c) => c.evaluator === "human-review@1",
    ) ?? false;
  return (
    <EvalFrame
      title={
        pair.data
          ? `${pair.data.pair.caseId} / sample ${pair.data.pair.sample}`
          : "Paired evidence"
      }
      action={
        <ReturnLink
          to={`/evals/experiments/${encodeURIComponent(experimentId)}/comparison`}
          label="Comparison"
        />
      }
    >
      <EvalError error={pair.error ?? experiment.error} reload={refresh} />
      {pair.data ? (
        <>
          <p>
            {pair.data.freshness} snapshot
            {pair.data.pair.regression ? " · Known quality regression" : ""}
          </p>
          <button type="button" className="secondary-button" onClick={refresh}>
            Refresh pair evidence
          </button>
          <div className="eval-variants">
            {(
              [
                ["A", pair.data.pair.a],
                ["B", pair.data.pair.b],
              ] as const
            ).map(([label, member]) => (
              <section
                className={`panel eval-panel eval-arm-${label.toLowerCase()}`}
                key={label}
              >
                <h2>
                  {label} · {member.member.variantId}
                </h2>
                <MemberSummary member={member} />
                <MemberExecutions id={experimentId} member={member} />
                {canReview && member.resultSha256 ? (
                  <button
                    type="button"
                    className="secondary-button"
                    onClick={() => setReview(member)}
                  >
                    Review {label}
                  </button>
                ) : null}
              </section>
            ))}
          </div>
          {pair.data.pair.exclusions.map((reason) => (
            <p key={reason}>{reason.replaceAll("_", " ")}</p>
          ))}
          <h2>Attributed records and evidence</h2>
          {pair.data.records.map((record) => (
            <details key={`${record.memberId}:${record.recordSha256}`}>
              <summary>
                {record.kind} · {record.actorId} ·{" "}
                {new Date(record.createdAt).toLocaleString()}
              </summary>
              <p className="eval-digest">
                Record: <code>{record.recordSha256}</code>
              </p>
              <p>
                Predecessor: {record.previousRecordSha256 ?? "First revision"}
              </p>
              {"outputs" in record.document ? (
                <>
                  <p>
                    Producer: {record.document.source.system}/
                    {record.document.source.id}. Collection:{" "}
                    {record.document.collection.status}
                  </p>
                  {record.document.collection.gaps.map((gap) => (
                    <p key={gap}>{gap}</p>
                  ))}
                  <ul>
                    {Object.entries(record.document.outputs).map(
                      ([role, artifact]) => (
                        <li key={role}>
                          <ContextLink
                            returnLabel="Paired evidence"
                            to={artifactHref(artifact)}
                          >
                            {role} · {artifact.name}
                          </ContextLink>
                        </li>
                      ),
                    )}
                  </ul>
                </>
              ) : (
                <>
                  <p>
                    Assessment source: {record.document.source.kind}
                    {record.document.source.producerId
                      ? ` · ${record.document.source.producerId}`
                      : ""}
                    . Attribution is not independent verification.
                  </p>
                  <ul>
                    {record.document.checks.map((check) => (
                      <li key={check.id}>
                        {check.id}: {check.status} · {check.reason}
                      </li>
                    ))}
                  </ul>
                </>
              )}
            </details>
          ))}
          {review && experiment.data ? (
            <EvalHumanReview
              key={`${review.member.memberId}:${review.resultSha256}`}
              experiment={experiment.data}
              member={review}
              onClose={() => setReview(null)}
              onSaved={refresh}
            />
          ) : null}
        </>
      ) : null}
    </EvalFrame>
  );
}
