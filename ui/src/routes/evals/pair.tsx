import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useId, useRef, useState } from "react";
import { useLocation, useParams, useSearchParams } from "react-router";
import { usePublicAPI } from "../../api/context";
import { getEvalPair, type EvalMember } from "../../api/evals";
import { queryKeys } from "../../api/query-keys";
import { ContextLink, ReturnLink } from "../../app/context-navigation";
import { useDocumentTitle } from "../../app/document-title";
import { RefreshButton } from "../../app/refresh-button";
import { IdChip, StatusChip, TechnicalDetails } from "../../ui";
import { ArmKey, EvalError, EvalFrame } from "./common";
import { experimentListPath, experimentPath } from "./labels";
import { artifactHref } from "./links";
import { MemberExecutions, MemberSummary } from "./member";
import { useEvalExperiment } from "./queries";
import { EvalHumanReview } from "./review";

type Arm = "A" | "B";

export function EvalPairRoute() {
  const { experimentId = "", pairId = "" } = useParams();
  const [params, setParams] = useSearchParams();
  const location = useLocation();
  const recordsHeading = useId();
  const snapshot = params.get("viewSnapshot") ?? undefined;
  const api = usePublicAPI(),
    experiment = useEvalExperiment(experimentId);
  const cache = useQueryClient();
  const pair = useQuery({
    queryKey: queryKeys.evals.pair(experimentId, pairId, snapshot),
    queryFn: () => getEvalPair(api, experimentId, pairId, snapshot),
  });
  const [review, setReview] = useState<{
    member: EvalMember;
    arm: Arm;
  } | null>(null);
  // Closing the review returns focus to the button that opened it.
  const reviewTrigger = useRef<HTMLButtonElement | null>(null);
  // Saving closes the review and reloads the pair. The page announces the
  // result and, once the reloaded pair is back, focuses the reviewed arm.
  const [recorded, setRecorded] = useState("");
  const focusAfterSave = useRef<Arm | null>(null);
  const armHeadings = useRef<Partial<Record<Arm, HTMLHeadingElement>>>({});
  useEffect(() => {
    const arm = focusAfterSave.current;
    if (
      arm === null ||
      review !== null ||
      snapshot !== undefined ||
      pair.isFetching
    )
      return;
    focusAfterSave.current = null;
    // Focus lost with the closed review moves on; focus the user moved
    // elsewhere meanwhile stays there.
    const active = document.activeElement;
    if (active === null || active === document.body || !active.isConnected)
      armHeadings.current[arm]?.focus();
  });
  function refresh() {
    setReview(null);
    setParams({}, { replace: true, state: location.state });
    void experiment.refetch();
    void pair.refetch();
    void cache.invalidateQueries({
      queryKey: queryKeys.evals.inventories(experimentId),
    });
  }
  const canReview =
    experiment.data?.setup?.checks.some(
      (c) => c.evaluator === "human-review@1",
    ) ?? false;
  const title = pair.data
    ? `${pair.data.pair.caseId} / sample ${pair.data.pair.sample}`
    : "Paired evidence";
  useDocumentTitle(title);
  return (
    <EvalFrame
      title={title}
      back={
        <ReturnLink
          to={experimentPath(experimentId, "comparison")}
          label="Comparison"
        />
      }
      breadcrumb={[
        { label: "Experiments", to: experimentListPath(experimentId) },
        {
          label: experiment.data?.name ?? "Experiment",
          to: experimentPath(experimentId, "overview"),
        },
        { label: "Paired evidence" },
      ]}
      status={
        pair.data?.pair.regression ? (
          <StatusChip tone="blocked">Known quality regression</StatusChip>
        ) : undefined
      }
      description={
        pair.data ? (
          <span>
            {pair.data.freshness === "stale"
              ? "Stale snapshot · refresh to review the current evidence"
              : "Current snapshot"}
          </span>
        ) : undefined
      }
      action={
        pair.data ? (
          <RefreshButton
            className="ui-btn"
            isFetching={pair.isFetching || experiment.isFetching}
            onRefresh={refresh}
            label="Refresh"
          />
        ) : undefined
      }
    >
      {/* Always present, so the live region is in place before it speaks. */}
      <p className="eval-announcement" role="status">
        {recorded}
      </p>
      <EvalError error={pair.error ?? experiment.error} reload={refresh} />
      {pair.data ? (
        <>
          <div className="eval-arms">
            {(
              [
                ["A", pair.data.pair.a],
                ["B", pair.data.pair.b],
              ] as const
            ).map(([label, member]) => (
              <section
                className={`eval-panel eval-arm eval-arm-${label.toLowerCase()}`}
                data-arm={label.toLowerCase()}
                key={label}
              >
                <h2
                  ref={(node) => {
                    if (node) armHeadings.current[label] = node;
                    else delete armHeadings.current[label];
                  }}
                  tabIndex={-1}
                >
                  <ArmKey arm={label === "A" ? "a" : "b"}>
                    {label} · {member.member.variantId}
                  </ArmKey>
                </h2>
                <MemberSummary member={member} />
                <MemberExecutions id={experimentId} member={member} />
                {canReview && member.resultSha256 ? (
                  <div className="eval-actions">
                    <button
                      type="button"
                      className="ui-btn"
                      data-size="sm"
                      onClick={(event) => {
                        reviewTrigger.current = event.currentTarget;
                        setRecorded("");
                        setReview({ member, arm: label });
                      }}
                    >
                      Review {label}
                    </button>
                  </div>
                ) : null}
              </section>
            ))}
          </div>
          {pair.data.pair.exclusions.map((reason) => (
            <p className="eval-muted" key={reason}>
              {reason.replaceAll("_", " ")}
            </p>
          ))}
          {review && experiment.data ? (
            <EvalHumanReview
              key={`${review.member.member.memberId}:${review.member.resultSha256}`}
              experiment={experiment.data}
              member={review.member}
              onClose={() => {
                setReview(null);
                reviewTrigger.current?.focus();
              }}
              onSaved={() => {
                setRecorded(
                  `Assessment recorded and selected for ${review.arm}.`,
                );
                focusAfterSave.current = review.arm;
                refresh();
              }}
            />
          ) : null}
          <section className="eval-section" aria-labelledby={recordsHeading}>
            <h2 id={recordsHeading}>Attributed records and evidence</h2>
            {pair.data.records.length ? (
              <ul className="eval-records" role="list">
                {pair.data.records.map((record) => (
                  <li
                    className="eval-record-disclosure"
                    key={`${record.memberId}:${record.recordSha256}`}
                  >
                    <details className="eval-disclosure">
                      <summary>
                        {record.kind === "result" ? "Result" : "Assessment"} ·{" "}
                        {record.actorId} ·{" "}
                        {new Date(record.createdAt).toLocaleString()}
                      </summary>
                      <div className="eval-disclosure-body">
                        {"outputs" in record.document ? (
                          <>
                            <p>
                              Producer: {record.document.source.system}/
                              {record.document.source.id}. Collection:{" "}
                              {record.document.collection.status}
                            </p>
                            {record.document.collection.gaps.map((gap) => (
                              <p className="eval-muted" key={gap}>
                                {gap}
                              </p>
                            ))}
                            {Object.keys(record.document.outputs).length ? (
                              <ul className="eval-criteria" role="list">
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
                            ) : null}
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
                            <ul className="eval-criteria" role="list">
                              {record.document.checks.map((check) => (
                                <li key={check.id}>
                                  <code>{check.id}</code>: {check.status} ·{" "}
                                  {check.reason}
                                </li>
                              ))}
                            </ul>
                          </>
                        )}
                        <TechnicalDetails summary="Record digests">
                          <dl className="eval-facts">
                            <dt>Record</dt>
                            <dd>
                              <IdChip
                                value={record.recordSha256}
                                label="record digest"
                              />
                            </dd>
                            <dt>Predecessor</dt>
                            <dd>
                              {record.previousRecordSha256 ? (
                                <IdChip
                                  value={record.previousRecordSha256}
                                  label="predecessor digest"
                                />
                              ) : (
                                "First revision"
                              )}
                            </dd>
                          </dl>
                        </TechnicalDetails>
                      </div>
                    </details>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="eval-muted">No attributed records yet.</p>
            )}
          </section>
        </>
      ) : null}
    </EvalFrame>
  );
}
