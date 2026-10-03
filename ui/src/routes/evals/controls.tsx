import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useId, useRef, useState } from "react";
import { useNavigate } from "react-router";
import { usePublicAPI } from "../../api/context";
import { PublicAPIError } from "../../api/error";
import {
  commandEvalExperiment,
  EVAL_POLL_MS,
  getEvalCommand,
  getEvalExperiment,
  type EvalCommand,
  type EvalCommandReceipt,
  type EvalExperiment,
  type EvalReceipt,
} from "../../api/evals";
import { Dialog } from "../../app/dialog";
import { createMutationIdempotencyKey } from "../../mutations/idempotency";
import { EvalError } from "./common";
import { EvalExecutionStatus } from "./execution-status";
import { useEvalOwner } from "./queries";
import {
  commandFinished,
  readCommand,
  RECOVERY_STORAGE_MESSAGE,
  writeCommand,
  type PendingCommand,
} from "./recovery";
import { queryKeys } from "../../api/query-keys";

const LABELS: Record<EvalCommand["kind"], string> = {
  prepare: "Prepare",
  start: "Start",
  pause: "Pause",
  resume: "Resume",
  cancel: "Cancel",
  duplicate: "Duplicate",
  finalize: "Finalize",
};

function needsCurrentRevision(kind: EvalCommand["kind"]): boolean {
  return kind === "pause" || kind === "resume" || kind === "cancel";
}

// The current experiment rules the stop intent out, so no replay or new
// revision can apply it. Callers drop the pending command instead of retrying.
class StopIntentUnavailableError extends Error {
  constructor(kind: EvalCommand["kind"]) {
    super(
      `The experiment changed and ${LABELS[kind]} is no longer available, so it was not applied. Choose an action for its current state.`,
    );
    this.name = "StopIntentUnavailableError";
  }
}

function stopIntentAvailable(
  latest: EvalExperiment,
  command: PendingCommand,
): boolean {
  return (
    latest.planSha256 === (command.body.planSha256 ?? null) &&
    latest.allowedCommands.includes(command.body.kind)
  );
}

export function EvalControls({
  experiment,
  disabled = false,
}: {
  experiment: EvalExperiment;
  disabled?: boolean;
}) {
  const api = usePublicAPI(),
    owner = useEvalOwner(),
    cache = useQueryClient(),
    navigate = useNavigate();
  const [pending, setPending] = useState<PendingCommand | null>(() =>
    readCommand(owner, experiment.experimentId),
  );
  const [confirm, setConfirm] = useState<{
    kind: EvalCommand["kind"];
    experiment: EvalExperiment;
  } | null>(null);
  const [storageError, setStorageError] = useState<Error | null>(null);
  const [preparing, setPreparing] = useState(false);
  const preparingRef = useRef(false);
  const heading = useId(),
    dismiss = useRef<HTMLButtonElement>(null);
  const [receipt, setReceipt] = useState<EvalCommandReceipt | null>(null);
  const submitted = useRef<string | null>(null);
  const native = experiment.controlMode === "server";
  async function settle(
    current: PendingCommand,
    result: EvalCommandReceipt | EvalReceipt,
  ) {
    if ("commandId" in result && !commandFinished(result)) {
      const next = { ...current, commandId: result.commandId };
      writeCommand(owner, experiment.experimentId, next);
      setPending(next);
    } else {
      writeCommand(owner, experiment.experimentId, null);
      setPending(null);
      if ("commandId" in result) setReceipt(result);
      else
        void navigate(
          `/evals/experiments/${encodeURIComponent(result.experimentId)}/setup`,
        );
    }
    await cache.invalidateQueries({
      queryKey: queryKeys.evals.experiment(experiment.experimentId),
    });
    await cache.invalidateQueries({ queryKey: queryKeys.evals.lists });
  }
  const send = useMutation({
    mutationFn: async (current: PendingCommand) => {
      const submit = (command: PendingCommand) =>
        commandEvalExperiment(
          api,
          experiment.experimentId,
          command.body,
          command.key,
          command.revision,
        );
      try {
        return { result: await submit(current), command: current };
      } catch (error) {
        if (
          !needsCurrentRevision(current.body.kind) ||
          !(error instanceof PublicAPIError) ||
          error.status !== 412 ||
          error.code !== "eval_revision_mismatch"
        )
          throw error;
        // A rejected CAS has not applied the command. Recheck intent and
        // persist a new correlation before one retry; a lost response still
        // replays the exact key, body and revision that were sent. An intent
        // the latest state rules out is dropped so it is never replayed.
        const latest = await getEvalExperiment(api, experiment.experimentId);
        cache.setQueryData(
          queryKeys.evals.experiment(experiment.experimentId),
          latest,
        );
        if (!stopIntentAvailable(latest, current)) {
          writeCommand(owner, experiment.experimentId, null);
          setPending(null);
          throw new StopIntentUnavailableError(current.body.kind);
        }
        const retry = {
          ...current,
          key: createMutationIdempotencyKey("eval"),
          revision: latest.revision,
        };
        writeCommand(owner, experiment.experimentId, retry);
        submitted.current = retry.key;
        setPending(retry);
        return { result: await submit(retry), command: retry };
      }
    },
    onSuccess: ({ result, command }) => settle(command, result),
  });
  const commandId = pending?.commandId;
  const command = useQuery({
    queryKey: queryKeys.evals.command(experiment.experimentId, commandId),
    enabled: native && !!pending && !!commandId,
    queryFn: async () => {
      const result = await getEvalCommand(
        api,
        experiment.experimentId,
        commandId!,
      );
      if (commandFinished(result)) await settle(pending!, result);
      return result;
    },
    refetchInterval: (query) =>
      query.state.error ||
      (query.state.data && commandFinished(query.state.data))
        ? false
        : EVAL_POLL_MS,
  });
  // A command without a receipt (including one recovered from storage) is
  // replayed with its original idempotency key. The ref keeps StrictMode's
  // double-invoked effect from sending it twice.
  const { mutate } = send;
  useEffect(() => {
    if (!native || !pending || pending.commandId) return;
    if (submitted.current === pending.key) return;
    submitted.current = pending.key;
    mutate(pending);
  }, [native, pending, mutate]);
  const error = send.error ?? command.error;
  const busy = !!pending || preparing;
  async function execute(kind: EvalCommand["kind"], reviewed = experiment) {
    if (preparingRef.current || pending) return;
    preparingRef.current = true;
    setPreparing(true);
    try {
      const currentRevision = needsCurrentRevision(kind)
        ? await getEvalExperiment(api, experiment.experimentId)
        : reviewed;
      if (needsCurrentRevision(kind))
        cache.setQueryData(
          queryKeys.evals.experiment(experiment.experimentId),
          currentRevision,
        );
      const next: PendingCommand = {
        key: createMutationIdempotencyKey("eval"),
        revision: currentRevision.revision,
        body: {
          kind,
          ...(kind !== "prepare" && kind !== "duplicate" && reviewed.planSha256
            ? { planSha256: reviewed.planSha256 }
            : {}),
        },
      };
      if (
        needsCurrentRevision(kind) &&
        !stopIntentAvailable(currentRevision, next)
      )
        throw new StopIntentUnavailableError(kind);
      try {
        writeCommand(owner, experiment.experimentId, next);
      } catch {
        throw new Error(RECOVERY_STORAGE_MESSAGE);
      }
      setPending(next);
      setReceipt(null);
      setConfirm(null);
      setStorageError(null);
    } catch (error) {
      setStorageError(
        error instanceof Error ? error : new Error(RECOVERY_STORAGE_MESSAGE),
      );
    } finally {
      preparingRef.current = false;
      setPreparing(false);
    }
  }
  async function recover() {
    // The ruled-out intent was already dropped; there is nothing to replay.
    if (send.error instanceof StopIntentUnavailableError) send.reset();
    else if (
      error instanceof PublicAPIError &&
      error.status >= 400 &&
      error.status < 500
    ) {
      writeCommand(owner, experiment.experimentId, null);
      setPending(null);
      send.reset();
      await cache.invalidateQueries({
        queryKey: queryKeys.evals.experiment(experiment.experimentId),
      });
    } else if (send.error && pending) mutate(pending);
    else await command.refetch();
  }
  return (
    <section className="eval-controls">
      <EvalExecutionStatus experiment={experiment} />
      {native ? (
        <div className="eval-actions">
          {experiment.allowedCommands
            .filter((kind) => kind !== "finalize")
            .map((kind) => (
              <button
                key={kind}
                type="button"
                className={
                  !disabled &&
                  (kind === "prepare" || kind === "start" || kind === "resume")
                    ? undefined
                    : "secondary-button"
                }
                disabled={busy || disabled}
                onClick={() => {
                  if (kind === "start" || kind === "cancel")
                    setConfirm({ kind, experiment });
                  else void execute(kind);
                }}
              >
                {LABELS[kind]}
              </button>
            ))}
        </div>
      ) : (
        <p>
          Externally controlled ·{" "}
          {experiment.setup?.source?.system ?? "Independent producer"}. Last
          producer update: {experiment.lastProducerActivityAt ?? "not observed"}
          . The producer owns dispatch; these pages provide inspection and
          review.
        </p>
      )}
      {pending && !error ? (
        <p role="status">
          {pending?.body.kind}: {command.data?.state ?? "recovering receipt"}.
          Waiting for the server to confirm.
        </p>
      ) : null}
      {receipt ? (
        <p role="status">
          {LABELS[receipt.kind]}: {receipt.state}.
        </p>
      ) : null}
      <EvalError
        error={storageError ?? error}
        reload={
          (storageError ?? error) instanceof StopIntentUnavailableError
            ? undefined
            : storageError && storageError.message !== RECOVERY_STORAGE_MESSAGE
              ? () => {
                  setStorageError(null);
                  void cache.invalidateQueries({
                    queryKey: queryKeys.evals.experiment(
                      experiment.experimentId,
                    ),
                  });
                }
              : error
                ? () => void recover()
                : undefined
        }
      />
      {confirm ? (
        <Dialog
          className="project-dialog panel"
          labelledBy={heading}
          initialFocusRef={dismiss}
          onRequestClose={() => {
            if (!preparingRef.current) setConfirm(null);
          }}
        >
          <h2 id={heading}>{LABELS[confirm.kind]} experiment?</h2>
          {confirm.kind === "start" ? (
            <>
              <p>
                {confirm.experiment.expectedMembers} expected members. Start
                uses the prepared A/B variants and budgets shown in Setup.
              </p>
              <p>Closing this browser will not stop execution.</p>
            </>
          ) : (
            <p>
              Stop new dispatch and request cancellation of accepted executions.
              The experiment stays Cancelling until the server confirms they
              have drained.
            </p>
          )}
          <div className="eval-actions">
            <button
              ref={dismiss}
              type="button"
              className="secondary-button"
              disabled={preparing}
              onClick={() => setConfirm(null)}
            >
              Keep current state
            </button>
            <button
              type="button"
              disabled={preparing}
              onClick={() => void execute(confirm.kind, confirm.experiment)}
            >
              Confirm {LABELS[confirm.kind].toLowerCase()}
            </button>
          </div>
        </Dialog>
      ) : null}
    </section>
  );
}
