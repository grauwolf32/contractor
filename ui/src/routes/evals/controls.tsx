import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useId, useRef, useState } from "react";
import { useNavigate } from "react-router";
import { usePublicAPI } from "../../api/context";
import { PublicAPIError } from "../../api/error";
import {
  commandEvalExperiment,
  EVAL_POLL_MS,
  getEvalCommand,
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

const LABELS: Record<EvalCommand["kind"], string> = {
  prepare: "Prepare",
  start: "Start",
  pause: "Pause",
  resume: "Resume",
  cancel: "Cancel",
  duplicate: "Duplicate",
  finalize: "Finalize",
};

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
      queryKey: ["evals", "experiment", experiment.experimentId],
    });
    await cache.invalidateQueries({ queryKey: ["evals", "list"] });
  }
  const send = useMutation({
    mutationFn: (current: PendingCommand) =>
      commandEvalExperiment(
        api,
        experiment.experimentId,
        current.body,
        current.key,
        current.revision,
      ),
    onSuccess: (result, current) => settle(current, result),
  });
  const commandId = pending?.commandId;
  const command = useQuery({
    queryKey: ["evals", "command", experiment.experimentId, commandId],
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
  const busy = !!pending;
  function execute(kind: EvalCommand["kind"], reviewed = experiment) {
    try {
      const next: PendingCommand = {
        key: createMutationIdempotencyKey("eval"),
        revision: reviewed.revision,
        body: {
          kind,
          ...(kind !== "prepare" && kind !== "duplicate" && reviewed.planSha256
            ? { planSha256: reviewed.planSha256 }
            : {}),
        },
      };
      writeCommand(owner, experiment.experimentId, next);
      setPending(next);
      setReceipt(null);
      setConfirm(null);
      setStorageError(null);
    } catch {
      setStorageError(new Error(RECOVERY_STORAGE_MESSAGE));
    }
  }
  async function recover() {
    if (
      error instanceof PublicAPIError &&
      error.status >= 400 &&
      error.status < 500
    ) {
      writeCommand(owner, experiment.experimentId, null);
      setPending(null);
      send.reset();
      await cache.invalidateQueries({
        queryKey: ["evals", "experiment", experiment.experimentId],
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
                onClick={() =>
                  kind === "start" || kind === "cancel"
                    ? setConfirm({ kind, experiment })
                    : execute(kind)
                }
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
      {busy && !error ? (
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
        reload={error ? () => void recover() : undefined}
      />
      {confirm ? (
        <Dialog
          className="project-dialog panel"
          labelledBy={heading}
          initialFocusRef={dismiss}
          onRequestClose={() => setConfirm(null)}
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
              onClick={() => setConfirm(null)}
            >
              Keep current state
            </button>
            <button
              type="button"
              onClick={() => execute(confirm.kind, confirm.experiment)}
            >
              Confirm {LABELS[confirm.kind].toLowerCase()}
            </button>
          </div>
        </Dialog>
      ) : null}
    </section>
  );
}
