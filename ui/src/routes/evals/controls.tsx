import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useId, useRef, useState } from "react";
import { useNavigate } from "react-router";
import { usePublicAPI } from "../../api/context";
import { PublicAPIError } from "../../api/error";
import {
  commandEvalExperiment,
  EVAL_POLL_MS,
  getEvalCommand,
  type EvalCommand,
  type EvalExperiment,
} from "../../api/evals";
import { Dialog } from "../../app/dialog";
import { createMutationIdempotencyKey } from "../../mutations/idempotency";
import { EvalError } from "./common";
import { EvalExecutionStatus } from "./execution-status";
import { useEvalOwner } from "./queries";
import {
  commandFinished,
  readCommand,
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
  const command = useQuery({
    queryKey: ["evals", "command", experiment.experimentId, pending?.key],
    enabled: pending !== null && experiment.controlMode === "server",
    queryFn: async () => {
      const current = readCommand(owner, experiment.experimentId) ?? pending!;
      const result = current.commandId
        ? await getEvalCommand(api, experiment.experimentId, current.commandId)
        : await commandEvalExperiment(
            api,
            experiment.experimentId,
            current.body,
            current.key,
            current.revision,
          );
      if ("commandId" in result) {
        writeCommand(
          owner,
          experiment.experimentId,
          commandFinished(result)
            ? null
            : { ...current, commandId: result.commandId },
        );
      } else {
        writeCommand(owner, experiment.experimentId, null);
        void navigate(
          `/evals/experiments/${encodeURIComponent(result.experimentId)}/setup`,
        );
      }
      await cache.invalidateQueries({
        queryKey: ["evals", "experiment", experiment.experimentId],
      });
      await cache.invalidateQueries({ queryKey: ["evals", "list"] });
      return result;
    },
    refetchInterval: (query) =>
      query.state.error ||
      (query.state.data &&
        (!("commandId" in query.state.data) ||
          commandFinished(query.state.data)))
        ? false
        : EVAL_POLL_MS,
  });
  const finished =
    command.data &&
    (!("commandId" in command.data) || commandFinished(command.data));
  const busy = !!pending && !finished;
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
      setConfirm(null);
      setStorageError(null);
    } catch {
      setStorageError(
        new Error(
          "Command recovery could not be saved in this browser. No command was sent.",
        ),
      );
    }
  }
  async function recover() {
    if (
      command.error instanceof PublicAPIError &&
      command.error.status >= 400 &&
      command.error.status < 500
    ) {
      writeCommand(owner, experiment.experimentId, null);
      setPending(null);
      await cache.invalidateQueries({
        queryKey: ["evals", "experiment", experiment.experimentId],
      });
    } else await command.refetch();
  }
  const native = experiment.controlMode === "server";
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
      {busy && !command.error ? (
        <p role="status">
          {pending?.body.kind}: {command.data?.state ?? "recovering receipt"}.
          Waiting for the server to confirm.
        </p>
      ) : null}
      {finished && command.data && "commandId" in command.data ? (
        <p role="status">
          {LABELS[command.data.kind]}: {command.data.state}.
        </p>
      ) : null}
      <EvalError
        error={storageError ?? command.error}
        reload={command.error ? () => void recover() : undefined}
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
                uses the prepared A/B variants and frozen budgets shown in
                Setup.
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
