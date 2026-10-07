import { useMutation, useQueryClient } from "@tanstack/react-query";
import { type ReactNode, useRef, useState } from "react";
import { Link, useNavigate } from "react-router";

import { usePublicAPI } from "../../api/context";
import { queryKeys } from "../../api/query-keys";
import {
  cancelRun,
  getRunRepeatDraft,
  isTerminalRunState,
  resumeRun,
  retryRunGateway,
  type RunStatus,
} from "../../api/runs";
import { ErrorNotice } from "../../app/error-notice";
import { Icon } from "../../app/icon";
import { useRunDraftStore } from "../../run-drafts/context";
import { auditDestination, prepareRepeatDraft } from "../../run-drafts/repeat";
import { CANCELLATION_REASON_LIMIT, CancelRunDialog } from "./cancel";
import { RetryModelDialog } from "./recovery";
import { ContinueRunDialog } from "./resume";

/** What one header action contributes: its button, a dialog and a notice. */
interface ActionParts {
  control: ReactNode;
  overlay?: ReactNode;
  notice?: ReactNode;
}

/**
 * Called after an action the Server accepted, once the Run was refetched:
 * the page announces the outcome and moves focus to its heading, because the
 * action's own button usually disappears with the new state.
 */
type ActionDone = (announcement: string) => void;

/**
 * Moves focus to the page heading. Used when the user closes a confirmation
 * after the Run stopped offering its action, so its button is gone too.
 */
type FocusHeading = () => void;

type Recovery = NonNullable<RunStatus["recovery"]>;

/** The failed stage a Continue confirmation was opened for. */
interface ContinueTarget {
  /** The stage execution the Server offered as the continuation source. */
  source: string;
  /** Its stage name, as the confirmation shows it. */
  stage: string;
}

/**
 * Retry model connection. The confirmation keeps the recovery it was opened
 * for: it stays open, with any refusal, until the user closes it, even when
 * a refetch ends that recovery, and a later recovery never reopens it
 * without a click.
 */
function useRetryModelConnection(
  run: RunStatus,
  onDone: ActionDone,
  focusHeading: FocusHeading,
): ActionParts {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const [target, setTarget] = useState<Recovery>();
  const retry = useMutation({
    mutationFn: () => retryRunGateway(api, run.runId),
    onSettled: () =>
      queryClient.invalidateQueries({ queryKey: queryKeys.runs.all }),
  });
  const offer = run.recovery?.requiresRetry === true ? run.recovery : undefined;

  function clear(): void {
    setTarget(undefined);
    retry.reset();
  }

  function dismiss(): void {
    clear();
    if (offer === undefined) focusHeading();
  }

  return {
    control:
      offer === undefined ? null : (
        <button
          className="ui-btn"
          data-size="sm"
          data-variant="primary"
          type="button"
          onClick={() => {
            retry.reset();
            setTarget(offer);
          }}
        >
          <Icon name="refresh" />
          Retry model connection
        </button>
      ),
    overlay:
      target === undefined ? null : (
        <RetryModelDialog
          recovery={target}
          pending={retry.isPending}
          error={retry.error}
          onConfirm={() =>
            retry.mutate(undefined, {
              onSuccess: () => {
                clear();
                onDone("Model connection retry requested.");
              },
            })
          }
          onCancel={dismiss}
        />
      ),
  };
}

/**
 * Continue from failed stage. Like the retry, the confirmation keeps the
 * stage it was opened for until the user closes it, and it sends that
 * stage's source.
 */
function useContinueFromFailedStage(
  run: RunStatus,
  onDone: ActionDone,
  focusHeading: FocusHeading,
): ActionParts {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const [target, setTarget] = useState<ContinueTarget>();
  const inFlight = useRef(false);
  const source =
    run.state === "failed" ? run.resumeStageExecutionId : undefined;
  const mutation = useMutation({
    mutationFn: (sourceID: string) => resumeRun(api, run.runId, sourceID),
    onSettled: async () => {
      try {
        await Promise.all([
          queryClient.invalidateQueries({ queryKey: queryKeys.runs.all }),
          queryClient.invalidateQueries({ queryKey: queryKeys.queue.all }),
          ...(run.projectId === undefined
            ? []
            : [
                queryClient.invalidateQueries({
                  queryKey: queryKeys.projects.detail(run.projectId),
                }),
              ]),
        ]);
      } finally {
        inFlight.current = false;
      }
    },
  });

  function clear(): void {
    setTarget(undefined);
    mutation.reset();
  }

  function dismiss(): void {
    clear();
    if (source === undefined) focusHeading();
  }

  // Sends the source the confirmation names, never a newer one: the Server
  // refuses a source that is no longer current.
  function confirm(captured: ContinueTarget): void {
    if (inFlight.current) return;
    inFlight.current = true;
    mutation.mutate(captured.source, {
      onSuccess: () => {
        clear();
        onDone(`Continuation of stage ${captured.stage} requested.`);
      },
    });
  }

  return {
    control:
      source === undefined ? null : (
        <button
          className="ui-btn"
          data-size="sm"
          data-variant="primary"
          type="button"
          onClick={() => {
            mutation.reset();
            setTarget({
              source,
              stage:
                run.attempts.find(
                  (attempt) => attempt.stageExecutionId === source,
                )?.stage ?? source,
            });
          }}
        >
          <Icon name="play" />
          Continue from failed stage
        </button>
      ),
    overlay:
      target === undefined ? null : (
        <ContinueRunDialog
          stage={target.stage}
          pending={mutation.isPending}
          error={mutation.error}
          onConfirm={() => confirm(target)}
          onCancel={dismiss}
        />
      ),
  };
}

type RepeatOutcome =
  | {
      kind: "conflict" | "capacity" | "blocked" | "audit-unavailable";
      message: string;
      destination?: string;
    }
  | undefined;

const REPEAT_OUTCOME_TITLES: Record<
  NonNullable<RepeatOutcome>["kind"],
  string
> = {
  conflict: "Existing draft preserved",
  capacity: "Draft limit reached",
  "audit-unavailable": "Continue from Audit",
  blocked: "Repeat draft unavailable",
};

function useConfigureAnotherRun(run: RunStatus): ActionParts {
  const api = usePublicAPI();
  const navigate = useNavigate();
  const drafts = useRunDraftStore();
  const [outcome, setOutcome] = useState<RepeatOutcome>();
  const mutation = useMutation({
    mutationFn: () => getRunRepeatDraft(api, run.runId),
  });

  async function configureAnotherRun(): Promise<void> {
    mutation.reset();
    setOutcome(undefined);
    try {
      const response = await mutation.mutateAsync();
      if (response.authority === "audit-managed") {
        const destination = auditDestination(response);
        if (destination === undefined) {
          setOutcome({
            kind: "audit-unavailable",
            message:
              response.notices[0]?.message ??
              "This Run is Audit-managed, but its owning Audit route is unavailable.",
          });
          return;
        }
        await navigate(destination);
        return;
      }
      const routeBlock = response.notices.find(
        (notice) =>
          notice.code === "workflow_unavailable" ||
          notice.code === "project_unavailable" ||
          notice.code === "project_deleting",
      );
      if (routeBlock !== undefined) {
        setOutcome({ kind: "blocked", message: routeBlock.message });
        return;
      }
      const prepared = prepareRepeatDraft(response);
      if (prepared === undefined) {
        setOutcome({
          kind: "blocked",
          message:
            response.notices.find((notice) => notice.severity === "blocking")
              ?.message ??
            "The Server did not provide an ordinary repeat draft.",
        });
        return;
      }
      const seeded = drafts.seed(prepared.identity, prepared.state);
      if (seeded.kind === "conflict") {
        setOutcome({
          kind: "conflict",
          message:
            "An edited draft already exists for this Workflow and scope. It was not overwritten.",
          destination: prepared.destination,
        });
        return;
      }
      if (seeded.kind === "capacity") {
        setOutcome({
          kind: "capacity",
          message: `The in-memory draft limit is reached (${seeded.drafts.length} retained). Open a draft and discard it before importing this request.`,
        });
        return;
      }
      await navigate(prepared.destination);
    } catch {
      // The mutation owns and renders the normalized API failure.
    }
  }

  if (!isTerminalRunState(run.state)) return { control: null };
  return {
    control: (
      <button
        className="ui-btn"
        data-size="sm"
        type="button"
        disabled={mutation.isPending}
        onClick={() => void configureAnotherRun()}
      >
        <Icon name="plus" />
        {mutation.isPending
          ? "Loading retained request…"
          : "Configure another Run"}
      </button>
    ),
    notice:
      mutation.error !== null ? (
        <ErrorNotice
          error={mutation.error}
          context="The retained request could not be loaded"
        />
      ) : outcome === undefined ? null : (
        <div className="notice notice-warning runs-repeat-notice" role="alert">
          <strong>{REPEAT_OUTCOME_TITLES[outcome.kind]}</strong>
          <p>{outcome.message}</p>
          {outcome.destination === undefined ? null : (
            <Link to={outcome.destination}>Open the existing draft →</Link>
          )}
        </div>
      ),
  };
}

function useCancelRun(
  run: RunStatus,
  onDone: ActionDone,
  focusHeading: FocusHeading,
): ActionParts {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const [open, setOpen] = useState(false);
  const [reason, setReason] = useState("");
  const [validationError, setValidationError] = useState<string | undefined>();
  const mutation = useMutation({
    mutationFn: (value: string) => cancelRun(api, run.runId, value),
    onSettled: async () => {
      // Both acceptance and a terminal-completion race are reconciled from
      // the authoritative aggregate. The mutation response is never projected.
      await queryClient.invalidateQueries({ queryKey: queryKeys.runs.all });
    },
  });
  const cancellable =
    !isTerminalRunState(run.state) && run.state !== "cancelling";

  function close(): void {
    setOpen(false);
    setValidationError(undefined);
    mutation.reset();
    if (!cancellable) focusHeading();
  }

  async function submit(): Promise<void> {
    mutation.reset();
    const normalized = reason.trim();
    if (
      normalized.length === 0 ||
      normalized.length > CANCELLATION_REASON_LIMIT
    ) {
      setValidationError("Give a cancellation reason of 1–4096 characters.");
      return;
    }
    setValidationError(undefined);
    try {
      await mutation.mutateAsync(normalized);
    } catch {
      // The dialog keeps the reason and shows the refusal for a new attempt.
      return;
    }
    setOpen(false);
    setReason("");
    const current = queryClient.getQueryData<RunStatus>(
      queryKeys.runs.detail(run.runId),
    );
    onDone(
      current === undefined ||
        current.state === "cancelling" ||
        current.state === "cancelled"
        ? "Cancellation requested. The Run stops after cleanup."
        : "The Run had finished before the cancellation request.",
    );
  }

  return {
    control: cancellable ? (
      <button
        className="ui-btn"
        data-size="sm"
        data-variant="ghost"
        type="button"
        onClick={() => {
          mutation.reset();
          setOpen(true);
        }}
      >
        <Icon name="stop" />
        Cancel Run
      </button>
    ) : null,
    overlay: open ? (
      <CancelRunDialog
        reason={reason}
        onReasonChange={(value) => {
          setReason(value);
          setValidationError(undefined);
        }}
        validationError={validationError}
        pending={mutation.isPending}
        error={mutation.error}
        onSubmit={() => void submit()}
        onCancel={close}
      />
    ) : null,
  };
}

/**
 * The Run page header actions: Retry model connection, Continue from failed
 * stage, Configure another Run and Cancel Run. The three recovery actions
 * stay separate, each with its own confirmation (two dialogs and the
 * reviewed repeat draft); none changes the displayed state before the Server
 * confirms it. Rendered as the action group plus a notices block, which the
 * header grid places.
 */
export function RunActions({
  run,
  refresh,
  onDone,
  focusHeading,
}: {
  run: RunStatus;
  refresh: ReactNode;
  onDone: ActionDone;
  focusHeading: FocusHeading;
}) {
  const retry = useRetryModelConnection(run, onDone, focusHeading);
  const resume = useContinueFromFailedStage(run, onDone, focusHeading);
  const repeat = useConfigureAnotherRun(run);
  const cancel = useCancelRun(run, onDone, focusHeading);
  const hasNotice = repeat.notice !== undefined && repeat.notice !== null;
  return (
    <>
      <div className="runs-run-actions">
        {retry.control}
        {resume.control}
        {repeat.control}
        {cancel.control}
        {refresh}
      </div>
      {hasNotice ? (
        <div className="runs-run-notices">{repeat.notice}</div>
      ) : null}
      {retry.overlay}
      {resume.overlay}
      {cancel.overlay}
    </>
  );
}
