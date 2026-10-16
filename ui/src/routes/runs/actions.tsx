import { useMutation, useQueryClient } from "@tanstack/react-query";
import { type ReactNode, useRef, useState } from "react";
import { Link } from "react-router";

import { usePublicAPI } from "../../api/context";
import { queryKeys } from "../../api/query-keys";
import {
  CANCELLATION_REASON_LIMIT,
  cancellationReasonLength,
  cancelRun,
  getRunRepeatDraft,
  isTerminalRunState,
  resumeRun,
  type RunStatus,
} from "../../api/runs";
import { ErrorNotice } from "../../app/error-notice";
import { Icon } from "../../app/icon";
import { useRunDraftStore } from "../../run-drafts/context";
import { auditDestination, prepareRepeatDraft } from "../../run-drafts/repeat";
import { CancelRunDialog } from "./cancel";
import { useRetryModelConnection } from "./recovery";
import { useGuardedNavigate } from "../../app/page-location";
import { AlertDialog, Button } from "../../ui";

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

/** The failed stage a Continue confirmation was opened for. */
interface ContinueTarget {
  /** The stage execution the Server offered as the continuation source. */
  source: string;
  /** Its stage name, as the confirmation shows it. */
  stage: string;
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
        <Button
          variant="primary"
          size="sm"
          icon={<Icon name="play" />}
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
          Continue from failed stage
        </Button>
      ),
    overlay:
      target === undefined ? null : (
        // It names the stage and warns that model and tool calls, including
        // external side effects, repeat (S06).
        <AlertDialog
          eyebrow="Continue failed Run"
          title="Continue from failed stage?"
          tone="primary"
          cancelLabel="Back"
          confirmLabel="Confirm continuation"
          pendingLabel="Continuing…"
          pending={mutation.isPending}
          error={
            mutation.error === null ? null : (
              <ErrorNotice error={mutation.error} />
            )
          }
          onConfirm={() => confirm(target)}
          onCancel={dismiss}
        >
          <p>
            Retry stage <strong>{target.stage}</strong>? Model and tool calls
            will run again and may repeat external side effects.
          </p>
          <p>
            Successful stages and their results are preserved. The failed stage
            gets a new attempt with its saved inputs and configuration.
          </p>
        </AlertDialog>
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
  const navigate = useGuardedNavigate();
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
      <Button
        size="sm"
        icon={<Icon name="plus" />}
        pending={mutation.isPending}
        pendingLabel="Loading retained request…"
        onClick={() => void configureAnotherRun()}
      >
        Configure another Run
      </Button>
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
    const length = cancellationReasonLength(normalized);
    if (length === 0 || length > CANCELLATION_REASON_LIMIT) {
      setValidationError(
        `Give a cancellation reason of 1–${CANCELLATION_REASON_LIMIT} characters.`,
      );
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
      <Button
        variant="ghost"
        size="sm"
        icon={<Icon name="stop" />}
        onClick={() => {
          mutation.reset();
          setOpen(true);
        }}
      >
        Cancel Run
      </Button>
    ) : null,
    overlay: open ? (
      <CancelRunDialog
        reason={reason}
        maxLength={CANCELLATION_REASON_LIMIT}
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
 * confirms it. The page places the buttons in its header, the notices under
 * it and the dialogs anywhere.
 */
export function useRunActions(
  run: RunStatus,
  onDone: ActionDone,
  focusHeading: FocusHeading,
): { controls: ReactNode; notices: ReactNode; overlays: ReactNode } {
  const retry = useRetryModelConnection(run, {
    onRequested: () => onDone("Model connection retry requested."),
    onClosedWithoutOffer: focusHeading,
  });
  const resume = useContinueFromFailedStage(run, onDone, focusHeading);
  const repeat = useConfigureAnotherRun(run);
  const cancel = useCancelRun(run, onDone, focusHeading);
  return {
    controls: (
      <>
        {retry.control}
        {resume.control}
        {repeat.control}
        {cancel.control}
      </>
    ),
    notices:
      repeat.notice === undefined || repeat.notice === null ? null : (
        <div className="runs-run-notices">{repeat.notice}</div>
      ),
    overlays: (
      <>
        {retry.overlay}
        {resume.overlay}
        {cancel.overlay}
      </>
    ),
  };
}
