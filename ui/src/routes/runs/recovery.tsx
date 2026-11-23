import { useMutation, useQueryClient } from "@tanstack/react-query";
import { type ReactNode, useState } from "react";

import { usePublicAPI } from "../../api/context";
import { queryKeys } from "../../api/query-keys";
import { retryRunGateway, type RunStatus } from "../../api/runs";
import { ErrorNotice } from "../../app/error-notice";
import { formatTimestamp } from "../../app/format";
import { Icon } from "../../app/icon";
import { runRecoveryReason } from "../../app/vocabulary";
import { AlertDialog, Button } from "../../ui";

type Recovery = NonNullable<RunStatus["recovery"]>;

/** What automatic model recovery of a waiting Run does now, in one sentence. */
export function recoveryProgress(recovery: Recovery): string {
  return recovery.requiresRetry
    ? "Automatic recovery has paused. Restore the model, then use Retry model connection to continue."
    : recovery.nextRetryAt === undefined
      ? "Waiting for automatic recovery."
      : `Next recovery check: ${formatTimestamp(recovery.nextRetryAt)}.`;
}

/**
 * Retry model connection, wherever a waiting Run offers it (the Run page
 * header, the Inbox): the button and its confirmation. It continues the
 * current invocation, unlike Continue from failed stage (a new attempt) or
 * Configure another Run (a new Run). The confirmation keeps the recovery it
 * was opened for: it stays open, with any refusal, until the user closes
 * it, even when a refetch ends that recovery, and a later recovery never
 * reopens it without a click. Nothing changes on screen before the Server's
 * Run says so: the Run is read again once the request settles.
 */
export function useRetryModelConnection(
  run: RunStatus,
  {
    onRequested,
    onClosedWithoutOffer,
  }: {
    /** The Server accepted the retry. */
    onRequested: () => void;
    /** The user closed the confirmation after the Run stopped offering it. */
    onClosedWithoutOffer?: () => void;
  },
): { control: ReactNode; overlay: ReactNode } {
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

  return {
    control:
      offer === undefined ? null : (
        <Button
          variant="primary"
          size="sm"
          icon={<Icon name="refresh" />}
          onClick={() => {
            retry.reset();
            setTarget(offer);
          }}
        >
          Retry model connection
        </Button>
      ),
    overlay:
      target === undefined ? null : (
        <AlertDialog
          eyebrow="Model recovery"
          title="Retry the model connection?"
          tone="primary"
          cancelLabel="Not now"
          confirmLabel="Confirm retry"
          pendingLabel="Enabling retry…"
          pending={retry.isPending}
          error={
            retry.error === null ? null : <ErrorNotice error={retry.error} />
          }
          onConfirm={() =>
            retry.mutate(undefined, {
              onSuccess: () => {
                clear();
                onRequested();
              },
            })
          }
          onCancel={() => {
            clear();
            if (offer === undefined) onClosedWithoutOffer?.();
          }}
        >
          <p>{runRecoveryReason(target.code)}</p>
          <p>
            The current invocation continues where it stopped. Completed work is
            kept and no new stage attempt starts. Restore the model before you
            retry.
          </p>
        </AlertDialog>
      ),
  };
}
