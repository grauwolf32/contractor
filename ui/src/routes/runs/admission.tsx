import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { ReactNode } from "react";

import { usePublicAPI } from "../../api/context";
import { getOwnerQueueControl, setOwnerQueuePaused } from "../../api/queue";
import { queryKeys } from "../../api/query-keys";
import { ErrorNotice } from "../../app/error-notice";
import { Icon } from "../../app/icon";
import { StatusChip } from "../../ui";

/**
 * The owner's durable queue admission gate (S18) for the Runs header: the
 * admission status, the Pause queue / Resume queue action and the notes
 * that go under the header. The control reflects only what the Server
 * returned; a write that fails refetches the authoritative state.
 */
export function useQueueAdmission(): { control: ReactNode; notice: ReactNode } {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const controlQuery = useQuery({
    queryKey: queryKeys.queue.control,
    queryFn: () => getOwnerQueueControl(api),
  });
  const controlMutation = useMutation({
    mutationFn: ({ paused, revision }: { paused: boolean; revision: string }) =>
      setOwnerQueuePaused(api, paused, revision),
    onSuccess: (control) => {
      // The Server's response to the write is the new authoritative state.
      queryClient.setQueryData(queryKeys.queue.control, control);
    },
    onError: async () => {
      await queryClient.invalidateQueries({
        queryKey: queryKeys.queue.control,
      });
    },
  });

  const control = controlQuery.data;
  const paused = control?.paused === true;
  const status = controlQuery.isPending ? (
    <StatusChip tone="idle" size="sm">
      Checking admission…
    </StatusChip>
  ) : control === undefined ? (
    <StatusChip tone="neutral" size="sm">
      Admission unknown
    </StatusChip>
  ) : paused ? (
    <StatusChip tone="warning" size="sm">
      Admission paused
    </StatusChip>
  ) : (
    <StatusChip tone="success" size="sm">
      Admission running
    </StatusChip>
  );

  return {
    control: (
      <div className="runs-admission">
        <span className="runs-admission-status" aria-live="polite">
          {status}
        </span>
        <button
          className="ui-btn"
          data-size="sm"
          type="button"
          disabled={
            control === undefined ||
            controlQuery.error !== null ||
            controlMutation.isPending
          }
          onClick={() => {
            if (control !== undefined) {
              controlMutation.mutate({
                paused: !control.paused,
                revision: control.revision,
              });
            }
          }}
        >
          <Icon name={paused ? "play" : "pause"} />
          {controlMutation.isPending
            ? paused
              ? "Resuming…"
              : "Pausing…"
            : paused
              ? "Resume queue"
              : "Pause queue"}
        </button>
      </div>
    ),
    notice:
      controlQuery.error !== null ? (
        <ErrorNotice
          error={controlQuery.error}
          context="Could not read the queue admission state"
          onRetry={() => void controlQuery.refetch()}
          retryPending={controlQuery.isFetching}
        />
      ) : controlMutation.error !== null ? (
        <ErrorNotice
          error={controlMutation.error}
          context="The queue admission change was not applied"
        />
      ) : paused ? (
        <p className="runs-admission-note" role="status">
          Running Stages and cleanup will finish. No next Stage starts until you
          resume the queue.
        </p>
      ) : null,
  };
}
