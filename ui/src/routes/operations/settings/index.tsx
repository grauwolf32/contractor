import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { type FormEvent, useState } from "react";
import { Link } from "react-router";

import { usePublicAPI } from "../../../api/context";
import { PublicAPIError } from "../../../api/error";
import {
  getSchedulerSettings,
  replaceSchedulerSettings,
  type SchedulerSettingsSnapshot,
} from "../../../api/operations";
import { queryKeys } from "../../../api/query-keys";
import { ErrorNotice, formatTimestamp } from "../../artifacts/common";

function validateMaximum(value: string): string | undefined {
  if (value.trim() === "") {
    return "Enter a maximum from 1 through 32.";
  }
  if (!/^[0-9]+$/.test(value)) {
    return "Maximum concurrent Workflow Runs must be a whole number.";
  }
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < 1 || parsed > 32) {
    return "Maximum concurrent Workflow Runs must be from 1 through 32.";
  }
  return undefined;
}

export function OperationsSettingsRoute() {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const query = useQuery({
    queryKey: queryKeys.operations.schedulerSettings,
    queryFn: () => getSchedulerSettings(api),
  });
  const [draft, setDraft] = useState<string>();
  const [editBaseline, setEditBaseline] = useState<SchedulerSettingsSnapshot>();
  const [conflict, setConflict] = useState<string>();
  const [saved, setSaved] = useState<{
    revision: string;
    message: string;
  }>();

  const mutation = useMutation({
    mutationFn: ({ value, etag }: { value: number; etag: string }) =>
      replaceSchedulerSettings(api, value, etag),
    onSuccess: (current) => {
      queryClient.setQueryData(queryKeys.operations.schedulerSettings, current);
      setEditBaseline(undefined);
      setDraft(undefined);
      setConflict(undefined);
      setSaved({
        revision: current.resource.revision,
        message: `Saved maximum ${current.resource.maxConcurrentRuns} at revision ${current.resource.revision}.`,
      });
    },
    onError: (error) => {
      setSaved(undefined);
      if (error instanceof PublicAPIError && error.status === 412) {
        setConflict(
          "Another operator saved Scheduler settings first. The authoritative value is being reloaded; no overwrite occurred.",
        );
        void query.refetch();
      }
    },
  });

  const displayedDraft =
    draft ??
    (query.data === undefined
      ? ""
      : String(query.data.resource.maxConcurrentRuns));
  const validationError = validateMaximum(displayedDraft);
  const stale =
    query.data !== undefined &&
    editBaseline !== undefined &&
    (query.data.resource.revision !== editBaseline.resource.revision ||
      query.data.etag !== editBaseline.etag);
  const dirty =
    draft !== undefined &&
    editBaseline !== undefined &&
    draft !== String(editBaseline.resource.maxConcurrentRuns);
  const conflictMessage =
    conflict ??
    (stale
      ? "Scheduler settings changed on the Server while this form was being edited. Reset to the saved value before making a new exact update."
      : undefined);

  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSaved(undefined);
    if (
      validationError !== undefined ||
      editBaseline === undefined ||
      query.data === undefined ||
      stale
    ) {
      return;
    }
    mutation.mutate({
      value: Number(displayedDraft),
      etag: editBaseline.etag,
    });
  };

  const reset = () => {
    if (query.data === undefined) {
      return;
    }
    mutation.reset();
    setEditBaseline(undefined);
    setDraft(undefined);
    setConflict(undefined);
    setSaved(undefined);
  };

  return (
    <div className="operations-library">
      <header className="section-heading">
        <div>
          <p className="eyebrow">Durable global policy</p>
          <h3>Settings</h3>
        </div>
      </header>

      <section
        className="panel configuration-draft"
        aria-labelledby="workflow-scheduling-heading"
      >
        <div>
          <p className="eyebrow">Workflow Scheduler</p>
          <h3 id="workflow-scheduling-heading">Workflow scheduling</h3>
          <p className="muted-copy">
            This is an upper bound on active Scheduler lanes. Actual concurrency
            can be lower when compatible Runtime Agent slots are unavailable,
            and one Workflow Run may require several agents.
          </p>
        </div>

        {query.isPending ? (
          <p className="loading-copy" aria-live="polite">
            Loading saved scheduling settings…
          </p>
        ) : query.data === undefined ? (
          <>
            <ErrorNotice error={query.error} />
            <button
              className="secondary-button"
              type="button"
              disabled={query.isFetching}
              onClick={() => void query.refetch()}
            >
              {query.isFetching ? "Reloading…" : "Retry settings load"}
            </button>
          </>
        ) : (
          <form onSubmit={submit}>
            <dl className="key-value-list">
              <div>
                <dt>Saved value</dt>
                <dd>{query.data.resource.maxConcurrentRuns}</dd>
              </div>
              <div>
                <dt>Revision</dt>
                <dd>
                  <code>{query.data.resource.revision}</code>
                </dd>
              </div>
              <div>
                <dt>Updated</dt>
                <dd>{formatTimestamp(query.data.resource.updatedAt)}</dd>
              </div>
            </dl>

            <div className="form-grid">
              <label>
                Maximum concurrent Workflow Runs
                <input
                  aria-label="Maximum concurrent Workflow Runs"
                  name="maxConcurrentRuns"
                  type="number"
                  inputMode="numeric"
                  min={1}
                  max={32}
                  step={1}
                  required
                  value={displayedDraft}
                  aria-invalid={validationError !== undefined}
                  aria-describedby="scheduler-maximum-guidance scheduler-maximum-error"
                  onChange={(event) => {
                    const value = event.target.value;
                    if (
                      value === String(query.data.resource.maxConcurrentRuns)
                    ) {
                      setDraft(undefined);
                      setEditBaseline(undefined);
                      setConflict(undefined);
                    } else {
                      setEditBaseline((current) => current ?? query.data);
                      setDraft(value);
                    }
                    setSaved(undefined);
                    mutation.reset();
                  }}
                />
                <small
                  id="scheduler-maximum-guidance"
                  className="field-guidance"
                >
                  A whole number from 1 through 32.
                </small>
              </label>
            </div>
            {validationError === undefined ? null : (
              <p
                id="scheduler-maximum-error"
                className="field-error"
                role="alert"
              >
                {validationError}
              </p>
            )}
            {conflictMessage === undefined ? null : (
              <div className="notice notice-warning" role="alert">
                {conflictMessage}
              </div>
            )}
            {mutation.error === null ||
            (mutation.error instanceof PublicAPIError &&
              mutation.error.status === 412) ? null : (
              <ErrorNotice error={mutation.error} reconcileWrite />
            )}
            {query.error === null || query.isPending ? null : (
              <ErrorNotice error={query.error} />
            )}
            {saved === undefined ||
            saved.revision !== query.data.resource.revision ? null : (
              <div className="notice notice-success" role="status">
                {saved.message}
              </div>
            )}

            <div className="project-form-actions">
              <button
                type="submit"
                disabled={
                  mutation.isPending ||
                  validationError !== undefined ||
                  !dirty ||
                  stale
                }
              >
                {mutation.isPending ? "Saving…" : "Save scheduling limit"}
              </button>
              <button
                className="secondary-button"
                type="button"
                disabled={mutation.isPending || (!dirty && !stale)}
                onClick={reset}
              >
                Reset to saved value
              </button>
              <button
                className="secondary-button"
                type="button"
                disabled={mutation.isPending || query.isFetching}
                onClick={() => void query.refetch()}
              >
                {query.isFetching ? "Reloading…" : "Reload saved value"}
              </button>
            </div>
          </form>
        )}

        <div className="notice notice-info">
          <strong>Admission, not capacity.</strong>
          <p>
            Decreasing this value drains existing work naturally; it does not
            cancel active Runs or allocations. Owner-scoped Queue Pause remains
            a separate control under{" "}
            <Link to="/runs?view=queue">Runs / Queue</Link>.
          </p>
        </div>
      </section>
    </div>
  );
}
