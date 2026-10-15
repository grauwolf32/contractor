import "./settings.css";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { type FormEvent, type ReactNode, useState } from "react";
import { Link } from "react-router";

import { schemaLimit } from "../../../api/contract";
import { usePublicAPI } from "../../../api/context";
import {
  getSchedulerSettings,
  replaceSchedulerSettings,
  type SchedulerSettingsSnapshot,
} from "../../../api/operations";
import { queryKeys } from "../../../api/query-keys";
import { useSession } from "../../../auth/session";
import { writeConflict } from "../../../app/conflict";
import { ErrorNotice } from "../../../app/error-notice";
import { formatTimestamp } from "../../../app/format";
import { QueryView } from "../../../app/query-view";
import { Button, Facts, Field, StatusGlyph } from "../../../ui";
import { AppearanceSettings } from "../../settings/appearance";
import { GitKeySettings } from "../../settings/git-key";
import { SettingSection } from "../../settings/section";

// The admission ceiling's bounds, as UpdateSchedulerSettingsRequest has them.
const MINIMUM = schemaLimit(
  "UpdateSchedulerSettingsRequest",
  "minimum",
  "properties",
  "maxConcurrentRuns",
);
const MAXIMUM = schemaLimit(
  "UpdateSchedulerSettingsRequest",
  "maximum",
  "properties",
  "maxConcurrentRuns",
);

function validateMaximum(value: string): string | undefined {
  if (value.trim() === "") {
    return `Enter a maximum from ${MINIMUM} through ${MAXIMUM}.`;
  }
  if (!/^[0-9]+$/.test(value)) {
    return "Maximum concurrent Workflow Runs must be a whole number.";
  }
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < MINIMUM || parsed > MAXIMUM) {
    return `Maximum concurrent Workflow Runs must be from ${MINIMUM} through ${MAXIMUM}.`;
  }
  return undefined;
}

function SchedulerSettings() {
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
      if (writeConflict(error) === "stale") {
        setConflict(
          "Another operator saved Scheduler settings first. The saved value is being reloaded; no overwrite occurred.",
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
      ? "Scheduler settings changed on the Server while this form was being edited. Reset to the saved value before making a new update."
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
    <SettingSection
      id="workflow-scheduling"
      eyebrow="Workflow Scheduler"
      title="Workflow scheduling"
      scope="Server-wide"
      about={
        <>
          <p>
            Set the admission ceiling for active Scheduler lanes. Actual
            throughput can be lower when compatible Runtime Agent slots are
            unavailable, and one Workflow Run may require several agents.
          </p>
          <div className="ops-callout">
            <StatusGlyph tone="info" size={15} />
            <div>
              <strong>Admission, not capacity.</strong>
              <p>
                Lowering the limit drains existing work naturally. It never
                cancels active Runs or allocations.
              </p>
            </div>
          </div>
          <p className="ui-caption">
            Queue admission remains separate under{" "}
            <Link to="/runs?view=queue">Runs / Queue</Link>.
          </p>
        </>
      }
    >
      <QueryView
        query={query}
        loading={
          <p className="ops-loading" role="status">
            Loading saved scheduling settings…
          </p>
        }
      >
        {(data) => (
          <form className="ops-setting-form" onSubmit={submit}>
            <Facts
              className="ops-glance"
              aria-label="Saved Scheduler settings"
              items={[
                { term: "Saved limit", value: data.resource.maxConcurrentRuns },
                {
                  term: "Revision",
                  value: <code>{data.resource.revision}</code>,
                },
                {
                  term: "Last updated",
                  value: formatTimestamp(data.resource.updatedAt),
                },
              ]}
            />
            <Field
              className="ops-number-field"
              label="Maximum concurrent Workflow Runs"
              hint={`A whole number from ${MINIMUM} through ${MAXIMUM}.`}
              error={validationError}
            >
              {(control) => (
                <input
                  {...control}
                  name="maxConcurrentRuns"
                  type="number"
                  inputMode="numeric"
                  min={MINIMUM}
                  max={MAXIMUM}
                  step={1}
                  required
                  value={displayedDraft}
                  onChange={(event) => {
                    const value = event.target.value;
                    if (value === String(data.resource.maxConcurrentRuns)) {
                      setDraft(undefined);
                      setEditBaseline(undefined);
                      setConflict(undefined);
                    } else {
                      setEditBaseline((current) => current ?? data);
                      setDraft(value);
                    }
                    setSaved(undefined);
                    mutation.reset();
                  }}
                />
              )}
            </Field>
            {conflictMessage === undefined ? null : (
              <div className="notice notice-warning" role="alert">
                {conflictMessage}
              </div>
            )}
            {mutation.error === null ||
            writeConflict(mutation.error) === "stale" ? null : (
              <ErrorNotice error={mutation.error} reconcileWrite />
            )}
            {saved === undefined ||
            saved.revision !== data.resource.revision ? null : (
              <div className="notice notice-success" role="status">
                {saved.message}
              </div>
            )}

            <div className="ops-form-actions">
              <Button
                variant="primary"
                type="submit"
                disabled={validationError !== undefined || !dirty || stale}
                pending={mutation.isPending}
                pendingLabel="Saving…"
              >
                Save scheduling limit
              </Button>
              <Button
                aria-label="Reset to saved value"
                disabled={mutation.isPending || (!dirty && !stale)}
                onClick={reset}
              >
                Reset
              </Button>
              <Button
                variant="ghost"
                className="ops-push-end"
                disabled={mutation.isPending}
                pending={query.isFetching}
                pendingLabel="Reloading…"
                onClick={() => void query.refetch()}
              >
                Reload saved value
              </Button>
            </div>
          </form>
        )}
      </QueryView>
    </SettingSection>
  );
}

function SettingsDirectory({ scheduler }: { scheduler: boolean }) {
  return (
    <nav className="ops-settings-directory" aria-label="Settings on this page">
      {scheduler ? (
        <a href="#workflow-scheduling">
          Workflow scheduling <small>Server-wide policy</small>
        </a>
      ) : null}
      <a href="#repository-access">
        Repository access <small>Personal credential</small>
      </a>
      <a href="#appearance">
        Appearance <small>This browser</small>
      </a>
    </nav>
  );
}

function PersonalSettingsPage({ children }: { children: ReactNode }) {
  return (
    <section
      className="ops-page ops-settings-page"
      aria-labelledby="settings-heading"
    >
      <header className="ops-page-head">
        <div className="ops-page-heading">
          <h1 id="settings-heading" className="ops-page-title">
            Settings
          </h1>
          <p className="ops-page-lede">
            Manage the personal credential used by your private repository
            imports and choose the theme.
          </p>
        </div>
      </header>
      <div className="ops-page-body">{children}</div>
    </section>
  );
}

export function OperationsSettingsRoute() {
  const { session } = useSession();
  const canManageScheduler =
    session?.principal.capabilities.includes("operations") === true;
  // Under Operations the sections sit below the Settings heading (h2);
  // the personal page has only its h1 above them.
  const titleAs = canManageScheduler ? "h3" : "h2";
  const sections = (
    <div className="ops-settings">
      <SettingsDirectory scheduler={canManageScheduler} />
      {canManageScheduler ? <SchedulerSettings /> : null}
      <GitKeySettings titleAs={titleAs} />
      <AppearanceSettings titleAs={titleAs} />
    </div>
  );
  if (!canManageScheduler)
    return <PersonalSettingsPage>{sections}</PersonalSettingsPage>;
  return (
    <div className="ops-stack">
      <header className="ops-section-head">
        <div className="ops-section-heading">
          <h2 className="ops-section-title">Settings</h2>
          <p className="ops-section-description">
            Tune execution admission, manage the credentials used by repository
            imports and choose the theme.
          </p>
        </div>
      </header>
      {sections}
    </div>
  );
}
