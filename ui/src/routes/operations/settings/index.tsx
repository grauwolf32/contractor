import "./settings.css";
import { ModelSettings } from "./models";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { type FormEvent, type ReactNode, useState } from "react";
import { Link } from "react-router";

import { usePublicAPI } from "../../../api/context";
import {
  getSchedulerSettings,
  replaceSchedulerSettings,
  schedulerLimitBounds,
  type SchedulerLimits,
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

// Each limit sizes one Scheduler lane pool (SCH-R175); its bounds come from
// UpdateSchedulerSettingsRequest.
const LIMITS = (
  [
    {
      name: "maxConcurrentRuns",
      pool: "model",
      label: "Concurrent Runs using a model",
      fact: "Using a model",
      hint: "Runs whose current step calls a model, such as analysis or review. This bounds the load on the LLM Gateway.",
    },
    {
      name: "maxConcurrentToolRuns",
      pool: "tool",
      label: "Concurrent Runs without a model",
      fact: "Without a model",
      hint: "Runs whose current step calls no model, such as a scanner. Free Runtime Agents bound them as well.",
    },
  ] as const
).map((limit) => ({ ...limit, ...schedulerLimitBounds(limit.name) }));

type LimitDraft = Record<keyof SchedulerLimits, string>;

function savedDraft(settings: SchedulerSettingsSnapshot): LimitDraft {
  return {
    maxConcurrentRuns: String(settings.resource.maxConcurrentRuns),
    maxConcurrentToolRuns: String(settings.resource.maxConcurrentToolRuns),
  };
}

function sameDraft(left: LimitDraft, right: LimitDraft): boolean {
  return LIMITS.every(({ name }) => left[name] === right[name]);
}

function validateLimit(
  { label, minimum, maximum }: (typeof LIMITS)[number],
  value: string,
): string | undefined {
  if (value.trim() === "") {
    return `Enter a maximum from ${minimum} through ${maximum}.`;
  }
  if (!/^[0-9]+$/.test(value)) {
    return `${label} must be a whole number.`;
  }
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < minimum || parsed > maximum) {
    return `${label} must be from ${minimum} through ${maximum}.`;
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
  // One draft holds both limits: a save always sends the pair the operator
  // sees, against the revision the edit started from.
  const [draft, setDraft] = useState<LimitDraft>();
  const [editBaseline, setEditBaseline] = useState<SchedulerSettingsSnapshot>();
  const [conflict, setConflict] = useState<string>();
  const [saved, setSaved] = useState<{
    revision: string;
    message: string;
  }>();

  const mutation = useMutation({
    mutationFn: ({ limits, etag }: { limits: SchedulerLimits; etag: string }) =>
      replaceSchedulerSettings(api, limits, etag),
    onSuccess: (current) => {
      queryClient.setQueryData(queryKeys.operations.schedulerSettings, current);
      setEditBaseline(undefined);
      setDraft(undefined);
      setConflict(undefined);
      setSaved({
        revision: current.resource.revision,
        message: `Saved ${current.resource.maxConcurrentRuns} using a model and ${current.resource.maxConcurrentToolRuns} without a model.`,
      });
    },
    onError: (error) => {
      setSaved(undefined);
      if (writeConflict(error) === "stale") {
        setConflict(
          "Another operator saved Scheduler settings first. The saved values are being reloaded; no overwrite occurred.",
        );
        void query.refetch();
      }
    },
  });

  const displayedDraft =
    draft ??
    (query.data === undefined
      ? { maxConcurrentRuns: "", maxConcurrentToolRuns: "" }
      : savedDraft(query.data));
  const validationErrors = Object.fromEntries(
    LIMITS.map((limit) => [
      limit.name,
      validateLimit(limit, displayedDraft[limit.name]),
    ]),
  ) as Record<keyof SchedulerLimits, string | undefined>;
  const invalid = LIMITS.some(
    ({ name }) => validationErrors[name] !== undefined,
  );
  const stale =
    query.data !== undefined &&
    editBaseline !== undefined &&
    (query.data.resource.revision !== editBaseline.resource.revision ||
      query.data.etag !== editBaseline.etag);
  const dirty =
    draft !== undefined &&
    editBaseline !== undefined &&
    !sameDraft(draft, savedDraft(editBaseline));
  const conflictMessage =
    conflict ??
    (stale
      ? "Scheduler settings changed on the Server while this form was being edited. Reset to the saved values before making a new update."
      : undefined);

  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSaved(undefined);
    if (
      invalid ||
      editBaseline === undefined ||
      query.data === undefined ||
      stale
    ) {
      return;
    }
    mutation.mutate({
      limits: {
        maxConcurrentRuns: Number(displayedDraft.maxConcurrentRuns),
        maxConcurrentToolRuns: Number(displayedDraft.maxConcurrentToolRuns),
      },
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
            Set how many Runs the Scheduler progresses at once. Runs using a
            model and Runs without one, such as scans, have separate limits, so
            scans never wait behind model work. Actual throughput can be lower
            when compatible Runtime Agent slots are unavailable, and one
            Workflow Run may require several agents.
          </p>
          <div className="ops-callout">
            <StatusGlyph tone="info" size={15} />
            <div>
              <strong>Admission, not capacity.</strong>
              <p>
                Lowering a limit drains existing work naturally. It never
                cancels active Runs or allocations. When both kinds of Run wait
                for the same Runtime Agent, the Run using a model gets it.
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
                ...LIMITS.map(({ name, fact }) => ({
                  term: fact,
                  value: data.resource[name],
                })),
                {
                  term: "Last updated",
                  value: formatTimestamp(data.resource.updatedAt),
                },
              ]}
              technical={[
                ...LIMITS.map(({ name, fact, pool }) => ({
                  term: fact,
                  value: (
                    <>
                      <code>{name}</code>, the <code>{pool}</code> lane pool
                    </>
                  ),
                })),
                {
                  term: "Revision",
                  value: <code>{data.resource.revision}</code>,
                },
              ]}
            />
            {LIMITS.map((limit) => (
              <Field
                key={limit.name}
                className="ops-number-field"
                label={limit.label}
                hint={`${limit.hint} A whole number from ${limit.minimum} through ${limit.maximum}.`}
                error={validationErrors[limit.name]}
              >
                {(control) => (
                  <input
                    {...control}
                    name={limit.name}
                    type="number"
                    inputMode="numeric"
                    min={limit.minimum}
                    max={limit.maximum}
                    step={1}
                    required
                    value={displayedDraft[limit.name]}
                    onChange={(event) => {
                      const next = {
                        ...displayedDraft,
                        [limit.name]: event.target.value,
                      };
                      if (sameDraft(next, savedDraft(data))) {
                        setDraft(undefined);
                        setEditBaseline(undefined);
                        setConflict(undefined);
                      } else {
                        setEditBaseline((current) => current ?? data);
                        setDraft(next);
                      }
                      setSaved(undefined);
                      mutation.reset();
                    }}
                  />
                )}
              </Field>
            ))}
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
                disabled={invalid || !dirty || stale}
                pending={mutation.isPending}
                pendingLabel="Saving…"
              >
                Save scheduling limits
              </Button>
              <Button
                aria-label="Reset to saved values"
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
                Reload saved values
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
        <>
          {" "}
          <a href="#model-selection">
            Models <small>Defaults for new executions</small>
          </a>
          <a href="#workflow-scheduling">
            Workflow scheduling <small>Server-wide policy</small>
          </a>
        </>
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
      {canManageScheduler ? (
        <>
          <SchedulerSettings />
          <ModelSettings />
        </>
      ) : null}
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
