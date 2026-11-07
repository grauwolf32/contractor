import { useId, useRef } from "react";
import { Link } from "react-router";

import { ErrorNotice } from "../../../app/error-notice";
import { Icon } from "../../../app/icon";
import { checkStateLabel } from "../../../app/vocabulary";
import { Button, ButtonLink, Field, useShortcuts } from "../../../ui";
import { AdvancedOptions } from "./advanced";
import { ScopeCard } from "./scope";
import { presentCheckType, checkTypeIcon, scopeSummary } from "./check-types";
import { TextAreaField, TextField } from "./fields";
import { isLostResponse, type StartFailure } from "./launch";
import { MaterialsCard } from "./materials";
import { projectPaths } from "./paths";
import { SCOPE_TEXT_LIMITS } from "./request";
import {
  isTimeLimitChoice,
  MAX_CUSTOM_HOURS,
  MIN_CUSTOM_HOURS,
  TIME_LIMIT_OPTIONS,
} from "./time-limit";
import type { StartCheck } from "./use-start-check";
import { formatTimeLimit } from "../../../app/format";
import { checkPath } from "../../projects/audits/check-links";

function labelOf(model: StartCheck, checkType: string): string {
  const family = model.catalog.families.find(
    (candidate) => candidate.name === checkType,
  );
  return family === undefined
    ? checkType
    : presentCheckType(family.preferred).label;
}

/** Breadcrumb back to the project and Cancel. */
export function SetupHeader({ model }: { model: StartCheck }) {
  const paths = projectPaths(model.projectId);
  return (
    <header className="start-detail-header">
      <nav aria-label="Breadcrumb" className="start-breadcrumb">
        <ol>
          <li>
            <Link to={paths.overview}>{model.project?.name ?? "Project"}</Link>
            <Icon name="chevron-right" size={13} strokeWidth={2} />
          </li>
          <li>
            <span aria-current="page">New check</span>
          </li>
        </ol>
      </nav>
      <ButtonLink variant="ghost" size="sm" to={paths.overview}>
        Cancel
      </ButtonLink>
    </header>
  );
}

export function ObjectiveField({ model }: { model: StartCheck }) {
  const id = useId();
  const { suggestion } = model.suggestions;
  const chosen = model.selection?.family.name;

  let line;
  if (model.objective.trim() === "")
    line =
      "Describe what you want to find out. Contractor suggests a check type from your words and the project's materials.";
  else if (suggestion === undefined)
    line =
      "No suggestion for these words with this project's materials. Pick a check type from the list.";
  else if (suggestion.checkType === chosen)
    line = (
      <>
        Contractor suggests {labelOf(model, suggestion.checkType)} for this
        objective. You can pick another type.
      </>
    );
  else
    line = (
      <>
        Contractor suggests {labelOf(model, suggestion.checkType)} for this
        objective.{" "}
        <Link to={model.href({ type: suggestion.checkType })}>
          Use {labelOf(model, suggestion.checkType)}
        </Link>
      </>
    );

  return (
    <div className="start-objective">
      <label htmlFor={`${id}-objective`} className="start-objective-label">
        Your objective, in your own words
      </label>
      <textarea
        id={`${id}-objective`}
        className="start-objective-field"
        rows={2}
        value={model.objective}
        maxLength={SCOPE_TEXT_LIMITS.objective}
        placeholder="For example: check the shop API for broken access control"
        aria-describedby={`${id}-suggestion`}
        onChange={(event) => model.setObjective(event.target.value)}
      />
      <p id={`${id}-suggestion`} className="start-suggestion-line">
        <Icon name="bulb" size={14} strokeWidth={2} />
        <span>{line}</span>
      </p>
    </div>
  );
}

function TypeSection({ model }: { model: StartCheck }) {
  const headingId = useId();
  const { selection, exact, detail } = model;
  if (selection === undefined) return null;
  const profile = exact ?? selection.profile;
  return (
    <section className="start-type" aria-labelledby={headingId}>
      <div className="start-selection-head">
        <span className="start-selection-icon">
          <Icon name={checkTypeIcon(profile)} size={20} />
        </span>
        <div className="start-selection-copy">
          <h2 id={headingId}>{selection.presentation.label}</h2>
          <p className="start-type-description">
            {selection.presentation.description}
          </p>
        </div>
      </div>
      {detail.error !== null && exact === undefined ? (
        <ErrorNotice
          error={detail.error}
          context="This check type could not be loaded."
          onRetry={() => void detail.refetch()}
          retryPending={detail.isFetching}
        />
      ) : null}
      {Object.values(profile.program?.workflows ?? {}).some(
        (binding) => binding.outputs.work !== undefined,
      ) ? (
        <section
          className="start-preparation-plan"
          aria-label="Planned check stages"
        >
          <h3>What will happen</h3>
          <ol>
            <li>
              Pin the selected materials and prepare the required outputs.
            </li>
            <li>
              Validate the generated input and accept the endpoint worklist.
            </li>
            <li>
              Run the selected checks under the project's active-check policy.
            </li>
            <li>Review observations, evidence and remaining coverage gaps.</li>
          </ol>
          <p className="ui-caption">
            Endpoint count and coverage become available after preparation.
            Missing or invalid generated input stops the check with an
            explanation.
          </p>
        </section>
      ) : null}
    </section>
  );
}

function LiveTargetFields({ model }: { model: StartCheck }) {
  const headingId = useId();
  const usesTarget = model.scopeFields.has("target");
  const usesAuthorizationScope = model.scopeFields.has("authorizationScope");
  if (!usesTarget && !usesAuthorizationScope) return null;
  return (
    <section className="start-card" aria-labelledby={headingId}>
      <div className="start-card-head">
        <h3 id={headingId}>Live target</h3>
        <span className="ui-caption">Passed to the check&apos;s workers</span>
      </div>
      <p className="ui-caption">
        This check type sends requests to the target you name. Test only what
        you are allowed to.
      </p>
      <div className="start-fields">
        {usesTarget ? (
          <TextField
            label="Target"
            value={model.target}
            onChange={model.setTarget}
            placeholder="https://staging.example.com"
            maxLength={SCOPE_TEXT_LIMITS.target}
            required
            hint={
              model.project?.httpTarget === undefined
                ? "The project has no live target; enter the target here."
                : `The project's live target is ${model.project.httpTarget.url}.`
            }
          />
        ) : null}
        {usesAuthorizationScope ? (
          <TextAreaField
            label="Authorization scope"
            value={model.authorizationScope}
            onChange={model.setAuthorizationScope}
            placeholder="For example: staging only, test accounts A and B, no denial of service"
            maxLength={SCOPE_TEXT_LIMITS.authorizationScope}
            required
            hint="What you are allowed to test, in your own words."
          />
        ) : null}
      </div>
    </section>
  );
}

function TimeLimitField({ model }: { model: StartCheck }) {
  const { timeLimit } = model;
  const invalid =
    timeLimit.choice === "custom" && model.deadlineSeconds === undefined;
  return (
    <div className="start-time">
      <div className="start-time-fields">
        <Field label="Time limit">
          {(control) => (
            <select
              {...control}
              value={timeLimit.choice}
              onChange={(event) => {
                const choice = event.target.value;
                if (isTimeLimitChoice(choice))
                  model.setTimeLimit({ ...timeLimit, choice });
              }}
            >
              {TIME_LIMIT_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          )}
        </Field>
        {timeLimit.choice === "custom" ? (
          <Field
            label="Time limit in hours"
            error={
              invalid
                ? `Enter a time limit from ${MIN_CUSTOM_HOURS} to ${MAX_CUSTOM_HOURS} hours (365 days).`
                : undefined
            }
          >
            {(control) => (
              <input
                {...control}
                type="number"
                inputMode="decimal"
                min={MIN_CUSTOM_HOURS}
                max={MAX_CUSTOM_HOURS}
                step="0.01"
                value={timeLimit.hours}
                onChange={(event) =>
                  model.setTimeLimit({
                    ...timeLimit,
                    hours: event.target.value,
                  })
                }
              />
            )}
          </Field>
        ) : null}
      </div>
      <p className="ui-caption">
        The limit counts time waiting in the queue but not pauses. When it is
        reached, the check starts no new work; work already running can finish.
      </p>
    </div>
  );
}

/** The detail pane body: objective, check type, materials and options. */
export function SetupBody({ model }: { model: StartCheck }) {
  return (
    <>
      <TypeSection model={model} />
      <MaterialsCard model={model} />
      <ScopeCard model={model} />
      <LiveTargetFields model={model} />
      <SetupOptions model={model} />
    </>
  );
}

export function SetupOptions({ model }: { model: StartCheck }) {
  return (
    <fieldset
      className="start-web-fields"
      disabled={model.launch.pending !== undefined}
    >
      <TimeLimitField model={model} />
      <AdvancedOptions model={model} />
    </fieldset>
  );
}

/**
 * After a start request failed and nobody knows yet whether it went through
 * (a lost answer, a server or gateway failure, or a check that could not be
 * read again).
 */
function UnconfirmedStartNotice({
  failure,
  open,
  changed,
  onRetry,
}: {
  failure: Extract<StartFailure, { kind: "lost" | "unconfirmed" }>;
  open: string;
  changed: boolean;
  onRetry: () => void;
}) {
  return (
    <>
      {failure.kind === "lost" ? null : (
        <ErrorNotice
          error={failure.error}
          context="The start request failed."
        />
      )}
      <div className="notice notice-warning" role="alert">
        <strong>The check may already have started.</strong>
        <p>
          {failure.kind === "lost"
            ? "The answer to the start request did not arrive."
            : failure.stillDraft
              ? "The server still shows it as a draft, but a request that failed this way can still go through."
              : "The page could not read the check again to see whether it started."}{" "}
          {changed
            ? "The form changed since, so Start check would create and start a second check while the first one may be running. “Retry same request” sends the first request again."
            : "“Retry same request” sends it again with the same key, so it cannot start the check twice."}{" "}
          <Link to={open}>Open the check</Link>
        </p>
        <Button size="sm" onClick={onRetry}>
          Retry same request
        </Button>
      </div>
    </>
  );
}

function LaunchNotices({
  model,
}: {
  model: Pick<StartCheck, "launch" | "request" | "projectId">;
}) {
  const { launch, request } = model;
  const draft = launch.draft;
  if (launch.pending !== undefined) return null;
  if (launch.preparationError !== undefined)
    return (
      <ErrorNotice
        error={launch.preparationError}
        context="Could not save the check materials."
      />
    );
  const createError = launch.createError;
  const failure = launch.startFailure;
  // The form no longer matches the check this page created.
  const changed =
    draft !== undefined &&
    (request === undefined || !launch.isDraftRequest(request));
  if (createError !== null && isLostResponse(createError)) {
    const same = request !== undefined && launch.isLastRequest(request);
    return (
      <div className="notice notice-warning" role="alert">
        <strong>The check may already exist.</strong>
        <p>
          {same
            ? "The answer to the last request did not arrive. “Retry same request” sends it again with the same key, so it cannot create a second check."
            : "The form changed after the answer was lost. Sending it now uses a new key and may create a second check; restore the earlier values to retry the first request."}
        </p>
        {same ? (
          <Button size="sm" onClick={launch.retryCreate}>
            Retry same request
          </Button>
        ) : null}
      </div>
    );
  }
  if (createError !== null)
    return (
      <ErrorNotice
        error={createError}
        reconcileWrite
        context="The check was not created."
      />
    );
  if (failure !== undefined && draft !== undefined) {
    const open = checkPath(model.projectId, draft.auditId);
    if (failure.kind === "started")
      return (
        <div className="notice notice-warning" role="alert">
          <strong>
            {draft.state === "deleting"
              ? "This check is being deleted."
              : "This check has already started."}
          </strong>
          <p>
            The server shows it as {checkStateLabel(draft.state).label}.{" "}
            {changed
              ? "The form changed, so Start check creates and starts another check. "
              : null}
            <Link to={open}>Open the check</Link>
          </p>
        </div>
      );
    // A fresh read shows the draft, and the server said why it refused.
    if (failure.kind === "refused")
      return (
        <>
          <ErrorNotice
            error={failure.error}
            context="The check was saved as a draft but did not start."
          />
          <p className="start-tight">
            {failure.revisionChanged
              ? "The check changed on the server after this page read it. The page has read it again; review it before you start it again. "
              : null}
            <Link to={open}>Open the draft</Link> to see it.{" "}
            {changed
              ? "The form changed, so starting now creates a new check; the draft stays."
              : "Start check tries this draft again."}
          </p>
        </>
      );
    return (
      <UnconfirmedStartNotice
        failure={failure}
        open={open}
        changed={changed}
        onRetry={launch.retryStart}
      />
    );
  }
  if (draft !== undefined && changed)
    return (
      <p className="ui-caption">
        {draft.state === "draft"
          ? "Your earlier attempt is saved as a draft ("
          : "Your earlier check has started ("}
        <Link to={checkPath(model.projectId, draft.auditId)}>open it</Link>).
        Starting now creates {draft.state === "draft" ? "a new" : "another"}{" "}
        check.
      </p>
    );
  return null;
}

/** False while a narrow layout hides the pane the element is in. */
function isRendered(element: Element): boolean {
  return typeof element.checkVisibility === "function"
    ? element.checkVisibility()
    : true;
}

/** The pinned bar: a plain summary, Save as draft and Start check. */
export function SetupFooter({ model }: { model: StartCheck }) {
  const id = useId();
  const startButton = useRef<HTMLButtonElement>(null);
  const { launch, selection } = model;
  const pending = launch.pending !== undefined;
  // Ctrl/⌘+Enter starts the check from any field of the setup, and from
  // anywhere else on the page while the Start check button shows. The
  // binding stays on while starting is blocked, so the key never types a
  // line break into a field instead; model.start() checks the blockers.
  useShortcuts({
    "mod+enter": {
      // On a link the key is the link's own (it opens the link in a new
      // tab), so the event is left alone.
      when: (event) =>
        !(
          event.target instanceof Element &&
          event.target.closest("a[href], .start-type-picker") !== null
        ),
      handler: () => {
        const button = startButton.current;
        if (button === null || !isRendered(button)) return;
        model.start();
      },
    },
  });
  if (selection === undefined) return null;
  const status =
    launch.pending === undefined
      ? ""
      : launch.step === "prepare"
        ? "Saving check materials…"
        : launch.step === "verify"
          ? "Checking whether the check started…"
          : launch.step === "start"
            ? "Starting the check…"
            : launch.pending === "draft"
              ? "Saving the draft…"
              : "Creating the check…";
  const hint = pending ? undefined : model.startBlocker;
  const projectName = model.project?.name ?? "this project";
  const profile = model.exact ?? selection.profile;
  const selected = model.slots.filter(
    (slot) => slot.selected !== undefined,
  ).length;
  const ready = model.startBlocker === undefined;
  const scope = scopeSummary(profile);
  return (
    <aside className="start-summary-card" aria-label="Your check">
      <h3>Your check</h3>
      <p className="start-summary-name">{selection.presentation.label}</p>
      <dl className="start-summary-facts">
        <div>
          <dt>Project</dt>
          <dd>{projectName}</dd>
        </div>
        <div>
          <dt>Materials</dt>
          <dd>
            {model.webScan === undefined
              ? `${selected} of ${model.slots.length} selected`
              : "Target and scan settings"}
          </dd>
        </div>
        {scope === undefined ? null : (
          <div>
            <dt>Scope</dt>
            <dd>{scope.size}</dd>
          </div>
        )}
        <div>
          <dt>Time limit</dt>
          <dd>
            {model.deadlineSeconds === undefined
              ? "Review the limit"
              : model.deadlineSeconds === 0
                ? "No time limit"
                : formatTimeLimit(model.deadlineSeconds)}
          </dd>
        </div>
      </dl>
      <div className="start-readiness" data-tone={ready ? "ready" : "missing"}>
        <span>{ready ? "Ready to start" : "Review the setup"}</span>
        <p id={`${id}-hint`}>
          {hint ??
            "Required inputs are selected. The Server validates the exact materials at start."}
        </p>
      </div>
      <div className="start-actions">
        <Button
          ref={startButton}
          variant="primary"
          className="start-primary"
          disabled={pending || model.startBlocker !== undefined}
          pending={launch.pending === "start"}
          aria-describedby={hint === undefined ? undefined : `${id}-hint`}
          aria-keyshortcuts="Control+Enter Meta+Enter"
          onClick={model.start}
        >
          {launch.pending === "start" ? "Starting…" : "Start check"}
        </Button>
        <Button
          variant="ghost"
          disabled={pending || model.saveBlocker !== undefined}
          pending={launch.pending === "draft"}
          pendingLabel="Saving…"
          aria-describedby={`${id}-draft`}
          onClick={model.saveDraft}
        >
          Save as draft
        </Button>
      </div>
      <span id={`${id}-draft`} className="ui-visually-hidden">
        Creates the check without starting it. Start it later from its page.
      </span>
      <LaunchNotices model={model} />
      <p className="ui-visually-hidden" role="status">
        {status}
      </p>
    </aside>
  );
}
