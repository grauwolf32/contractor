import { useId, useRef } from "react";
import { Link, useNavigate } from "react-router";

import { ErrorNotice } from "../../../app/error-notice";
import { checkStateLabel } from "../../../app/vocabulary";
import { IdChip, Kbd, modKeyLabel, useShortcuts } from "../../../ui";
import { AdvancedOptions } from "./advanced";
import {
  compatibilityReasonText,
  missingLabel,
  presentCheckType,
  scopeSummary,
} from "./check-types";
import { TextAreaField, TextField } from "./fields";
import { BulbIcon, ChevronIcon, PlayIcon } from "./icons";
import { checkPath, isLostResponse, type StartFailure } from "./launch";
import { MaterialsCard } from "./materials";
import { projectPaths } from "./paths";
import { SCOPE_TEXT_LIMIT } from "./request";
import {
  describeTimeLimit,
  isTimeLimitChoice,
  MAX_CUSTOM_HOURS,
  MIN_CUSTOM_HOURS,
  TIME_LIMIT_OPTIONS,
} from "./time-limit";
import type { StartCheck } from "./use-start-check";

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
            <ChevronIcon />
          </li>
          <li>
            <span aria-current="page">New check</span>
          </li>
        </ol>
      </nav>
      <Link
        className="ui-btn"
        data-variant="ghost"
        data-size="sm"
        to={paths.overview}
      >
        Cancel
      </Link>
    </header>
  );
}

function ObjectiveField({ model }: { model: StartCheck }) {
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
        maxLength={SCOPE_TEXT_LIMIT}
        placeholder="For example: check the shop API for broken access control"
        aria-describedby={`${id}-suggestion`}
        onChange={(event) => model.setObjective(event.target.value)}
      />
      <p id={`${id}-suggestion`} className="start-suggestion-line">
        <BulbIcon />
        <span>{line}</span>
      </p>
    </div>
  );
}

function TypeSection({ model }: { model: StartCheck }) {
  const headingId = useId();
  const { selection, suggestions, readiness, exact, detail } = model;
  if (selection === undefined) return null;
  const paths = projectPaths(model.projectId);
  const profile = exact ?? selection.profile;
  const exactName = `${profile.ref.name}@${profile.ref.version}`;
  const suggested = suggestions.suggestion?.checkType === selection.family.name;
  const alternative = suggested ? suggestions.alternative : undefined;
  return (
    <section className="start-type" aria-labelledby={headingId}>
      <div className="start-type-title">
        <h2 id={headingId}>{selection.presentation.label}</h2>
        {suggested ? <span className="start-badge">Suggested</span> : null}
      </div>
      <div className="start-type-id">
        <IdChip
          value={exactName}
          display={exactName}
          label="check type version"
        />
      </div>
      <p className="start-type-description">
        {selection.presentation.description}
      </p>
      {suggested && suggestions.suggestion !== undefined ? (
        <div className="start-why">
          <BulbIcon size={17} />
          <p>
            <strong>Why this fits:</strong> {suggestions.suggestion.reason}
          </p>
        </div>
      ) : null}
      {alternative === undefined ? null : (
        <p className="start-tight">
          Or try{" "}
          <Link to={model.href({ type: alternative.checkType })}>
            {labelOf(model, alternative.checkType)}
          </Link>
        </p>
      )}
      {detail.error !== null && exact === undefined ? (
        <ErrorNotice
          error={detail.error}
          context="This check type could not be loaded."
          onRetry={() => void detail.refetch()}
          retryPending={detail.isFetching}
        />
      ) : null}
      {/* From the version itself: it does not depend on the materials. */}
      {selection.profile.serverCompatible ? null : (
        <div className="notice notice-error" role="note">
          <strong>This check type can&apos;t run on this server.</strong>
          <p>
            {selection.profile.compatibilityReasons.length === 0
              ? "The server gives no reason."
              : `The server does not support ${selection.profile.compatibilityReasons
                  .map(compatibilityReasonText)
                  .join(", ")}.`}
          </p>
        </div>
      )}
      {readiness?.state === "missing" ? (
        <div className="notice notice-warning" role="note">
          <strong>
            Needs more materials:{" "}
            {readiness.missing.map(missingLabel).join(", ")}
          </strong>
          <p>
            {readiness.missing.some((need) => need.kind === "shared")
              ? "The project has fewer materials in these formats than the check type has inputs. "
              : null}
            {readiness.missing.some((need) => need.kind !== "live-target") ? (
              <>
                <Link to={paths.addMaterial}>Add materials</Link> to the
                project, then come back.{" "}
              </>
            ) : null}
            {readiness.missing.some((need) => need.kind === "live-target") ? (
              <>
                <Link to={paths.settings}>Set the live target</Link> in the
                project settings, or enter a target for this check below.
              </>
            ) : null}
          </p>
        </div>
      ) : null}
    </section>
  );
}

function ScopeCard({ model }: { model: StartCheck }) {
  const id = useId();
  const navigate = useNavigate();
  const { selection } = model;
  if (selection === undefined) return null;
  const { variants } = selection.row.row;
  if (variants.length <= 1) {
    const summary = scopeSummary(model.exact ?? selection.profile);
    return (
      <section className="start-card" aria-labelledby={`${id}-heading`}>
        <div className="start-card-head">
          <h3 id={`${id}-heading`}>Scope</h3>
        </div>
        <p className="start-scope-size">{summary.size}</p>
        <p className="start-quiet start-tight">{summary.detail}</p>
      </section>
    );
  }
  return (
    <fieldset className="start-card start-scope">
      <legend className="start-card-legend">Scope</legend>
      {variants.map((family) => {
        const summary = scopeSummary(family.preferred);
        // Still selectable: choosing it shows the server's reasons, and the
        // footer keeps it from being created or started.
        const unsupported = !family.preferred.serverCompatible;
        return (
          <label key={family.name} className="start-scope-option">
            <input
              type="radio"
              name={`${id}-scope`}
              checked={family.name === selection.family.name}
              onChange={() =>
                void navigate(model.href({ type: family.name }), {
                  replace: true,
                })
              }
            />
            <span>
              <span className="start-scope-size">{summary.size}</span>
              <span className="start-quiet start-scope-detail">
                {family.preferred.inventory.standardSelection === undefined
                  ? `${presentCheckType(family.preferred).label} · ${summary.detail}`
                  : summary.detail}
                {unsupported ? " · not supported by this server" : ""}
              </span>
            </span>
          </label>
        );
      })}
    </fieldset>
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
        <span className="start-quiet">Passed to the check&apos;s workers</span>
      </div>
      <p className="start-quiet start-tight">
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
            maxLength={SCOPE_TEXT_LIMIT}
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
            maxLength={SCOPE_TEXT_LIMIT}
            required
            hint="What you are allowed to test, in your own words."
          />
        ) : null}
      </div>
    </section>
  );
}

function TimeLimitField({ model }: { model: StartCheck }) {
  const id = useId();
  const { timeLimit } = model;
  const invalid =
    timeLimit.choice === "custom" && model.deadlineSeconds === undefined;
  return (
    <div className="start-time">
      <div className="start-time-fields">
        <div className="start-field">
          <label htmlFor={`${id}-choice`}>Time limit</label>
          <select
            id={`${id}-choice`}
            className="start-input start-select"
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
        </div>
        {timeLimit.choice === "custom" ? (
          <div className="start-field">
            <label htmlFor={`${id}-hours`}>Time limit in hours</label>
            <input
              id={`${id}-hours`}
              className="start-input"
              type="number"
              inputMode="decimal"
              min={MIN_CUSTOM_HOURS}
              max={MAX_CUSTOM_HOURS}
              step="0.01"
              value={timeLimit.hours}
              aria-invalid={invalid || undefined}
              aria-describedby={invalid ? `${id}-error` : undefined}
              onChange={(event) =>
                model.setTimeLimit({ ...timeLimit, hours: event.target.value })
              }
            />
          </div>
        ) : null}
      </div>
      <p className="start-field-hint">
        The limit counts time waiting in the queue but not pauses. When it is
        reached, the check starts no new work; work already running can finish.
      </p>
      {invalid ? (
        <p id={`${id}-error`} className="start-field-error" role="alert">
          Enter a time limit from {MIN_CUSTOM_HOURS} to {MAX_CUSTOM_HOURS} hours
          (365 days).
        </p>
      ) : null}
    </div>
  );
}

/** The detail pane body: objective, check type, materials and options. */
export function SetupBody({ model }: { model: StartCheck }) {
  return (
    <>
      <ObjectiveField model={model} />
      <TypeSection model={model} />
      <div className="start-cards">
        <MaterialsCard model={model} />
        <ScopeCard model={model} />
      </div>
      <LiveTargetFields model={model} />
      <TimeLimitField model={model} />
      <AdvancedOptions model={model} />
    </>
  );
}

function timeSentence(seconds: number | undefined): string {
  if (seconds === undefined) return "";
  return seconds === 0
    ? "It runs without a time limit."
    : `It starts no new work after ${describeTimeLimit(seconds)}; work already running can finish.`;
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
        <button
          type="button"
          className="ui-btn"
          data-size="sm"
          onClick={onRetry}
        >
          Retry same request
        </button>
      </div>
    </>
  );
}

function LaunchNotices({ model }: { model: StartCheck }) {
  const { launch, request } = model;
  const draft = launch.draft;
  if (launch.pending !== undefined) return null;
  const createError = launch.createError;
  const failure = launch.startFailure;
  // The form no longer matches the check this page created.
  const changed = request !== undefined && !launch.isDraftRequest(request);
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
          <button
            type="button"
            className="ui-btn"
            data-size="sm"
            onClick={launch.retryCreate}
          >
            Retry same request
          </button>
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
      <p className="start-tight start-quiet">
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
          event.target.closest("a[href]") !== null
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
      : launch.step === "verify"
        ? "Checking whether the check started…"
        : launch.step === "start"
          ? "Starting the check…"
          : launch.pending === "draft"
            ? "Saving the draft…"
            : "Creating the check…";
  const hint = pending ? undefined : model.startBlocker;
  const projectName = model.project?.name ?? "this project";
  return (
    <div className="start-footer">
      <LaunchNotices model={model} />
      <div className="start-footer-row">
        <p className="start-summary">
          <strong>
            {selection.presentation.label} on {projectName}.
          </strong>{" "}
          <span className="start-quiet">
            {timeSentence(model.deadlineSeconds)}
          </span>
        </p>
        <div className="start-actions">
          <button
            type="button"
            className="ui-btn"
            disabled={pending || model.saveBlocker !== undefined}
            aria-describedby={`${id}-draft`}
            onClick={model.saveDraft}
          >
            {launch.pending === "draft" ? "Saving…" : "Save as draft"}
          </button>
          <button
            ref={startButton}
            type="button"
            className="ui-btn start-primary"
            data-variant="primary"
            disabled={pending || model.startBlocker !== undefined}
            aria-describedby={hint === undefined ? undefined : `${id}-hint`}
            aria-keyshortcuts="Control+Enter Meta+Enter"
            onClick={model.start}
          >
            <PlayIcon />
            <span>
              {launch.pending === "start" ? "Starting…" : "Start check"}
            </span>
            <span className="ui-kbd-hint" aria-hidden="true">
              <Kbd>{modKeyLabel()}</Kbd>
              <Kbd>Enter</Kbd>
            </span>
          </button>
        </div>
      </div>
      <span id={`${id}-draft`} className="ui-visually-hidden">
        Creates the check without starting it. Start it later from its page.
      </span>
      {hint === undefined ? null : (
        <p id={`${id}-hint`} className="start-hint">
          {hint}
        </p>
      )}
      <p className="ui-visually-hidden" role="status">
        {status}
      </p>
    </div>
  );
}
