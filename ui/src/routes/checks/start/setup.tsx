import { useId, type KeyboardEvent } from "react";
import { Link, useNavigate } from "react-router";

import { ErrorNotice } from "../../../app/error-notice";
import { IdChip, Kbd, modKeyLabel } from "../../../ui";
import { AdvancedOptions } from "./advanced";
import {
  compatibilityReasonText,
  missingLabel,
  presentCheckType,
  scopeSummary,
} from "./check-types";
import { TextAreaField, TextField } from "./fields";
import { BulbIcon, ChevronIcon, PlayIcon } from "./icons";
import { checkPath, isLostResponse } from "./launch";
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

  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (
      event.key === "Enter" &&
      (event.ctrlKey || event.metaKey) &&
      !event.nativeEvent.isComposing
    ) {
      event.preventDefault();
      if (
        model.startBlocker === undefined &&
        model.launch.pending === undefined
      )
        model.start();
    }
  }

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
        aria-keyshortcuts="Control+Enter Meta+Enter"
        onChange={(event) => model.setObjective(event.target.value)}
        onKeyDown={onKeyDown}
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
      {readiness?.state === "unavailable" ? (
        <div className="notice notice-error" role="note">
          <strong>This check type can&apos;t run on this server.</strong>
          <p>
            {readiness.reasons.length === 0
              ? "The server gives no reason."
              : `The server does not support ${readiness.reasons
                  .map(compatibilityReasonText)
                  .join(", ")}.`}
          </p>
        </div>
      ) : null}
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
        const unsupported = !family.preferred.serverCompatible;
        return (
          <label key={family.name} className="start-scope-option">
            <input
              type="radio"
              name={`${id}-scope`}
              checked={family.name === selection.family.name}
              disabled={unsupported}
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

function LaunchNotices({ model }: { model: StartCheck }) {
  const { launch, request } = model;
  const draft = launch.draft;
  if (launch.pending !== undefined) return null;
  const createError = launch.createError;
  const startError = launch.startError;
  // The form no longer matches the draft this page created.
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
  if (startError !== null && draft !== undefined) {
    const open = checkPath(model.projectId, draft.auditId);
    if (isLostResponse(startError))
      return (
        <div className="notice notice-warning" role="alert">
          <strong>The check may already have started.</strong>
          <p>
            The answer to the start request did not arrive. Retrying sends the
            same request; it cannot start the check twice.{" "}
            <Link to={open}>Open the check</Link>
          </p>
          <button
            type="button"
            className="ui-btn"
            data-size="sm"
            onClick={launch.retryStart}
          >
            Retry same request
          </button>
        </div>
      );
    return (
      <>
        <ErrorNotice
          error={startError}
          reconcileWrite
          context="The check was saved as a draft but did not start."
        />
        <p className="start-tight">
          <Link to={open}>Open the draft</Link> to see it.{" "}
          {changed
            ? "The form changed, so starting now creates a new check; the draft stays."
            : "Start check tries this draft again."}
        </p>
      </>
    );
  }
  if (draft !== undefined && changed)
    return (
      <p className="start-tight start-quiet">
        Your earlier attempt is saved as a draft (
        <Link to={checkPath(model.projectId, draft.auditId)}>open it</Link>).
        Starting now creates a new check.
      </p>
    );
  return null;
}

/** The pinned bar: a plain summary, Save as draft and Start check. */
export function SetupFooter({ model }: { model: StartCheck }) {
  const id = useId();
  const { launch, selection } = model;
  if (selection === undefined) return null;
  const pending = launch.pending !== undefined;
  const status =
    launch.pending === undefined
      ? ""
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
            disabled={pending || model.draftBlocker !== undefined}
            aria-describedby={`${id}-draft`}
            onClick={model.saveDraft}
          >
            {launch.pending === "draft" ? "Saving…" : "Save as draft"}
          </button>
          <button
            type="button"
            className="ui-btn start-primary"
            data-variant="primary"
            disabled={pending || model.startBlocker !== undefined}
            aria-describedby={hint === undefined ? undefined : `${id}-hint`}
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
