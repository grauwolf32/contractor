import {
  useId,
  useRef,
  type KeyboardEvent as ReactKeyboardEvent,
  type ReactNode,
} from "react";

import { Kbd } from "./chips";
import { modKeyLabel, useShortcuts, type ShortcutHandler } from "./shortcuts";

export interface DecisionOption {
  id: string;
  label: string;
  /** One key that chooses this option, e.g. "c". Shown as a key hint. */
  shortcut?: string | undefined;
  /** "primary" is filled, "danger" uses the danger colour. Default "secondary". */
  tone?: "primary" | "secondary" | "danger" | undefined;
  /** Decorative icon before the label. */
  icon?: ReactNode;
}

export interface DecisionSeverityOption {
  value: string;
  label: string;
}

export interface DecisionSeverity {
  options: readonly DecisionSeverityOption[];
  value?: string | undefined;
  onChange: (value: string) => void;
  /** Recording stays disabled until a severity is chosen. */
  required: boolean;
  /** Default "Severity". */
  label?: string | undefined;
  /** Short note after the options, e.g. "Pick one when you confirm." */
  hint?: ReactNode;
}

export interface DecisionRationale {
  value: string;
  onChange: (value: string) => void;
  /** Default "Why". */
  label?: string | undefined;
  placeholder?: string | undefined;
  maxLength?: number | undefined;
  /** Standing help for the reason, shown above it and linked to the field. */
  hint?: ReactNode;
}

export interface DecisionNext {
  label: string;
  /** One key, e.g. "j". Shown as a key hint. */
  shortcut?: string | undefined;
  onNext: () => void;
}

export interface DecisionBarProps {
  /** The verdicts, in display order. */
  options: readonly DecisionOption[];
  /** Id of the chosen option. */
  selected?: string | undefined;
  onSelect: (id: string) => void;
  /** Severity choice, e.g. while "Confirm issue" is chosen. */
  severity?: DecisionSeverity | undefined;
  /** The required reason. Recording needs non-blank text. */
  rationale: DecisionRationale;
  /** Called only when the decision is complete. */
  onSubmit: () => void;
  /** Default "Record decision". */
  submitLabel?: string | undefined;
  /** While recording: every control is disabled and the reason is read-only. */
  pending?: boolean | undefined;
  /** Shown under the bar as an alert. */
  error?: ReactNode;
  /** Why the user cannot decide here; disables deciding but not Next. */
  disabledReason?: string | undefined;
  /** Slot after the verdicts, e.g. a More menu with Duplicate and Reopen. */
  more?: ReactNode;
  /** Moves on to the next item. */
  next?: DecisionNext | undefined;
  /** Full-width slot above the reason, e.g. a duplicate target picker. */
  extra?: ReactNode;
  /** Default "Your decision". */
  "aria-label"?: string | undefined;
}

function isShown(node: ReactNode): boolean {
  return node !== undefined && node !== null && node !== false;
}

/**
 * Decision bar for the bottom of a detail pane: optional severity, verdict
 * buttons with key hints, a required reason and the record button.
 *
 * Choosing a verdict (click or its key) selects it and focuses the reason.
 * Ctrl+Enter or ⌘+Enter in the reason records the decision. Recording is
 * disabled until a verdict is chosen, the reason is not blank and a
 * required severity is set; a line under the bar says what is missing.
 */
export function DecisionBar({
  options,
  selected,
  onSelect,
  severity,
  rationale,
  onSubmit,
  submitLabel = "Record decision",
  pending = false,
  error,
  disabledReason,
  more,
  next,
  extra,
  "aria-label": ariaLabel = "Your decision",
}: DecisionBarProps) {
  const id = useId();
  const reasonField = useRef<HTMLTextAreaElement>(null);
  const locked = pending || disabledReason !== undefined;
  const chosen = options.find((option) => option.id === selected);
  const severityLabel = severity?.label ?? "Severity";
  const severityMissing =
    severity !== undefined &&
    severity.required &&
    (severity.value === undefined || severity.value === "");
  const reasonMissing = rationale.value.trim() === "";
  const canSubmit =
    !locked && chosen !== undefined && !severityMissing && !reasonMissing;
  const hint =
    disabledReason ??
    (pending
      ? "Recording…"
      : chosen === undefined
        ? "Choose a decision."
        : severityMissing
          ? `Choose ${severityLabel.toLowerCase()}.`
          : reasonMissing
            ? "Write a short reason."
            : undefined);
  const hintId = hint === undefined ? undefined : `${id}-hint`;

  function choose(optionId: string) {
    onSelect(optionId);
    reasonField.current?.focus();
  }

  function submit() {
    if (canSubmit) onSubmit();
  }

  function onReasonKeyDown(event: ReactKeyboardEvent<HTMLTextAreaElement>) {
    if (
      event.key === "Enter" &&
      (event.ctrlKey || event.metaKey) &&
      !event.nativeEvent.isComposing
    ) {
      event.preventDefault();
      submit();
    }
  }

  const choosable: readonly DecisionOption[] = locked ? [] : options;
  const bindings: Record<string, ShortcutHandler> = Object.fromEntries([
    ...choosable.flatMap((option) =>
      option.shortcut === undefined
        ? []
        : [[option.shortcut.toLowerCase(), () => choose(option.id)] as const],
    ),
    ...(next?.shortcut === undefined
      ? []
      : [[next.shortcut.toLowerCase(), () => next.onNext()] as const]),
  ]);
  useShortcuts(bindings, { enabled: !pending });

  return (
    <section
      className="ui-decision"
      aria-label={ariaLabel}
      aria-busy={pending || undefined}
    >
      {severity === undefined ? null : (
        <div className="ui-decision-severity">
          <span id={`${id}-severity`} className="ui-decision-label">
            {severityLabel}
          </span>
          <div
            role="radiogroup"
            aria-labelledby={`${id}-severity`}
            aria-required={severity.required || undefined}
            className="ui-decision-severity-options"
          >
            {severity.options.map((option) => (
              <label key={option.value} className="ui-decision-severity-option">
                <input
                  type="radio"
                  name={`${id}-severity-value`}
                  value={option.value}
                  checked={severity.value === option.value}
                  disabled={locked}
                  onChange={() => severity.onChange(option.value)}
                />
                <span>{option.label}</span>
              </label>
            ))}
          </div>
          {isShown(severity.hint) ? (
            <span className="ui-decision-note">{severity.hint}</span>
          ) : null}
        </div>
      )}
      <div className="ui-decision-actions">
        {options.map((option) => (
          <button
            key={option.id}
            type="button"
            className="ui-btn"
            data-variant={option.tone ?? "secondary"}
            aria-pressed={option.id === selected}
            aria-keyshortcuts={option.shortcut?.toUpperCase()}
            disabled={locked}
            onClick={() => choose(option.id)}
          >
            {option.icon}
            <span>{option.label}</span>
            {option.shortcut === undefined ? null : (
              <span className="ui-kbd-hint" aria-hidden="true">
                <Kbd>{option.shortcut.toUpperCase()}</Kbd>
              </span>
            )}
          </button>
        ))}
        {isShown(more) ? (
          <div className="ui-decision-more" inert={pending}>
            {more}
          </div>
        ) : null}
        <span className="ui-decision-spacer" />
        {next === undefined ? null : (
          <button
            type="button"
            className="ui-btn"
            data-variant="ghost"
            aria-keyshortcuts={next.shortcut?.toUpperCase()}
            disabled={pending}
            onClick={next.onNext}
          >
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.8"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
              focusable="false"
            >
              <path d="M12 5v14M6.5 13.5L12 19l5.5-5.5" />
            </svg>
            <span>{next.label}</span>
            {next.shortcut === undefined ? null : (
              <span className="ui-kbd-hint" aria-hidden="true">
                <Kbd>{next.shortcut.toUpperCase()}</Kbd>
              </span>
            )}
          </button>
        )}
      </div>
      {isShown(extra) ? (
        <div className="ui-decision-extra" inert={pending}>
          {extra}
        </div>
      ) : null}
      {isShown(rationale.hint) ? (
        <p id={`${id}-reason-hint`} className="ui-decision-note">
          {rationale.hint}
        </p>
      ) : null}
      <div className="ui-decision-compose">
        <label htmlFor={`${id}-reason`} className="ui-decision-label">
          {rationale.label ?? "Why"}
        </label>
        <textarea
          ref={reasonField}
          id={`${id}-reason`}
          className="ui-decision-field"
          rows={2}
          value={rationale.value}
          placeholder={rationale.placeholder}
          maxLength={rationale.maxLength}
          aria-required="true"
          aria-describedby={
            [isShown(rationale.hint) ? `${id}-reason-hint` : undefined, hintId]
              .filter((part) => part !== undefined)
              .join(" ") || undefined
          }
          aria-keyshortcuts="Control+Enter Meta+Enter"
          readOnly={pending}
          disabled={disabledReason !== undefined}
          onChange={(event) => rationale.onChange(event.target.value)}
          onKeyDown={onReasonKeyDown}
        />
        <button
          type="button"
          className="ui-btn"
          data-variant="primary"
          disabled={!canSubmit}
          aria-describedby={hintId}
          onClick={submit}
        >
          <span>{submitLabel}</span>
          <span className="ui-kbd-hint" aria-hidden="true">
            <Kbd>{modKeyLabel()}</Kbd>
            <Kbd>Enter</Kbd>
          </span>
        </button>
      </div>
      <p id={`${id}-hint`} className="ui-decision-hint">
        {hint}
      </p>
      {isShown(error) ? (
        <p role="alert" className="ui-decision-error">
          {error}
        </p>
      ) : null}
    </section>
  );
}
