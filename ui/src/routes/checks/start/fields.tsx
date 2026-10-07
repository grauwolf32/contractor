import { useId, type ReactNode } from "react";

/**
 * The label row of a field. "Required" sits next to the label, outside it,
 * so the field's name stays the label alone; aria-required says it.
 */
function FieldLabel({
  id,
  label,
  required,
}: {
  id: string;
  label: string;
  required: boolean;
}) {
  return (
    <div className="start-field-label">
      <label htmlFor={id}>{label}</label>
      {required ? (
        <span className="start-required" aria-hidden="true">
          Required
        </span>
      ) : null}
    </div>
  );
}

/** A labelled one-line text field with an optional hint and error. */
export function TextField({
  label,
  value,
  onChange,
  placeholder,
  hint,
  error,
  required = false,
  maxLength,
  autoComplete = "off",
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string | undefined;
  hint?: ReactNode;
  error?: string | undefined;
  required?: boolean | undefined;
  maxLength?: number | undefined;
  autoComplete?: string | undefined;
}) {
  const id = useId();
  const described = [
    hint === undefined ? undefined : `${id}-hint`,
    error === undefined ? undefined : `${id}-error`,
  ].filter((part) => part !== undefined);
  return (
    <div className="start-field">
      <FieldLabel id={id} label={label} required={required} />
      <input
        id={id}
        className="start-input"
        value={value}
        placeholder={placeholder}
        maxLength={maxLength}
        autoComplete={autoComplete}
        aria-required={required || undefined}
        aria-invalid={error === undefined ? undefined : true}
        aria-describedby={
          described.length === 0 ? undefined : described.join(" ")
        }
        onChange={(event) => onChange(event.target.value)}
      />
      {hint === undefined ? null : (
        <p id={`${id}-hint`} className="start-field-hint">
          {hint}
        </p>
      )}
      {error === undefined ? null : (
        <p id={`${id}-error`} className="start-field-error">
          {error}
        </p>
      )}
    </div>
  );
}

/** A labelled multi-line text field. */
export function TextAreaField({
  label,
  value,
  onChange,
  placeholder,
  hint,
  required = false,
  maxLength,
  rows = 2,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string | undefined;
  hint?: ReactNode;
  required?: boolean | undefined;
  maxLength?: number | undefined;
  rows?: number | undefined;
}) {
  const id = useId();
  return (
    <div className="start-field">
      <FieldLabel id={id} label={label} required={required} />
      <textarea
        id={id}
        className="start-input start-textarea"
        rows={rows}
        value={value}
        placeholder={placeholder}
        maxLength={maxLength}
        aria-required={required || undefined}
        aria-describedby={hint === undefined ? undefined : `${id}-hint`}
        onChange={(event) => onChange(event.target.value)}
      />
      {hint === undefined ? null : (
        <p id={`${id}-hint`} className="start-field-hint">
          {hint}
        </p>
      )}
    </div>
  );
}
