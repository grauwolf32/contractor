import { useId, type ReactNode } from "react";

/** The attributes Field hands to its control. */
interface FieldControl {
  id: string;
  "aria-describedby"?: string;
  "aria-invalid"?: true;
  "aria-required"?: true;
  required?: true;
}

/**
 * A form field: label, control, hint and error, wired for assistive
 * technology. The control is rendered by `children`, which receives the
 * id, aria-describedby (hint and error), aria-invalid and required
 * attributes to spread on it:
 *
 *   <Field label="Name" hint="Shown in lists." error={nameError} required>
 *     {(control) => <input {...control} value={name} onChange={…} />}
 *   </Field>
 *
 * `required` marks the label with an asterisk (hidden from screen readers,
 * which hear the control's required state); `optional` adds "(optional)"
 * instead. The error replaces nothing: it shows under the hint as an alert.
 * Inputs, selects and textareas inside a Field get the shared form-control
 * look (src/app/ui-primitives.css).
 */
export function Field({
  label,
  hint,
  error,
  required = false,
  optional = false,
  id,
  className,
  children,
}: {
  label: ReactNode;
  hint?: ReactNode;
  /** A message for the current value; absent or null when it is valid. */
  error?: ReactNode;
  required?: boolean | undefined;
  optional?: boolean | undefined;
  /** The control's id, when the page needs to know it. */
  id?: string | undefined;
  className?: string | undefined;
  children: (control: FieldControl) => ReactNode;
}) {
  const generated = useId();
  const controlId = id ?? generated;
  const hintId = `${controlId}-hint`;
  const errorId = `${controlId}-error`;
  const invalid = error !== undefined && error !== null && error !== false;
  const describedBy = [
    hint === undefined ? undefined : hintId,
    invalid ? errorId : undefined,
  ]
    .filter((part) => part !== undefined)
    .join(" ");
  return (
    <div
      className={className === undefined ? "ui-field" : `ui-field ${className}`}
    >
      <label className="ui-field-label" htmlFor={controlId}>
        {label}
        {required ? (
          <span className="ui-field-required" aria-hidden="true">
            {" *"}
          </span>
        ) : optional ? (
          <>
            {" "}
            <span className="ui-field-optional">(optional)</span>
          </>
        ) : null}
      </label>
      {children({
        id: controlId,
        ...(describedBy === "" ? {} : { "aria-describedby": describedBy }),
        ...(invalid ? { "aria-invalid": true } : {}),
        ...(required ? { required: true, "aria-required": true } : {}),
      })}
      {hint === undefined ? null : (
        <p id={hintId} className="ui-field-hint">
          {hint}
        </p>
      )}
      {invalid ? (
        <p id={errorId} className="ui-field-error" role="alert">
          {error}
        </p>
      ) : null}
    </div>
  );
}
