import type { ReactNode } from "react";

import { Field } from "../../../ui";

interface TextProps {
  label: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string | undefined;
  hint?: ReactNode;
  required?: boolean | undefined;
  maxLength?: number | undefined;
}

/** A one-line text field of the Start form, with an optional error. */
export function TextField({
  label,
  value,
  onChange,
  placeholder,
  hint,
  error,
  required,
  maxLength,
}: TextProps & { error?: string | undefined }) {
  return (
    <Field label={label} hint={hint} error={error} required={required}>
      {(control) => (
        <input
          {...control}
          value={value}
          placeholder={placeholder}
          maxLength={maxLength}
          autoComplete="off"
          onChange={(event) => onChange(event.target.value)}
        />
      )}
    </Field>
  );
}

/** A two-line text field of the Start form. */
export function TextAreaField({
  label,
  value,
  onChange,
  placeholder,
  hint,
  required,
  maxLength,
}: TextProps) {
  return (
    <Field label={label} hint={hint} required={required}>
      {(control) => (
        <textarea
          {...control}
          className="start-textarea"
          rows={2}
          value={value}
          placeholder={placeholder}
          maxLength={maxLength}
          onChange={(event) => onChange(event.target.value)}
        />
      )}
    </Field>
  );
}
