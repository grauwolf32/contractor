import {
  cloneElement,
  isValidElement,
  useId,
  useState,
  type ReactNode,
} from "react";
import { Link } from "react-router";
import { PublicAPIError } from "../../api/error";
import { ErrorNotice } from "../artifacts/common";
import "./evals.css";

export function EvalFrame({
  title,
  children,
  action,
}: {
  title: string;
  children: ReactNode;
  action?: ReactNode;
}) {
  return (
    <section className="workspace-page eval-page">
      <header className="page-heading">
        <div>
          <p className="eyebrow">
            <Link to="/evals">Evals</Link>
          </p>
          <h1>{title}</h1>
        </div>
        {action}
      </header>
      {children}
    </section>
  );
}

export function EvalError({
  error,
  reload,
}: {
  error: unknown;
  reload?: (() => void) | undefined;
}) {
  if (!error) return null;
  const code = error instanceof PublicAPIError ? error.code : "";
  const changed = [
    "eval_revision_mismatch",
    "eval_view_changed",
    "eval_member_conflict",
  ].includes(code);
  return (
    <div role="alert">
      <ErrorNotice error={error} />
      {changed ? (
        <p>
          The saved evidence or draft changed. Reload it before continuing; your
          previous action has not been applied to the new revision.
        </p>
      ) : null}
      {reload ? (
        <button type="button" className="secondary-button" onClick={reload}>
          {changed ? "Reload current revision" : "Retry"}
        </button>
      ) : null}
    </div>
  );
}

export function EvalField({
  label,
  children,
  hint,
}: {
  label: string;
  children: ReactNode;
  hint?: string | undefined;
}) {
  const id = useId();
  return (
    <div className="eval-field">
      <label htmlFor={id}>{label}</label>
      {isValidElement<{ id?: string; "aria-describedby"?: string }>(children)
        ? cloneElement(children, {
            id,
            ...(hint ? { "aria-describedby": `${id}-hint` } : {}),
          })
        : children}
      {hint ? <small id={`${id}-hint`}>{hint}</small> : null}
    </div>
  );
}

export function EvalPages({
  previous,
  next,
}: {
  previous?: (() => void) | undefined;
  next?: (() => void) | undefined;
}) {
  return (
    <nav className="eval-actions" aria-label="Pagination">
      <button
        type="button"
        className="secondary-button"
        disabled={!previous}
        onClick={previous}
      >
        Previous page
      </button>
      <button
        type="button"
        className="secondary-button"
        disabled={!next}
        onClick={next}
      >
        Next page
      </button>
    </nav>
  );
}

export function KeyValueEditor({
  label,
  value,
  onChange,
}: {
  label: string;
  value: Record<string, string>;
  onChange: (value: Record<string, string>) => void;
}) {
  const entries = Object.entries(value);
  const [error, setError] = useState<string | null>(null);
  function rename(index: number, name: string) {
    if (entries.some(([key], i) => i !== index && key === name)) {
      setError("This name is already in use. Choose a different name.");
      return;
    }
    setError(null);
    onChange(
      Object.fromEntries(
        entries.map((entry, i) => (i === index ? [name, entry[1]] : entry)),
      ),
    );
  }
  return (
    <fieldset className="eval-key-values">
      <legend>{label}</legend>
      {error ? <p role="alert">{error}</p> : null}
      {entries.map(([key, val], index) => (
        <div className="eval-key-value" key={index}>
          <input
            aria-label={`${label} name ${index + 1}`}
            value={key}
            onChange={(e) => rename(index, e.target.value)}
          />
          <input
            aria-label={`${label} value ${index + 1}`}
            value={val}
            onChange={(e) => onChange({ ...value, [key]: e.target.value })}
          />
          <button
            type="button"
            className="secondary-button"
            aria-label={`Remove ${label} ${index + 1}`}
            onClick={() =>
              onChange(
                Object.fromEntries(entries.filter((_, i) => i !== index)),
              )
            }
          >
            Remove
          </button>
        </div>
      ))}
      <button
        type="button"
        className="secondary-button"
        onClick={() =>
          onChange({
            ...value,
            [entries.some(([k]) => k === "")
              ? `field-${entries.length + 1}`
              : ""]: "",
          })
        }
      >
        Add {label.toLowerCase()}
      </button>
    </fieldset>
  );
}
