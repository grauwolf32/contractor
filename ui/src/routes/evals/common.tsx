import {
  cloneElement,
  isValidElement,
  useContext,
  useEffect,
  useId,
  useRef,
  useState,
  type ComponentProps,
  type ReactNode,
} from "react";
import { Link } from "react-router";
import { PublicAPIError } from "../../api/error";
import { ErrorNotice } from "../../app/error-notice";
import { KeyValueValidityContext } from "./key-value-validity";
import "./evals.css";

export function EvalFrame({
  title,
  children,
  action,
  description,
}: {
  title: string;
  children: ReactNode;
  action?: ReactNode;
  description?: string;
}) {
  return (
    <section className="route-page eval-page">
      <header className="route-header-row">
        <div>
          <p className="eyebrow">
            <Link to="/evals">Evals</Link>
          </p>
          <h2>{title}</h2>
          {description ? <p className="lede">{description}</p> : null}
        </div>
        {action}
      </header>
      {children}
    </section>
  );
}

export function CommaSeparatedInput({
  value,
  onChange,
  ...props
}: Omit<ComponentProps<"input">, "value" | "onChange"> & {
  value: string[];
  onChange: (values: string[]) => void;
}) {
  const normalized = value.join(", ");
  const [buffer, setBuffer] = useState({
    source: normalized,
    text: normalized,
  });
  if (buffer.source !== normalized) {
    setBuffer({ source: normalized, text: normalized });
  }
  return (
    <input
      {...props}
      value={buffer.text}
      onChange={(event) => {
        const text = event.target.value;
        const values = text
          .split(",")
          .map((item) => item.trim())
          .filter(Boolean);
        setBuffer({ source: values.join(", "), text });
        onChange(values);
      }}
    />
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
          {changed ? "Reload current revision" : "Try again"}
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

type KeyValueRow = { id: number; name: string; value: string };

function sameKeyValues(
  left: Record<string, string>,
  right: Record<string, string>,
): boolean {
  const keys = Object.keys(left);
  return (
    keys.length === Object.keys(right).length &&
    keys.every((key) => Object.hasOwn(right, key) && left[key] === right[key])
  );
}

function rowProblem(rows: KeyValueRow[], index: number): string | null {
  const name = rows[index]!.name;
  if (!name.trim()) return "Enter a name before saving.";
  if (rows.some((row, other) => other !== index && row.name === name)) {
    return "This name is already in use. Choose a different name.";
  }
  return null;
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
  const editorId = useId();
  const register = useContext(KeyValueValidityContext);
  const nextRowId = useRef(Object.keys(value).length);
  const published = useRef(value);
  const [rows, setRows] = useState<KeyValueRow[]>(() =>
    Object.entries(value).map(([name, rowValue], id) => ({
      id,
      name,
      value: rowValue,
    })),
  );
  const invalid = rows.some((_, index) => rowProblem(rows, index) !== null);

  useEffect(() => {
    if (!sameKeyValues(value, published.current)) {
      published.current = value;
      setRows(
        Object.entries(value).map(([name, rowValue]) => ({
          id: nextRowId.current++,
          name,
          value: rowValue,
        })),
      );
    }
  }, [value]);

  useEffect(() => {
    register?.(editorId, invalid);
  }, [editorId, invalid, register]);
  useEffect(
    () => () => {
      register?.(editorId, false);
    },
    [editorId, register],
  );

  function update(next: KeyValueRow[]) {
    setRows(next);
    if (next.some((_, index) => rowProblem(next, index) !== null)) return;
    const nextValue = Object.fromEntries(
      next.map((row) => [row.name, row.value]),
    );
    published.current = nextValue;
    onChange(nextValue);
  }
  return (
    <fieldset className="eval-key-values">
      <legend>{label}</legend>
      {rows.map((row, index) => (
        <div className="eval-key-value" key={row.id}>
          <input
            aria-label={`${label} name ${index + 1}`}
            aria-invalid={rowProblem(rows, index) !== null}
            aria-describedby={
              rowProblem(rows, index)
                ? `${editorId}-${row.id}-error`
                : undefined
            }
            value={row.name}
            onChange={(event) =>
              update(
                rows.map((item) =>
                  item.id === row.id
                    ? { ...item, name: event.target.value }
                    : item,
                ),
              )
            }
          />
          <input
            aria-label={`${label} value ${index + 1}`}
            value={row.value}
            onChange={(event) =>
              update(
                rows.map((item) =>
                  item.id === row.id
                    ? { ...item, value: event.target.value }
                    : item,
                ),
              )
            }
          />
          <button
            type="button"
            className="secondary-button"
            aria-label={`Remove ${label} ${index + 1}`}
            onClick={() => update(rows.filter((item) => item.id !== row.id))}
          >
            Remove
          </button>
          {rowProblem(rows, index) ? (
            <small id={`${editorId}-${row.id}-error`} role="alert">
              {rowProblem(rows, index)}
            </small>
          ) : null}
        </div>
      ))}
      <button
        type="button"
        className="secondary-button"
        onClick={() => {
          let number = 1;
          while (rows.some((row) => row.name === `field-${number}`)) number++;
          update([
            ...rows,
            { id: nextRowId.current++, name: `field-${number}`, value: "" },
          ]);
        }}
      >
        Add {label.toLowerCase()}
      </button>
    </fieldset>
  );
}
