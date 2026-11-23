import { Field } from "../../ui";
import { at, textValue, type Path } from "./document";
import type { FormEditor } from "./form-editor";

/**
 * A draft scalar in a text, number or multi-line control. The draft changes
 * when the control loses focus, and only when the value changed.
 */
export function DraftField({
  editor,
  path,
  label,
  number = false,
  multiline = false,
  placeholder,
}: {
  editor: FormEditor;
  path: Path;
  label: string;
  number?: boolean;
  multiline?: boolean;
  placeholder?: string;
}) {
  const current = textValue(at(editor.draft.value, path));
  const commit = (next: string) => {
    if (next !== current)
      editor.onPatch(path, number ? (next === "" ? null : Number(next)) : next);
  };
  return (
    <Field label={label} className="studio-field">
      {(control) =>
        multiline ? (
          <textarea
            {...control}
            key={current}
            defaultValue={current}
            rows={4}
            onBlur={(event) => commit(event.target.value)}
          />
        ) : (
          <input
            {...control}
            key={current}
            defaultValue={current}
            type={number ? "number" : "text"}
            step={number ? "any" : undefined}
            placeholder={placeholder}
            onBlur={(event) => commit(event.target.value)}
          />
        )
      }
    </Field>
  );
}

/** A draft list of strings typed comma separated, such as media types. */
export function DraftList({
  editor,
  path,
  label,
}: {
  editor: FormEditor;
  path: Path;
  label: string;
}) {
  const current = at(editor.draft.value, path);
  return (
    <Field label={label} className="studio-field">
      {(control) => (
        <input
          {...control}
          key={JSON.stringify(current)}
          defaultValue={Array.isArray(current) ? current.join(", ") : ""}
          onBlur={(event) => {
            const next = event.target.value
              .split(",")
              .map((item) => item.trim())
              .filter(Boolean);
            if (JSON.stringify(next) !== JSON.stringify(current))
              editor.onPatch(path, next);
          }}
        />
      )}
    </Field>
  );
}

/** A draft value chosen from `options`; an unknown value stays selectable. */
export function DraftChoice({
  editor,
  path,
  label,
  options,
  fallback = "",
}: {
  editor: FormEditor;
  path: Path;
  label: string;
  options: string[];
  fallback?: string;
}) {
  const value = at(editor.draft.value, path);
  const current = value === undefined ? fallback : textValue(value);
  return (
    <Field label={label} className="studio-field">
      {(control) => (
        <select
          {...control}
          value={current}
          onChange={(event) => editor.onPatch(path, event.target.value)}
        >
          {!options.includes(current) ? (
            <option value={current}>{current || "Choose…"}</option>
          ) : null}
          {options.map((option) => (
            <option key={option}>{option}</option>
          ))}
        </select>
      )}
    </Field>
  );
}
