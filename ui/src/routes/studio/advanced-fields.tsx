import { at, textValue, type Path } from "./document";
import type { FormEditor } from "./form-editor";

export function DraftField({
  editor,
  path,
  label,
  number = false,
  placeholder,
}: {
  editor: FormEditor;
  path: Path;
  label: string;
  number?: boolean;
  placeholder?: string;
}) {
  const current = textValue(at(editor.draft.value, path));
  return (
    <label className="studio-field">
      <span>{label}</span>
      <input
        key={current}
        defaultValue={current}
        type={number ? "number" : "text"}
        step={number ? "any" : undefined}
        placeholder={placeholder}
        onBlur={(event) => {
          const next = event.target.value;
          if (next !== current)
            editor.onPatch(
              path,
              number ? (next === "" ? null : Number(next)) : next,
            );
        }}
      />
    </label>
  );
}
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
    <label className="studio-field">
      <span>{label}</span>
      <select
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
    </label>
  );
}
