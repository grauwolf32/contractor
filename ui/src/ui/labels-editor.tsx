import { useEffect, useRef } from "react";
import { Button } from "./button";
import { Field } from "./field";

/** An editable ordinary key/value label; validation belongs to its caller. */
export interface LabelEntry {
  readonly key: string;
  readonly value: string;
}

/** Bounded label chips with keyboard-accessible editing and quick additions. */
export function LabelsEditor({
  entries,
  onChange,
  maximum,
  error,
  hint,
  suggestions = [],
}: {
  entries: readonly LabelEntry[];
  onChange: (entries: LabelEntry[]) => void;
  maximum: number;
  error?: string | undefined;
  hint: string;
  suggestions?: readonly string[] | undefined;
}) {
  const keyInputs = useRef(new Map<number, HTMLInputElement>());
  const focusNext = useRef<number | undefined>(undefined);
  useEffect(() => {
    if (focusNext.current !== undefined) {
      keyInputs.current.get(focusNext.current)?.focus();
      focusNext.current = undefined;
    }
  }, [entries]);
  function add(key = "") {
    focusNext.current = entries.length;
    onChange([...entries, { key, value: "" }]);
  }
  function update(index: number, field: "key" | "value", value: string) {
    onChange(
      entries.map((entry, item) =>
        item === index ? { ...entry, [field]: value } : entry,
      ),
    );
  }
  return (
    <Field label="Labels" hint={hint} error={error}>
      {(control) => (
        <div
          {...control}
          role="group"
          aria-label="Labels"
          className="ui-labels-editor"
        >
          <div className="ui-labels-toolbar">
            <span className="ui-caption">
              {entries.length} / {maximum}
            </span>
            {suggestions.map((key) => (
              <Button
                key={key}
                size="xs"
                disabled={
                  entries.length >= maximum ||
                  entries.some((entry) => entry.key === key)
                }
                onClick={() => add(key)}
              >
                + {key}
              </Button>
            ))}
            <Button
              size="xs"
              disabled={entries.length >= maximum}
              onClick={() => add()}
            >
              + Add label
            </Button>
          </div>
          {entries.length === 0 ? (
            <p className="ui-labels-empty">
              No labels. Add one to choose cloud or local, or to organise this
              check.
            </p>
          ) : (
            <div className="ui-labels-list">
              {entries.map((entry, index) => (
                <div className="ui-labels-row" key={index}>
                  <input
                    ref={(element) => {
                      if (element === null) keyInputs.current.delete(index);
                      else keyInputs.current.set(index, element);
                    }}
                    aria-label={`Label key ${index + 1}`}
                    aria-describedby={control["aria-describedby"]}
                    autoComplete="off"
                    spellCheck={false}
                    placeholder="Label"
                    value={entry.key}
                    onChange={(event) =>
                      update(index, "key", event.target.value)
                    }
                  />
                  <span aria-hidden="true">:</span>
                  <input
                    aria-label={`Label value ${index + 1}`}
                    aria-describedby={control["aria-describedby"]}
                    autoComplete="off"
                    spellCheck={false}
                    placeholder={
                      entry.key === "cloud"
                        ? "Any provider"
                        : entry.key === "local"
                          ? "Any site"
                          : "Optional value"
                    }
                    value={entry.value}
                    onChange={(event) =>
                      update(index, "value", event.target.value)
                    }
                  />
                  <Button
                    size="xs"
                    aria-label={`Remove label ${index + 1}`}
                    onClick={() => {
                      focusNext.current = Math.max(0, index - 1);
                      onChange(entries.filter((_, item) => item !== index));
                    }}
                  >
                    ×
                  </Button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </Field>
  );
}
