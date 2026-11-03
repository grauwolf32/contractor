import { useId, useLayoutEffect, useRef } from "react";
import { Icon } from "../../../app/icon";
import { Button } from "../../../ui";
import { CHECKLIST_TEXT_BYTES } from "./checklist-document";
import type { ChecklistDraft } from "./checklist-state";

export function ChecklistEditor({
  draft,
  busy,
}: {
  draft: ChecklistDraft;
  busy: boolean;
}) {
  const id = useId();
  const entries = useRef<HTMLOListElement>(null);
  const addButton = useRef<HTMLButtonElement>(null);
  const focusAfterChange = useRef<number | undefined>(undefined);
  const count = draft.entries.length;
  const ready = !draft.dirty && draft.error === undefined;
  useLayoutEffect(() => {
    entries.current?.querySelector("textarea")?.focus();
  }, []);
  useLayoutEffect(() => {
    const index = focusAfterChange.current;
    if (index === undefined) return;
    focusAfterChange.current = undefined;
    const fields = entries.current?.querySelectorAll("textarea");
    (fields?.[Math.min(index, count - 1)] ?? addButton.current)?.focus();
  }, [count]);
  const feedback = draft.empty
    ? "Add at least one item to define your checklist."
    : (draft.error ??
      (ready
        ? `${count} ${count === 1 ? "item" : "items"} · ready to use`
        : `${count} ${count === 1 ? "item" : "items"} · unapplied changes`));
  const tone =
    draft.empty || draft.error?.includes("Write what")
      ? "draft"
      : draft.error !== undefined
        ? "error"
        : ready
          ? "ready"
          : "draft";
  return (
    <div className="start-checklist-editor">
      <header className="start-checklist-head">
        <h4>
          <Icon name="docs" size={16} />
          Your checklist{" "}
          <span
            className="start-editor-state"
            data-tone={ready ? "ready" : "draft"}
          >
            {ready ? "Ready" : "Draft"}
          </span>
        </h4>
        <Button
          variant="ghost"
          size="xs"
          disabled={busy}
          aria-label="Use a project material"
          onClick={() => draft.setMode("material")}
        >
          Use existing
        </Button>
      </header>
      <p className="start-checklist-intro">One clear check per item.</p>
      <ol ref={entries} className="start-checklist-entries">
        {draft.entries.map((entry, index) => (
          <li key={entry.key} className="start-checklist-entry">
            <span className="start-checklist-ordinal" aria-hidden="true">
              {String(index + 1).padStart(2, "0")}
            </span>
            <div className="start-checklist-entry-body">
              <label
                htmlFor={`${id}-${entry.key}`}
                className="ui-visually-hidden"
              >
                Item {index + 1}
              </label>
              <textarea
                id={`${id}-${entry.key}`}
                className="start-checklist-statement"
                rows={2}
                maxLength={CHECKLIST_TEXT_BYTES}
                value={entry.statement}
                disabled={busy}
                placeholder="Describe what should be verified…"
                onChange={(event) =>
                  draft.update(entry.key, event.target.value)
                }
              />
            </div>
            <Button
              variant="ghost"
              size="xs"
              className="start-checklist-remove"
              disabled={busy}
              aria-label={`Remove item ${index + 1}`}
              onClick={() => {
                focusAfterChange.current = index;
                draft.remove(entry.key);
              }}
            >
              <Icon name="delete" size={16} />
            </Button>
          </li>
        ))}
      </ol>
      <div className="start-checklist-actions">
        <Button
          ref={addButton}
          variant="ghost"
          className="start-checklist-add"
          disabled={busy || count >= draft.maximumItems}
          icon={<Icon name="plus" size={16} />}
          onClick={() => {
            focusAfterChange.current = count;
            draft.add();
          }}
        >
          Add item
        </Button>
        <Button
          className="start-checklist-apply"
          data-applied={ready}
          disabled={busy || ready || draft.error !== undefined}
          onClick={draft.apply}
        >
          {ready ? "Checklist ready" : "Use checklist"}
        </Button>
      </div>
      <p className="start-checklist-feedback" data-tone={tone} role="status">
        {feedback}
      </p>
      {ready && draft.retained === undefined ? (
        <p className="ui-caption start-tight">
          Will be saved as a project material with the check.
        </p>
      ) : null}
    </div>
  );
}
