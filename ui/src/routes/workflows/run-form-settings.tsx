import { useSharedConfig } from "../../config/shared-config";
import { useRef, type ReactNode } from "react";

import type { RuntimeLabelBinding } from "../../api/operations";
import { type RunMetadataLabelDraft } from "../../api/run-metadata-labels";
import { ErrorNotice } from "../../app/error-notice";
import { compactDigest, plural } from "../../app/format";
import { LoadMoreButton } from "../../app/load-more";
import { Button, Field } from "../../ui";
import type { RunFormInventories } from "./run-form-inventories";
import {
  EVAL_METADATA_PRESET,
  type RunDraftEditor,
  type RunDraftErrors,
} from "./run-form-state";
import "./run-drafts.css";

/**
 * An advanced-settings disclosure of the Run form: its summary says what it
 * holds and its state ("Default only", "2 selected"), highlighted while it
 * changes a default. The form opens it when it holds an invalid value.
 */
function SettingsDisclosure({
  title,
  description,
  status,
  active = false,
  open,
  onOpenChange,
  className,
  children,
}: {
  title: string;
  description: string;
  status: string;
  active?: boolean;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  className?: string;
  children: ReactNode;
}) {
  return (
    <details
      className={
        className === undefined
          ? "run-draft-disclosure"
          : `run-draft-disclosure ${className}`
      }
      open={open}
      onToggle={(event) => onOpenChange(event.currentTarget.open)}
    >
      <summary>
        <span className="run-draft-disclosure-title">
          <strong>{title}</strong>
          <small>{description}</small>
        </span>
        <span
          className={`run-draft-disclosure-status ${active ? "is-active" : ""}`}
        >
          {status}
        </span>
      </summary>
      {children}
    </details>
  );
}

function RuntimeLabelPreview({ binding }: { binding: RuntimeLabelBinding }) {
  return (
    <small className="runtime-label-preview">
      <code>
        {binding.config.name}@{binding.config.version}
      </code>
      <span>binding revision {binding.revision}</span>
      <code title={binding.config.digest}>
        {compactDigest(binding.config.digest)}
      </code>
    </small>
  );
}

/**
 * Runtime configuration: the default binding always applies; explicit labels
 * add infrastructure. A retained label no loaded binding has stays
 * selected until it is cleared.
 */
export function RuntimePlacement({
  selected,
  errors,
  inventories,
  open,
  onOpenChange,
  onChange,
}: {
  selected: readonly string[];
  errors: RunDraftErrors;
  inventories: Pick<
    RunFormInventories,
    "runtimeLabelInventory" | "runtimeLabels" | "defaultRuntimeConfig"
  >;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onChange: (update: (current: string[]) => string[]) => void;
}) {
  const { runtimeLabelInventory, runtimeLabels, defaultRuntimeConfig } =
    inventories;
  const visible = new Set(runtimeLabels.map((binding) => binding.label));
  const retained = selected.filter((label) => !visible.has(label));
  return (
    <SettingsDisclosure
      title="Runtime configuration"
      description="Published default is always applied"
      status={
        runtimeLabelInventory.error !== null
          ? "Unavailable"
          : open && runtimeLabelInventory.isPending
            ? "Loading…"
            : selected.length === 0
              ? "Default only"
              : `${selected.length} selected`
      }
      active={selected.length > 0}
      open={open}
      onOpenChange={onOpenChange}
    >
      <fieldset
        className="run-draft-section runtime-label-picker"
        aria-label="Runtime infrastructure labels"
      >
        <p className="muted-copy">
          Default is always applied; explicit labels configure infrastructure,
          not Workflow behavior or model budgets.
        </p>
        {runtimeLabelInventory.error === null ? null : (
          <ErrorNotice error={runtimeLabelInventory.error} />
        )}
        <div className="run-runtime-default">
          <strong>Default · always applied</strong>
          {defaultRuntimeConfig === undefined ? (
            <span className="muted-copy">Loading binding…</span>
          ) : (
            <RuntimeLabelPreview binding={defaultRuntimeConfig} />
          )}
        </div>
        {runtimeLabels.length === 0 &&
        retained.length === 0 &&
        !runtimeLabelInventory.isPending ? (
          <p className="compact-empty">No explicit Runtime labels are bound.</p>
        ) : (
          <div className="runtime-label-options">
            {retained.map((label) => (
              <label
                className="runtime-label-option retained-selection"
                key={`retained:${label}`}
              >
                <input
                  type="checkbox"
                  checked
                  onChange={() =>
                    onChange((current) =>
                      current.filter((candidate) => candidate !== label),
                    )
                  }
                />
                <span>
                  <strong>{label}</strong>
                  <small>
                    Retained label · current binding unavailable or not loaded
                  </small>
                </span>
              </label>
            ))}
            {runtimeLabels.map((binding) => (
              <label className="runtime-label-option" key={binding.label}>
                <input
                  type="checkbox"
                  checked={selected.includes(binding.label)}
                  onChange={(event) =>
                    onChange((current) =>
                      (event.target.checked
                        ? [...current, binding.label]
                        : current.filter((label) => label !== binding.label)
                      ).sort(),
                    )
                  }
                />
                <span>
                  <strong>{binding.label}</strong>
                  <RuntimeLabelPreview binding={binding} />
                </span>
              </label>
            ))}
          </div>
        )}
        {errors.runtimeLabels === undefined ? null : (
          <p className="field-error" role="alert">
            {errors.runtimeLabels}
          </p>
        )}
        <LoadMoreButton
          size="sm"
          query={runtimeLabelInventory}
          label="Load more Runtime labels"
          pendingLabel="Loading Runtime labels…"
        />
      </fieldset>
    </SettingsDisclosure>
  );
}

/** One label as an editable key:value badge with its remove button. */
function MetadataLabelBadge({
  label,
  index,
  errors,
  labels,
  onReturn,
}: {
  label: RunMetadataLabelDraft;
  index: number;
  errors: RunDraftErrors;
  labels: RunDraftEditor["metadataLabels"];
  /** Enter in the badge returns to the label input. */
  onReturn: () => void;
}) {
  const keyError = errors[`metadataLabel:${label.id}:key`];
  const valueError = errors[`metadataLabel:${label.id}:value`];
  const keyErrorID = `metadata-label-${label.id}-key-error`;
  const valueErrorID = `metadata-label-${label.id}-value-error`;
  return (
    <div className="run-label-badge-field">
      <div
        className={`run-label-badge${label.value === "" ? " is-key-only" : ""}${keyError !== undefined || valueError !== undefined ? " has-error" : ""}`}
        onKeyDown={(event) => {
          if (
            event.key === "Enter" &&
            !event.nativeEvent.isComposing &&
            event.target instanceof HTMLInputElement
          ) {
            event.preventDefault();
            onReturn();
          }
        }}
      >
        <label
          className="ui-visually-hidden"
          htmlFor={`metadata-label-${label.id}-key`}
        >
          Run metadata label key {index + 1}
        </label>
        <input
          id={`metadata-label-${label.id}-key`}
          name={`metadata-label-key-${index + 1}`}
          type="text"
          autoComplete="off"
          spellCheck={false}
          placeholder="key"
          className="run-label-badge-key"
          style={{
            width: `calc(${Math.max(3, Math.min(24, label.key.length))}ch + 0.2rem)`,
          }}
          value={label.key}
          title={label.key}
          aria-invalid={keyError === undefined ? undefined : true}
          aria-describedby={keyError === undefined ? undefined : keyErrorID}
          onChange={(event) =>
            labels.update(label.id, "key", event.target.value)
          }
        />
        <span className="run-label-badge-separator" aria-hidden="true">
          :
        </span>
        <label
          className="ui-visually-hidden"
          htmlFor={`metadata-label-${label.id}-value`}
        >
          Run metadata label value {index + 1}
        </label>
        <input
          id={`metadata-label-${label.id}-value`}
          name={`metadata-label-value-${index + 1}`}
          type="text"
          autoComplete="off"
          spellCheck={false}
          placeholder="value"
          className="run-label-badge-value"
          style={{
            width: `calc(${Math.max(5, Math.min(28, label.value.length))}ch + 0.2rem)`,
          }}
          value={label.value}
          title={label.value || "Optional value"}
          aria-invalid={valueError === undefined ? undefined : true}
          aria-describedby={valueError === undefined ? undefined : valueErrorID}
          onChange={(event) =>
            labels.update(label.id, "value", event.target.value)
          }
        />
        <button
          className="run-label-badge-remove"
          type="button"
          aria-label={`Remove Run metadata label ${index + 1}`}
          onClick={() => labels.remove(label.id)}
        >
          <span aria-hidden="true">×</span>
        </button>
      </div>
      {keyError === undefined ? null : (
        <p className="field-error" id={keyErrorID} role="alert">
          {keyError}
        </p>
      )}
      {valueError === undefined ? null : (
        <p className="field-error" id={valueErrorID} role="alert">
          {valueError}
        </p>
      )}
    </div>
  );
}

/**
 * Run metadata: searchable key:value labels, typed as "key:value" or "key"
 * and edited as badges, with the eval preset. Labels are visible through
 * the API, so the editor warns against secrets.
 */
export function RunMetadata({
  labels,
  labelInput,
  errors,
  editor,
  open,
  onOpenChange,
}: {
  labels: readonly RunMetadataLabelDraft[];
  labelInput: string;
  errors: RunDraftErrors;
  editor: RunDraftEditor["metadataLabels"];
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const { limits } = useSharedConfig();
  const labelInputRef = useRef<HTMLInputElement>(null);
  const addButtonRef = useRef<HTMLButtonElement>(null);
  const atLimit = labels.length >= limits.runMetadataMaxLabels;
  const existingKeys = new Set(labels.map((label) => label.key));
  const missingEvalLabels = EVAL_METADATA_PRESET.filter(
    (label) => !existingKeys.has(label.key),
  ).length;

  function addInput(): void {
    if (labelInput.trim() === "" || atLimit) {
      return;
    }
    const separator = labelInput.indexOf(":");
    editor.add(
      (separator < 0 ? labelInput : labelInput.slice(0, separator)).trim(),
      separator < 0 ? "" : labelInput.slice(separator + 1),
    );
    editor.setInput("");
  }

  return (
    <SettingsDisclosure
      title="Run metadata"
      description="Searchable labels and eval identifiers"
      status={
        labels.length === 0 ? "No labels" : plural(labels.length, "label")
      }
      active={labels.length > 0}
      open={open}
      onOpenChange={onOpenChange}
    >
      <fieldset
        className="run-draft-section run-metadata-label-editor"
        aria-label="Run metadata labels"
      >
        <p className="muted-copy">
          Add labels to find and group Runs. Edit a badge directly or remove it
          with × before starting the Run. cloud with an empty value selects any
          cloud; cloud:aws selects AWS, and local:lab selects a local site.
        </p>
        <div className="run-label-secret-warning">
          <strong>Do not put secrets in labels.</strong>
          <p>
            Keys and values are visible through the Run API, list filters and
            telemetry.
          </p>
        </div>
        <div className="run-label-composer">
          <Field
            className="run-field"
            id="run-label-input"
            label="Add a label"
            hint={
              atLimit
                ? `Limit reached: ${limits.runMetadataMaxLabels} labels.`
                : "Type key:value or just key and press Enter. Labels are fixed once the Run starts."
            }
          >
            {(control) => (
              <input
                {...control}
                ref={labelInputRef}
                placeholder="team:platform or debug"
                autoComplete="off"
                spellCheck={false}
                value={labelInput}
                disabled={atLimit}
                onChange={(event) => editor.setInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.nativeEvent.isComposing) {
                    event.preventDefault();
                    addInput();
                  }
                }}
                onBlur={(event) => {
                  if (event.relatedTarget !== addButtonRef.current) {
                    addInput();
                  }
                }}
              />
            )}
          </Field>
          <Button
            ref={addButtonRef}
            size="sm"
            disabled={atLimit}
            onClick={() => {
              if (labelInput.trim() === "") {
                editor.add();
              } else {
                addInput();
                labelInputRef.current?.focus();
              }
            }}
          >
            Add metadata label
          </Button>
        </div>
        <div className="run-metadata-label-actions">
          <Button
            size="sm"
            disabled={
              missingEvalLabels === 0 ||
              labels.length + missingEvalLabels > limits.runMetadataMaxLabels
            }
            onClick={editor.addEvalPreset}
          >
            Add eval metadata preset
          </Button>
          <small>
            {labels.length}/{limits.runMetadataMaxLabels} labels
          </small>
        </div>
        {errors.metadataLabels === undefined ? null : (
          <p className="field-error" role="alert">
            {errors.metadataLabels}
          </p>
        )}
        {labels.length === 0 ? (
          <p className="compact-empty">No Run metadata labels.</p>
        ) : (
          <div className="run-label-badges">
            {labels.map((label, index) => (
              <MetadataLabelBadge
                key={label.id}
                label={label}
                index={index}
                errors={errors}
                labels={editor}
                onReturn={() => labelInputRef.current?.focus()}
              />
            ))}
          </div>
        )}
      </fieldset>
    </SettingsDisclosure>
  );
}
