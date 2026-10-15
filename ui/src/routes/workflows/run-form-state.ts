import { useEffect, useRef, useState } from "react";

import type { ArtifactMetadata } from "../../api/artifacts";
import {
  RUN_METADATA_LABEL_LIMIT,
  type RunMetadataLabelDraft,
} from "../../api/run-metadata-labels";
import type {
  RunDraftEntry,
  RunDraftMemoryStore,
  RunDraftState,
} from "../../run-drafts/memory";
import {
  artifactOptionKey,
  type ConsumerOverrideDraft,
} from "../../run-drafts/validation";

/** The labels "Add eval metadata preset" adds when they are missing. */
export const EVAL_METADATA_PRESET = [
  { key: "purpose", value: "eval" },
  { key: "eval.name", value: "" },
  { key: "eval.id", value: "" },
  { key: "eval.leg", value: "" },
] as const;

/** Validation messages by field key ("parameter:objective", "runtimeLabels"). */
export type RunDraftErrors = Readonly<Record<string, string>>;

function updateRecord(
  source: Record<string, string>,
  name: string,
  value: string,
): Record<string, string> {
  if (value === "") {
    const result = { ...source };
    delete result[name];
    return result;
  }
  return { ...source, [name]: value };
}

/**
 * The editable state of one Run draft: every change goes through the draft
 * store first (a draft another tab discarded or replaced is not changed),
 * and clears the validation message of the field it changed.
 */
export function useRunDraftState(
  draftEntry: RunDraftEntry,
  draftStore: RunDraftMemoryStore,
) {
  const [draft, setDraft] = useState<RunDraftState>(() =>
    structuredClone(draftEntry.state),
  );
  const [errors, setErrors] = useState<Record<string, string>>({});
  const labelSequence = useRef(
    Math.max(
      0,
      ...draft.metadataLabels.map(
        (label) => Number.parseInt(label.id, 10) || 0,
      ),
    ),
  );

  useEffect(() => {
    draftStore.retain(draftEntry);
    return () => draftStore.release(draftEntry);
  }, [draftEntry, draftStore]);

  function updateDraft(
    update: (current: RunDraftState) => RunDraftState,
  ): void {
    if (!draftStore.isCurrent(draftEntry)) return;
    const next = update(structuredClone(draftEntry.state));
    if (draftStore.replaceState(draftEntry, next)) {
      setDraft(next);
    }
  }

  /** Updates one member of the draft. */
  function change<K extends keyof RunDraftState>(
    key: K,
    update: (current: RunDraftState[K]) => RunDraftState[K],
  ): void {
    updateDraft((current) => ({ ...current, [key]: update(current[key]) }));
  }

  function clearError(key: string): void {
    setErrors((current) => {
      if (!(key in current)) {
        return current;
      }
      const result = { ...current };
      delete result[key];
      return result;
    });
  }

  function clearMetadataLabelErrors(): void {
    setErrors((current) =>
      Object.fromEntries(
        Object.entries(current).filter(
          ([key]) =>
            key !== "metadataLabels" && !key.startsWith("metadataLabel:"),
        ),
      ),
    );
  }

  function newMetadataLabel(key = "", value = ""): RunMetadataLabelDraft {
    labelSequence.current += 1;
    return { id: String(labelSequence.current), key, value };
  }

  function changeMetadataLabels(
    update: (current: RunMetadataLabelDraft[]) => RunMetadataLabelDraft[],
  ): void {
    change("metadataLabels", update);
    clearMetadataLabelErrors();
  }

  const metadataLabels = {
    add(key = "", value = ""): void {
      if (draft.metadataLabels.length >= RUN_METADATA_LABEL_LIMIT) {
        setErrors((current) => ({
          ...current,
          metadataLabels: `A Run can have at most ${RUN_METADATA_LABEL_LIMIT} metadata labels.`,
        }));
        return;
      }
      const label = newMetadataLabel(key, value);
      changeMetadataLabels((current) => [...current, label]);
    },
    addEvalPreset(): void {
      const existingKeys = new Set(
        draft.metadataLabels.map((label) => label.key),
      );
      const missing = EVAL_METADATA_PRESET.filter(
        (label) => !existingKeys.has(label.key),
      );
      if (
        draft.metadataLabels.length + missing.length >
        RUN_METADATA_LABEL_LIMIT
      ) {
        setErrors((current) => ({
          ...current,
          metadataLabels: `The eval preset would exceed ${RUN_METADATA_LABEL_LIMIT} metadata labels.`,
        }));
        return;
      }
      const additions = missing.map((label) =>
        newMetadataLabel(label.key, label.value),
      );
      changeMetadataLabels((current) => [...current, ...additions]);
    },
    update(id: string, field: "key" | "value", value: string): void {
      changeMetadataLabels((current) =>
        current.map((label) =>
          label.id === id ? { ...label, [field]: value } : label,
        ),
      );
    },
    remove(id: string): void {
      changeMetadataLabels((current) =>
        current.filter((label) => label.id !== id),
      );
    },
    setInput(value: string): void {
      change("metadataLabelInput", () => value);
    },
  };

  return {
    draft,
    errors,
    setErrors,
    clearError,
    change,
    metadataLabels,
    setRepeatReviewed(reviewed: boolean): void {
      updateDraft((current) => ({
        ...current,
        ...(current.repeat === undefined
          ? {}
          : { repeat: { ...current.repeat, reviewed } }),
      }));
      if (reviewed) clearError("repeatReview");
    },
    /** Selects `selected` (an exact ref key) for `slot`, as reviewed. */
    selectArtifact(
      slot: string,
      selected: string,
      metadata?: ArtifactMetadata,
    ): void {
      updateDraft((current) => {
        const known =
          metadata === undefined
            ? current.knownArtifacts
            : [
                ...current.knownArtifacts.filter(
                  (candidate) =>
                    artifactOptionKey(candidate.artifact) !== selected,
                ),
                metadata,
              ];
        return {
          ...current,
          artifactSelections: updateRecord(
            current.artifactSelections,
            slot,
            selected,
          ),
          artifactReviews: updateRecord(
            current.artifactReviews,
            slot,
            selected,
          ),
          knownArtifacts: known,
        };
      });
      clearError(`artifact:${slot}`);
      clearError(`artifactReview:${slot}`);
    },
    /** Confirms the suggested selection of `slot`. */
    confirmArtifact(slot: string): void {
      const selected = draft.artifactSelections[slot];
      if (selected === undefined || selected === "") return;
      updateDraft((current) => ({
        ...current,
        artifactReviews: { ...current.artifactReviews, [slot]: selected },
      }));
      clearError(`artifactReview:${slot}`);
    },
    updateOverride(
      role: "planner" | "workers",
      field: keyof ConsumerOverrideDraft,
      value: string,
    ): void {
      change("overrides", (current) => ({
        ...current,
        [role]: { ...current[role], [field]: value },
      }));
    },
  };
}

export type RunDraftEditor = ReturnType<typeof useRunDraftState>;
