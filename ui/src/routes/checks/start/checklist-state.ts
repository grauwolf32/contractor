import { useCallback, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import type { AuditProfile, ExactArtifactRef } from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { useSharedConfig } from "../../../config/shared-config";
import { queryKeys } from "../../../api/query-keys";
import { retainAuthoredMaterial } from "./authored-material";
import {
  authoredChecklistEntry,
  CHECKLIST_COMPOSER_ITEMS,
  CHECKLIST_DOCUMENT_BYTES,
  checklistDocument,
  checklistInputName,
  checklistStatementError,
  type ChecklistDocument,
  type ChecklistEntry,
} from "./checklist-document";

interface EditorEntry {
  value: ChecklistEntry;
  error: string | undefined;
  bytes: number;
  filled: boolean;
}

function cachedEntry(value: ChecklistEntry): EditorEntry {
  return {
    value,
    filled: value.statement.trim() !== "",
    error: checklistStatementError(value.statement),
    bytes: new TextEncoder().encode(JSON.stringify(value)).byteLength,
  };
}

interface Draft {
  entries: EditorEntry[];
  nextKey: number;
  applied: { document: ChecklistDocument; json: string } | undefined;
}

const EMPTY_DOCUMENT_BYTES = new TextEncoder().encode(
  JSON.stringify(checklistDocument([])),
).byteLength;

/** Local Apply is not persistence; launch preparation retains an ordinary exact Artifact. */
export function useChecklistDraft(
  projectId: string,
  profile: AuditProfile | undefined,
) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const { limits } = useSharedConfig();
  const inputName = checklistInputName(profile);
  const maximumItems = Math.min(
    CHECKLIST_COMPOSER_ITEMS,
    profile?.execution.maxItemsTotal ?? CHECKLIST_COMPOSER_ITEMS,
  );
  const maximumBytes = Math.min(
    CHECKLIST_DOCUMENT_BYTES,
    limits.artifactPayloadMaxBytes,
  );
  const [mode, setMode] = useState<"material" | "write">("material");
  const [draft, setDraft] = useState<Draft>(() => ({
    entries: [cachedEntry(authoredChecklistEntry("check-0001"))],
    nextKey: 2,
    applied: undefined,
  }));
  const [prepared, setPrepared] = useState<{
    json: string;
    ref: ExactArtifactRef;
  }>();
  const dirty =
    draft.applied === undefined ||
    draft.entries.length !== draft.applied.document.items.length ||
    draft.entries.some((entry, index) => {
      const applied = draft.applied?.document.items[index];
      return (
        applied?.key !== entry.value.key ||
        applied.statement !== entry.value.statement ||
        applied.version !== entry.value.version
      );
    });
  const empty = draft.entries.every((entry) => !entry.filled);
  const documentBytes =
    EMPTY_DOCUMENT_BYTES +
    draft.entries.reduce((sum, entry) => sum + entry.bytes, 0) +
    Math.max(0, draft.entries.length - 1);
  const invalidEntry = draft.entries.findIndex(
    (entry) => entry.error !== undefined,
  );
  const error =
    draft.entries.length === 0
      ? "Add at least one checklist item."
      : draft.entries.length > maximumItems
        ? `Use at most ${maximumItems} checklist items for this check type.`
        : invalidEntry >= 0
          ? `Item ${invalidEntry + 1}: ${draft.entries[invalidEntry]!.error}`
          : documentBytes > maximumBytes
            ? "The checklist exceeds the document size limit."
            : undefined;
  const blocker =
    mode !== "write"
      ? undefined
      : (error ??
        (dirty ? "Apply your checklist changes before starting." : undefined));

  const update = useCallback((key: string, statement: string) => {
    setDraft((current) => ({
      ...current,
      entries: current.entries.map((entry) => {
        if (entry.value.key !== key) return entry;
        const previous = current.applied?.document.items.find(
          (item) => item.key === key,
        );
        const version =
          previous === undefined
            ? "1"
            : statement === previous.statement
              ? previous.version
              : String(Number(previous.version) + 1);
        return cachedEntry(authoredChecklistEntry(key, statement, version));
      }),
    }));
  }, []);
  const remove = useCallback((key: string) => {
    setDraft((current) => ({
      ...current,
      entries: current.entries.filter((entry) => entry.value.key !== key),
    }));
  }, []);
  const add = () => {
    setDraft((current) =>
      current.entries.length >= maximumItems
        ? current
        : {
            ...current,
            entries: [
              ...current.entries,
              cachedEntry(
                authoredChecklistEntry(
                  `check-${String(current.nextKey).padStart(4, "0")}`,
                ),
              ),
            ],
            nextKey: current.nextKey + 1,
          },
    );
  };
  const apply = () => {
    if (error !== undefined || !dirty) return;
    const document = checklistDocument(
      draft.entries.map((entry) => entry.value),
    );
    const json = JSON.stringify(document);
    setDraft((current) => ({ ...current, applied: { document, json } }));
  };
  const json = !dirty && error === undefined ? draft.applied?.json : undefined;
  const ref =
    json !== undefined && prepared?.json === json ? prepared.ref : undefined;

  async function prepare(ensureCurrent: () => void): Promise<ExactArtifactRef> {
    ensureCurrent();
    if (
      inputName === undefined ||
      mode !== "write" ||
      blocker !== undefined ||
      json === undefined
    )
      throw new Error(blocker ?? "Apply a valid checklist first.");
    if (ref !== undefined) return ref;
    const retained = await retainAuthoredMaterial(
      api,
      projectId,
      "check-inputs",
      "checklist",
      new Blob([json], { type: "application/json" }),
      ensureCurrent,
    );
    ensureCurrent();
    setPrepared({ json, ref: retained });
    void queryClient.invalidateQueries({
      queryKey: queryKeys.projects.artifacts.all(projectId),
    });
    return retained;
  }

  return {
    inputName,
    mode,
    setMode,
    entries: draft.entries.map((entry) => entry.value),
    maximumItems,
    dirty,
    empty,
    error,
    blocker,
    update,
    remove,
    add,
    apply,
    count: draft.entries.filter((entry) => entry.filled).length,
    applied: json === undefined ? undefined : draft.applied?.document,
    retained: ref,
    prepare,
  };
}

export type ChecklistDraft = ReturnType<typeof useChecklistDraft>;
