import { useState } from "react";
import type { Document } from "yaml";
import { editDraft, type Draft, type Path } from "./document";

export interface FormEditor {
  draft: Draft;
  onPatch: (path: Path, value: unknown) => void;
  onReplace: (draft: Draft) => void;
  onRemove: (path: Path, title: string) => void;
}
export function useDraftEdit(draft: Draft, onReplace: FormEditor["onReplace"]) {
  const [error, setError] = useState("");
  return {
    error,
    edit: (apply: (document: Document) => void) => {
      try {
        onReplace(editDraft(draft, apply));
        setError("");
      } catch (error) {
        setError(
          error instanceof Error
            ? error.message
            : "Could not edit this setting.",
        );
      }
    },
  };
}
