import type { AuditProfile } from "../../../api/audits";
import { mediaTypeMatches } from "./check-types";

export const CHECKLIST_SCHEMA = "contractor.audit.checklist.v1";
// Shared domain bounds are checked against api/testdata/audit-common/boundaries.json.
export const CHECKLIST_TEXT_BYTES = 64 * 1024;
export const CHECKLIST_DOCUMENT_BYTES = 8 * 1024 * 1024;
// LIM-UI-20: the inline composer is deliberately smaller than a server inventory.
export const CHECKLIST_COMPOSER_ITEMS = 128;

export interface ChecklistEntry {
  key: string;
  version: string;
  statement: string;
  applicability: "always";
  allowed_methods: ["static"];
  required_evidence: [];
  review_policy: "automatic";
}

export interface ChecklistDocument {
  schema: typeof CHECKLIST_SCHEMA;
  items: ChecklistEntry[];
}

export function checklistInputName(
  profile: AuditProfile | undefined,
): string | undefined {
  const source = profile?.inventory.source;
  if (
    profile?.inventory.implementation !== "checklist@1" ||
    source?.source !== "audit-input" ||
    source.name === undefined
  )
    return undefined;
  const input = profile.inputs[source.name];
  return input !== undefined &&
    mediaTypeMatches("application/json", input.mediaTypes)
    ? source.name
    : undefined;
}

export function authoredChecklistEntry(
  key: string,
  statement = "",
  version = "1",
): ChecklistEntry {
  return {
    key,
    version,
    statement,
    applicability: "always",
    allowed_methods: ["static"],
    required_evidence: [],
    review_policy: "automatic",
  };
}

export function checklistStatementError(statement: string): string | undefined {
  if (statement.trim() === "") return "Write what should be verified.";
  if (
    Array.from(statement).some((character) => {
      const code = character.codePointAt(0)!;
      return code >= 0xd800 && code <= 0xdfff;
    }) ||
    statement.includes("\0")
  )
    return "Remove the invalid text character.";
  if (new TextEncoder().encode(statement).byteLength > CHECKLIST_TEXT_BYTES)
    return "Use at most 64 KiB of text per item.";
  return undefined;
}

export function checklistDocument(
  entries: readonly ChecklistEntry[],
): ChecklistDocument {
  return {
    schema: CHECKLIST_SCHEMA,
    items: entries.map((entry) => ({ ...entry })),
  };
}
