import {
  canonicalMutationRequest,
  createMutationIdempotencyKey,
} from "../../mutations/idempotency";
import type { EvalCommand, EvalCommandReceipt } from "../../api/evals";

const STORAGE_PREFIX = "contractor.eval-recovery.v1:";
export interface PendingCommand {
  key: string;
  revision: number;
  body: EvalCommand;
  commandId?: string;
}

function storageKey(owner: string, resource: string): string {
  return (
    STORAGE_PREFIX +
    encodeURIComponent(owner) +
    ":" +
    encodeURIComponent(resource)
  );
}

async function digest(value: unknown): Promise<string> {
  const data = new TextEncoder().encode(canonicalMutationRequest(value));
  const hash = await crypto.subtle.digest("SHA-256", data);
  return Array.from(new Uint8Array(hash), (x) =>
    x.toString(16).padStart(2, "0"),
  ).join("");
}

// Persist only correlation, never request bodies containing cases, rubrics,
// credentials or input parameters. The server owns the saved draft.
export async function mutationKey(
  owner: string,
  operation: string,
  request: unknown,
): Promise<string> {
  const entry = storageKey(owner, operation + ":" + (await digest(request)));
  const existing = localStorage.getItem(entry);
  if (existing && /^eval-ui-[0-9a-f]{32}$/.test(existing)) return existing;
  const key = createMutationIdempotencyKey("eval");
  localStorage.setItem(entry, key);
  return key;
}

export async function finishMutation(
  owner: string,
  operation: string,
  request: unknown,
): Promise<void> {
  localStorage.removeItem(
    storageKey(owner, operation + ":" + (await digest(request))),
  );
}

export function readCommand(owner: string, id: string): PendingCommand | null {
  try {
    const raw = localStorage.getItem(storageKey(owner, id));
    if (!raw) return null;
    const value = JSON.parse(raw) as PendingCommand;
    if (
      !/^eval-ui-[0-9a-f]{32}$/.test(value.key) ||
      !Number.isSafeInteger(value.revision) ||
      value.revision < 1
    )
      return null;
    if (
      !["prepare", "start", "pause", "resume", "cancel", "duplicate"].includes(
        value.body.kind,
      )
    )
      return null;
    if (
      value.body.planSha256 !== undefined &&
      !/^sha256:[a-f0-9]{64}$/.test(value.body.planSha256)
    )
      return null;
    if (
      value.commandId !== undefined &&
      !/^[A-Za-z0-9_.:-]{1,256}$/.test(value.commandId)
    )
      return null;
    return {
      key: value.key,
      revision: value.revision,
      body: {
        kind: value.body.kind,
        ...(value.body.planSha256 ? { planSha256: value.body.planSha256 } : {}),
      },
      ...(value.commandId ? { commandId: value.commandId } : {}),
    };
  } catch {
    return null;
  }
}

export function writeCommand(
  owner: string,
  id: string,
  pending: PendingCommand | null,
): void {
  const key = storageKey(owner, id);
  if (pending) localStorage.setItem(key, JSON.stringify(pending));
  else localStorage.removeItem(key);
}

export function commandFinished(command: EvalCommandReceipt): boolean {
  return command.state === "completed" || command.state === "failed";
}
