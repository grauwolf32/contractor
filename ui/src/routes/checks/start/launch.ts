/**
 * Creating and starting a check from the Start page.
 *
 * "Start check" creates the check (POST /v1/projects/{projectId}/audits) and
 * then starts it (POST /v1/audits/{auditId}/start with If-Match on the
 * draft's revision and the time limit). "Save as draft" only creates it.
 * Both requests carry an Idempotency-Key that stays the same while the
 * request does, so sending an unchanged request again after a lost answer
 * cannot create a second check, and the If-Match fence keeps a draft from
 * starting twice. When the start fails, the draft this page created is
 * reused for the same request instead of creating another one.
 *
 * A start that fails with an answer is followed by a fresh read of the
 * check (GET /v1/audits/{auditId}) that replaces the page's copy: a check
 * that is no longer a draft has started already (an earlier request got
 * through), and a draft at a newer revision is started at that revision
 * next time. Only a refusal whose fresh read still shows a draft counts as
 * "did not start"; a lost answer, a server or gateway failure (5xx) and a
 * failed read leave the outcome open.
 *
 * Nothing is shown as created or started before the Server says so: lists
 * are invalidated and refetched after every answer.
 */
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router";

import {
  auditMutationAudit,
  createAudit,
  getAudit,
  mutateAudit,
  type Audit,
  type CreateAuditRequest,
} from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { invalidateCrossProject } from "../../../api/cross-project";
import { PublicAPIError } from "../../../api/error";
import { queryKeys } from "../../../api/query-keys";
import { usePageLocationNow } from "../../../app/page-location";
import {
  canonicalMutationRequest,
  MutationDraftKeyring,
} from "../../../mutations/idempotency";

export type LaunchIntent = "start" | "draft";

export interface LaunchRequest {
  request: CreateAuditRequest;
  /** Seconds for the start request; 0 means no time limit. */
  deadlineSeconds: number;
}

interface CreateVariables {
  projectId: string;
  request: CreateAuditRequest;
}

interface StartKey {
  auditId: string;
  revision: number;
  deadlineSeconds: number;
}

/** The check this page created, with the create request it came from. */
interface DraftRecord {
  canonical: string;
  /** As last read: the create answer, or a read after a failed start. */
  audit: Audit;
}

/** One start request: the check as the request saw it, and the time limit. */
interface StartAttempt {
  record: DraftRecord;
  deadlineSeconds: number;
}

/** What the page knows after a start request failed. */
export type StartFailure =
  /** The answer did not arrive (status 0): the start may have gone through. */
  | { readonly kind: "lost"; readonly error: Error }
  /**
   * The server or a gateway failed (5xx), or the check could not be read
   * again, so the start may still have gone through. `stillDraft`: a fresh
   * read shows a draft (false when the read failed).
   */
  | {
      readonly kind: "unconfirmed";
      readonly error: Error;
      readonly stillDraft: boolean;
    }
  /** A fresh read shows that the check is no longer a draft. */
  | { readonly kind: "started"; readonly error: Error }
  /** Refused, and a fresh read shows the check is still a draft. */
  | {
      readonly kind: "refused";
      readonly error: Error;
      /** The draft had changed (412 or a newer revision on the fresh read). */
      readonly revisionChanged: boolean;
    };

export function checkPath(projectId: string, auditId: string): string {
  return `/projects/${encodeURIComponent(projectId)}/audits/${encodeURIComponent(auditId)}`;
}

/** A request whose answer did not arrive: the Server may have applied it. */
export function isLostResponse(error: unknown): boolean {
  return error instanceof PublicAPIError && error.status === 0;
}

/** A server or gateway failure: the request may still have been applied. */
function isServerFailure(error: unknown): boolean {
  return error instanceof PublicAPIError && error.status >= 500;
}

function isPreconditionFailure(error: unknown): boolean {
  return error instanceof PublicAPIError && error.status === 412;
}

export interface Launch {
  /** Creates (unless this page already created it) and optionally starts. */
  launch: (intent: LaunchIntent, input: LaunchRequest) => void;
  /** Sends the last create request again, with its key. */
  retryCreate: () => void;
  /** Sends the start request that failed without an outcome again, unchanged. */
  retryStart: () => void;
  /** The action in flight. */
  pending: LaunchIntent | undefined;
  /** The request in flight, or the read that follows a failed start. */
  step: "create" | "start" | "verify" | undefined;
  createError: Error | null;
  /** The last start failure, once the page has looked at the check again. */
  startFailure: StartFailure | undefined;
  /**
   * The check this page created, as last read: its state shows whether it
   * is still a draft.
   */
  draft: Audit | undefined;
  /** Whether a create request equals the draft's request. */
  isDraftRequest: (request: CreateAuditRequest) => boolean;
  /** Whether a create request equals the last one sent (same key). */
  isLastRequest: (request: CreateAuditRequest) => boolean;
}

export function useLaunch(projectId: string): Launch {
  const api = usePublicAPI();
  const navigate = useNavigate();
  const pageNow = usePageLocationNow();
  const queryClient = useQueryClient();
  const [createKeys] = useState(
    () => new MutationDraftKeyring<CreateVariables>("create-audit"),
  );
  const [startKeys] = useState(
    () => new MutationDraftKeyring<StartKey>("start-audit"),
  );
  const [draft, setDraft] = useState<DraftRecord>();
  const [failed, setFailed] = useState<{
    failure: StartFailure;
    attempt: StartAttempt;
  }>();
  const [verifying, setVerifying] = useState(false);
  const [last, setLast] = useState<{
    intent: LaunchIntent;
    input: LaunchRequest;
  }>();
  const [pending, setPending] = useState<LaunchIntent>();
  const inFlight = useRef(false);
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  const create = useMutation({
    mutationFn: (variables: CreateVariables) =>
      createAudit(api, {
        projectId: variables.projectId,
        request: variables.request,
        idempotencyKey: createKeys.keyFor(variables),
      }),
  });
  const start = useMutation({
    mutationFn: ({
      audit,
      deadlineSeconds,
    }: {
      audit: Audit;
      deadlineSeconds: number;
    }) =>
      mutateAudit(api, "start", {
        auditId: audit.auditId,
        expectedRevision: audit.revision,
        idempotencyKey: startKeys.keyFor({
          auditId: audit.auditId,
          revision: audit.revision,
          deadlineSeconds,
        }),
        deadlineSeconds,
      }),
  });

  function refresh(auditId: string | undefined): void {
    void Promise.all([
      queryClient.invalidateQueries({
        queryKey: queryKeys.projects.audits.all(projectId),
      }),
      invalidateCrossProject(queryClient),
      ...(auditId === undefined
        ? []
        : [
            queryClient.invalidateQueries({
              queryKey: queryKeys.audits.detail(auditId),
            }),
          ]),
    ]);
  }

  // A started check opens its page, unless the user went elsewhere while
  // the request was on its way (the next page may still be loading, with
  // this one on screen).
  function open(auditId: string): void {
    if (pageNow() !== undefined) void navigate(checkPath(projectId, auditId));
  }

  async function readAgain(auditId: string): Promise<Audit | undefined> {
    setVerifying(true);
    try {
      return await getAudit(api, auditId);
    } catch {
      return undefined;
    } finally {
      if (mounted.current) setVerifying(false);
    }
  }

  // Works out what a failed start means. A lost answer is retried with the
  // same key instead; anything else reads the check again, because an
  // earlier request (or another tab) may have started it meanwhile.
  async function settle(attempt: StartAttempt, error: unknown): Promise<void> {
    const reason =
      error instanceof Error ? error : new Error("The start request failed");
    if (isLostResponse(error)) {
      setFailed({ failure: { kind: "lost", error: reason }, attempt });
      return;
    }
    const current = await readAgain(attempt.record.audit.auditId);
    if (current !== undefined)
      setDraft({ canonical: attempt.record.canonical, audit: current });
    let failure: StartFailure;
    if (current === undefined)
      failure = { kind: "unconfirmed", error: reason, stillDraft: false };
    else if (current.state !== "draft")
      failure = { kind: "started", error: reason };
    else if (isServerFailure(error))
      failure = { kind: "unconfirmed", error: reason, stillDraft: true };
    else
      failure = {
        kind: "refused",
        error: reason,
        revisionChanged:
          isPreconditionFailure(error) ||
          current.revision !== attempt.record.audit.revision,
      };
    setFailed({ failure, attempt });
  }

  async function startDraft(attempt: StartAttempt): Promise<void> {
    try {
      const started = auditMutationAudit(
        await start.mutateAsync({
          audit: attempt.record.audit,
          deadlineSeconds: attempt.deadlineSeconds,
        }),
      );
      refresh(started.auditId);
      open(started.auditId);
    } catch (error) {
      // The notice explains the answer; the draft stays for another try.
      await settle(attempt, error);
      refresh(attempt.record.audit.auditId);
    }
  }

  async function run(intent: LaunchIntent, input: LaunchRequest) {
    const variables = { projectId, request: input.request };
    const canonical = canonicalMutationRequest(variables);
    let record = draft?.canonical === canonical ? draft : undefined;
    // A check of this request that left the draft state is not sent again
    // (the page blocks both actions for it).
    if (record !== undefined && record.audit.state !== "draft") return;
    setLast({ intent, input });
    create.reset();
    start.reset();
    setFailed(undefined);
    if (record === undefined) {
      let audit: Audit;
      try {
        audit = await create.mutateAsync(variables);
      } catch {
        refresh(undefined);
        return;
      }
      record = { canonical, audit };
      setDraft(record);
    }
    if (intent === "draft") {
      refresh(record.audit.auditId);
      open(record.audit.auditId);
      return;
    }
    await startDraft({ record, deadlineSeconds: input.deadlineSeconds });
  }

  function guarded(intent: LaunchIntent, work: () => Promise<void>): void {
    if (inFlight.current) return;
    inFlight.current = true;
    setPending(intent);
    void work().finally(() => {
      inFlight.current = false;
      if (mounted.current) setPending(undefined);
    });
  }

  return {
    launch: (intent, input) => guarded(intent, () => run(intent, input)),
    retryCreate: () => {
      if (last !== undefined)
        guarded(last.intent, () => run(last.intent, last.input));
    },
    retryStart: () => {
      if (
        failed === undefined ||
        (failed.failure.kind !== "lost" &&
          failed.failure.kind !== "unconfirmed")
      )
        return;
      const { attempt } = failed;
      guarded("start", async () => {
        start.reset();
        setFailed(undefined);
        await startDraft(attempt);
      });
    },
    pending,
    step: create.isPending
      ? "create"
      : start.isPending
        ? "start"
        : verifying
          ? "verify"
          : undefined,
    createError: create.error,
    startFailure: failed?.failure,
    draft: draft?.audit,
    isDraftRequest: (request) =>
      draft !== undefined &&
      draft.canonical === canonicalMutationRequest({ projectId, request }),
    isLastRequest: (request) => createKeys.matches({ projectId, request }),
  };
}
