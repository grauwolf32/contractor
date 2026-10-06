/**
 * Creating and starting a check from the Start page.
 *
 * "Start check" creates the check (POST /v1/projects/{projectId}/audits) and
 * then starts it (POST /v1/audits/{auditId}/start with If-Match on the new
 * draft's revision and the time limit). "Save as draft" only creates it.
 * Both requests carry an Idempotency-Key that stays the same while the
 * request does, so sending an unchanged request again after a lost answer
 * cannot create a second check, and the If-Match fence keeps a draft from
 * starting twice. When the start fails, the draft this page created is
 * reused for the same request instead of creating another one.
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
  mutateAudit,
  type Audit,
  type CreateAuditRequest,
} from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { invalidateCrossProject } from "../../../api/cross-project";
import { PublicAPIError } from "../../../api/error";
import { queryKeys } from "../../../api/query-keys";
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

export function checkPath(projectId: string, auditId: string): string {
  return `/projects/${encodeURIComponent(projectId)}/audits/${encodeURIComponent(auditId)}`;
}

/** A request whose answer did not arrive: the Server may have applied it. */
export function isLostResponse(error: unknown): boolean {
  return error instanceof PublicAPIError && error.status === 0;
}

export interface Launch {
  /** Creates (unless this page already created it) and optionally starts. */
  launch: (intent: LaunchIntent, input: LaunchRequest) => void;
  /** Sends the last create request again, with its key. */
  retryCreate: () => void;
  /** Starts the draft this page created again. */
  retryStart: () => void;
  /** The action in flight. */
  pending: LaunchIntent | undefined;
  /** "create" or "start" while a request is in flight. */
  step: "create" | "start" | undefined;
  createError: Error | null;
  startError: Error | null;
  /** The draft this page created and has not started yet. */
  draft: Audit | undefined;
  /** Whether a create request equals the draft's request. */
  isDraftRequest: (request: CreateAuditRequest) => boolean;
  /** Whether a create request equals the last one sent (same key). */
  isLastRequest: (request: CreateAuditRequest) => boolean;
}

export function useLaunch(projectId: string): Launch {
  const api = usePublicAPI();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [createKeys] = useState(
    () => new MutationDraftKeyring<CreateVariables>("create-audit"),
  );
  const [startKeys] = useState(
    () => new MutationDraftKeyring<StartKey>("start-audit"),
  );
  const [draft, setDraft] = useState<{ canonical: string; audit: Audit }>();
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

  function open(auditId: string): void {
    if (mounted.current) void navigate(checkPath(projectId, auditId));
  }

  async function startDraft(audit: Audit, deadlineSeconds: number) {
    try {
      const started = auditMutationAudit(
        await start.mutateAsync({ audit, deadlineSeconds }),
      );
      refresh(started.auditId);
      open(started.auditId);
    } catch {
      // The notice explains the answer; the draft stays for another try.
      refresh(audit.auditId);
    }
  }

  async function run(intent: LaunchIntent, input: LaunchRequest) {
    const variables = { projectId, request: input.request };
    const canonical = canonicalMutationRequest(variables);
    setLast({ intent, input });
    create.reset();
    start.reset();
    let audit = draft?.canonical === canonical ? draft.audit : undefined;
    if (audit === undefined) {
      try {
        audit = await create.mutateAsync(variables);
      } catch {
        refresh(undefined);
        return;
      }
      setDraft({ canonical, audit });
    }
    if (intent === "draft") {
      refresh(audit.auditId);
      open(audit.auditId);
      return;
    }
    await startDraft(audit, input.deadlineSeconds);
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
      const audit = draft?.audit;
      if (audit === undefined || last === undefined) return;
      guarded("start", async () => {
        start.reset();
        await startDraft(audit, last.input.deadlineSeconds);
      });
    },
    pending,
    step: create.isPending ? "create" : start.isPending ? "start" : undefined,
    createError: create.error,
    startError: start.error,
    draft: draft?.audit,
    isDraftRequest: (request) =>
      draft !== undefined &&
      draft.canonical === canonicalMutationRequest({ projectId, request }),
    isLastRequest: (request) => createKeys.matches({ projectId, request }),
  };
}
