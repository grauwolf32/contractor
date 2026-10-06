// Fixtures and a fake Server for the decision component tests.
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render } from "@testing-library/react";
import type { ReactElement } from "react";
import { vi } from "vitest";

import type {
  AuditFinding,
  AuditReport,
  AuditReviewRequest,
} from "../../api/audits";
import { PublicAPI } from "../../api/client";
import { PublicAPIProvider } from "../../api/context";
import type { RuntimeConfig } from "../../config/runtime-config";
import type { AuditReviewDecision } from "./model";

export const AUDIT_ID = "audit_1";
export const FINDING_ID = "finding_1";
export const DIGEST = `sha256:${"a".repeat(64)}`;
export const NOW = "2026-10-05T10:00:00Z";

const runtimeConfig: RuntimeConfig = {
  uiVersion: "0.1.0",
  supportedApiVersions: ["contractor.public.v1"],
  apiBaseUrl: "http://127.0.0.1:8080",
};

export function json(
  value: unknown,
  status = 200,
  headers: Record<string, string> = {},
): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: {
      "Content-Type": "application/json",
      "X-Contractor-API-Version": "contractor.public.v1",
      ...headers,
    },
  });
}

export function failure(status: number, code: string, message: string) {
  return json({ code, message, retryable: false, requestId: "req_1" }, status);
}

export function makeFinding(
  overrides: Partial<AuditFinding> = {},
  document: Partial<AuditFinding["firstProposal"]["document"]> = {},
): AuditFinding {
  return {
    findingId: FINDING_ID,
    auditId: AUDIT_ID,
    state: "proposed",
    firstProposal: {
      receiptId: "receipt_1",
      proposalId: "proposal_1",
      requestDigest: DIGEST,
      clientKey: "call-1",
      proposal: {
        ref: { namespace: "audit-findings", name: "call-1", revision: "r1" },
        digest: DIGEST,
        mediaType: "application/json",
        sizeBytes: 512,
      },
      document: {
        schema: "contractor.audit.finding-proposal.v1",
        client_key: "call-1",
        title: "Missing object authorization",
        description:
          "The **order endpoint** may read another owner's record.\n\nIt never compares `ownerId` with the caller.",
        subject: null,
        preconditions: [],
        standard_refs: [],
        evidence_ids: [],
        proposed_checks: [],
        severity_suggestion: "high",
        limitations: [],
        ...document,
      },
      evidence: [],
      origin: {
        runId: "run_1",
        stageExecutionId: "stage_1",
        allocationId: "allocation_1",
        invocationId: "invocation_1",
        logicalAgentName: "reviewer",
        workflow: {
          name: "source-review",
          version: "1",
          schemaVersion: "contractor/v1alpha1",
          configurationRef: { name: "source-review", version: "1" },
          closureDigest: DIGEST,
        },
        runDeleted: false,
      },
      retention: "audit-held",
      auditHolds: [],
      createdAt: NOW,
    },
    revision: 3,
    createdAt: NOW,
    updatedAt: NOW,
    ...overrides,
  };
}

export const ALL_VERDICTS: AuditReviewRequest["requestedActions"] = [
  "true_positive",
  "false_positive",
  "duplicate",
  "reopen",
  "needs_evidence",
];

export function makeTriageReview(
  overrides: Partial<AuditReviewRequest> = {},
): AuditReviewRequest {
  return {
    requestId: "review_open",
    auditId: AUDIT_ID,
    findingId: FINDING_ID,
    subjectKind: "finding",
    subjectId: FINDING_ID,
    kind: "finding-triage",
    subjectRevision: 3,
    subjectDigest: DIGEST,
    requestedActions: ALL_VERDICTS,
    state: "pending",
    revision: 1,
    createdAt: NOW,
    updatedAt: NOW,
    ...overrides,
  };
}

export function makeActionReview(
  overrides: Partial<AuditReviewRequest> = {},
): AuditReviewRequest {
  return {
    requestId: "review_action",
    auditId: AUDIT_ID,
    subjectKind: "audit-item-action",
    subjectId: "item_1",
    kind: "active-check-approval",
    subjectRevision: 1,
    subjectDigest: DIGEST,
    requestedActions: ["approve", "reject"],
    state: "pending",
    revision: 2,
    createdAt: NOW,
    updatedAt: NOW,
    ...overrides,
  };
}

export function makeDecision(
  overrides: Partial<AuditReviewDecision> = {},
): AuditReviewDecision {
  return {
    decisionId: "decision_1",
    requestId: "review_open",
    auditId: AUDIT_ID,
    findingId: FINDING_ID,
    actorId: "user_analyst",
    verdict: "true_positive",
    severity: "high",
    rationale: "Confirmed against the **retained** source.",
    subjectRevision: 3,
    subjectDigest: DIGEST,
    createdAt: NOW,
    ...overrides,
  };
}

export function proposedReport(review: AuditReviewRequest): AuditReport {
  return {
    status: "proposed",
    review,
    summary: "# Report",
    summaryArtifact: {
      ref: { namespace: "audit-reports", name: "summary", revision: "r1" },
      digest: DIGEST,
      mediaType: "text/markdown",
    },
  };
}

export type Handler = (
  request: Request,
  url: URL,
) => Response | Promise<Response>;

/**
 * Renders `ui` with a Public API client whose requests go to `handle`. The
 * client holds a CSRF token, as after sign-in. Every request is recorded.
 */
export function renderWithServer(ui: ReactElement, handle: Handler) {
  const requests: Request[] = [];
  const api = new PublicAPI(
    runtimeConfig,
    vi.fn(async (input: RequestInfo | URL) => {
      const request = input instanceof Request ? input : new Request(input);
      requests.push(request.clone());
      return handle(request, new URL(request.url));
    }),
  );
  api.csrf.replace("a".repeat(43));
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, refetchOnWindowFocus: false },
      mutations: { retry: false },
    },
  });
  const wrap = (node: ReactElement) => (
    <QueryClientProvider client={queryClient}>
      <PublicAPIProvider api={api}>{node}</PublicAPIProvider>
    </QueryClientProvider>
  );
  const view = render(wrap(ui));
  return {
    ...view,
    api,
    queryClient,
    requests,
    rerender: (node: ReactElement) => view.rerender(wrap(node)),
    /** Requests with this method whose path ends with `suffix`. */
    sent: (method: string, suffix: string) =>
      requests.filter(
        (request) =>
          request.method === method &&
          new URL(request.url).pathname.endsWith(suffix),
      ),
  };
}
