import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter } from "react-router";
import { describe, expect, it, vi } from "vitest";

import type { RuntimeConfig } from "../../../config/runtime-config";
import { PublicAPI } from "../../../api/client";
import type { Audit, AuditProfile } from "../../../api/audits";
import { Application } from "../../../app/application";
import { applicationRoutes } from "../../../app/router";

const runtimeConfig: RuntimeConfig = {
  uiVersion: "0.1.0",
  supportedApiVersions: ["contractor.public.v1"],
  apiBaseUrl: "http://127.0.0.1:8080",
};

const session = {
  principal: {
    userId: "user_local",
    username: "owner",
    capabilities: ["user", "operations"],
  },
  csrfToken: "a".repeat(43),
  idleExpiresAt: "2026-09-06T20:00:00Z",
  absoluteExpiresAt: "2026-09-07T12:00:00Z",
};

const project = {
  projectId: "project_example",
  kind: "project",
  name: "Payment service",
  description: "Reusable service analysis",
  lifecycle: "active",
  revision: "1",
  createdAt: "2026-09-06T10:00:00Z",
  updatedAt: "2026-09-06T10:00:00Z",
};

const profile: AuditProfile = {
  ref: {
    name: "source-checklist",
    version: "1",
    digest: `sha256:${"2".repeat(64)}`,
  },
  mode: "custom-checklist",
  standards: [],
  inputs: { sources: { required: true, mediaTypes: ["application/zip"] } },
  inventory: {
    implementation: "checklist@1",
    sourceInput: "sources",
    itemWorkflowRole: "check",
  },
  execution: {
    roundMode: "fixed-barrier",
    maxRounds: 1,
    batchSize: 1,
    maxItemsPerRound: 8,
    maxItemsTotal: 8,
    maxSubmittedRuns: 8,
    maxItemRunAttempts: 2,
    deadlineSeconds: 600,
    maxEvidenceBytes: 1_048_576,
    incompleteRound: "assess-with-gaps",
  },
  interaction: {
    activeChecks: "prohibited",
    findingConfirmation: "disabled",
    notApplicable: "profile-rule",
    reportAcceptance: "automatic",
  },
  serverCompatible: true,
  requiresInputValidation: true,
  compatibilityReasons: [],
};

const sourceArtifact = {
  artifact: {
    namespace: "sources",
    name: "payment-service",
    revision: "revision-7",
  },
  mediaType: "application/zip",
  size: 2048,
  current: true,
  frozen: false,
  createdAt: "2026-09-06T10:00:00Z",
};

function auditAt(state: Audit["state"], revision: number): Audit {
  return {
    auditId: "audit_example",
    projectId: project.projectId,
    profile: { ...profile.ref },
    inputs: {
      sources: {
        ref: { ...sourceArtifact.artifact },
        digest: `sha256:${"1".repeat(64)}`,
        mediaType: sourceArtifact.mediaType,
        sizeBytes: sourceArtifact.size,
      },
    },
    scope: { objective: "Review source controls" },
    runtimeLabels: ["debug"],
    state,
    revision,
    dispatchState: state === "active" ? "open" : "closed",
    holdState: state === "draft" ? "pending" : "held",
    limits: {
      maxRounds: 1,
      batchSize: 1,
      maxItemsPerRound: 8,
      maxItemsTotal: 8,
      maxSubmittedRuns: 8,
      maxItemRunAttempts: 2,
      maxEvidenceBytes: 1_048_576,
    },
    reservedRunCount: state === "draft" ? 0 : 1,
    submittedRunCount: state === "draft" ? 0 : 1,
    outstandingRunCount: state === "active" ? 1 : 0,
    retainedEvidenceBytes: 0,
    eventSequence: revision,
    createdAt: "2026-09-06T10:00:00Z",
    updatedAt: `2026-09-06T10:0${revision}:00Z`,
  };
}

function jsonResponse(value: unknown, options: ResponseInit = {}): Response {
  const headers = new Headers(options.headers);
  headers.set("content-type", "application/json");
  headers.set("X-Contractor-API-Version", "contractor.public.v1");
  return new Response(JSON.stringify(value), { ...options, headers });
}

function renderApplication(api: PublicAPI, path: string) {
  const router = createMemoryRouter(applicationRoutes(), {
    initialEntries: [path],
  });
  return {
    ...render(<Application api={api} publicAPI={api} router={router} />),
    router,
  };
}

describe("Project Audit routes", () => {
  it("creates a draft from one compatible exact Project Artifact", async () => {
    const requests: Request[] = [];
    const draft = auditAt("draft", 1);
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        requests.push(request.clone());
        const path = new URL(request.url).pathname;
        if (path === "/v1/auth/session") return jsonResponse(session);
        if (path === "/v1/projects/project_example") {
          return jsonResponse(project, { headers: { ETag: '"1"' } });
        }
        if (path === "/v1/audit-profiles") {
          return jsonResponse({ items: [profile], page: { hasMore: false } });
        }
        if (path === "/v1/audit-profiles/source-checklist/versions/1") {
          return jsonResponse(profile, {
            headers: { ETag: `"${profile.ref.digest}"` },
          });
        }
        if (path === "/v1/projects/project_example/artifacts") {
          return jsonResponse({
            items: [sourceArtifact],
            page: { hasMore: false },
          });
        }
        if (path === "/v1/projects/project_example/audits") {
          if (request.method === "POST") {
            return jsonResponse(draft, {
              status: 201,
              headers: { ETag: '"1"' },
            });
          }
          return jsonResponse({ items: [], page: { hasMore: false } });
        }
        if (path === "/v1/audits/audit_example") {
          return jsonResponse(draft, { headers: { ETag: '"1"' } });
        }
        throw new Error(`unexpected ${request.method} ${path}`);
      }),
    );
    const { router } = renderApplication(
      api,
      "/projects/project_example/audits",
    );
    const user = userEvent.setup();

    await screen.findByRole("heading", { name: "New Audit" });
    const createButton = screen.getByRole("button", {
      name: "Create Audit draft",
    });
    expect(createButton).toBeDisabled();
    await user.selectOptions(
      await screen.findByLabelText("Input sources"),
      screen.getByRole("option", {
        name: /sources\/payment-service@revision-7/u,
      }),
    );
    expect(createButton).toBeEnabled();
    await user.type(screen.getByLabelText("Objective"), "Map attack surface");
    await user.click(createButton);

    await vi.waitFor(() =>
      expect(router.state.location.pathname).toBe(
        "/projects/project_example/audits/audit_example",
      ),
    );
    expect(
      await screen.findByRole("heading", { name: "audit_example" }),
    ).toBeVisible();
    const create = requests.find(
      (request) =>
        request.method === "POST" &&
        new URL(request.url).pathname === "/v1/projects/project_example/audits",
    );
    expect(create?.headers.get("Idempotency-Key")).toMatch(
      /^create-audit-ui-/u,
    );
    expect(create?.headers.get("X-CSRF-Token")).toBe(session.csrfToken);
    await expect(create?.json()).resolves.toEqual({
      profile: { name: "source-checklist", version: "1" },
      inputs: { sources: sourceArtifact.artifact },
      scope: { objective: "Map attack surface" },
    });
  });

  it("renders mixed coverage as assessments and refetches active Audits", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    let auditReads = 0;
    const active = auditAt("active", 2);
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const path = new URL(request.url).pathname;
        if (path === "/v1/auth/session") return jsonResponse(session);
        if (path === "/v1/projects/project_example") {
          return jsonResponse(project, { headers: { ETag: '"1"' } });
        }
        if (path === "/v1/audits/audit_example") {
          auditReads += 1;
          return jsonResponse(active, { headers: { ETag: '"2"' } });
        }
        if (path.endsWith("/coverage")) {
          return jsonResponse({
            items: [
              {
                roundId: "round_example",
                itemId: "item_1",
                ordinal: 0,
                itemKey: "check-auth",
                subjectKey: "Authentication controls",
                coverage: {
                  status: "violated",
                  requested: ["implementation"],
                  completed: ["implementation"],
                  gaps: [],
                  rationale: "Missing authorization check",
                },
                updatedAt: active.updatedAt,
              },
              {
                roundId: "round_example",
                itemId: "item_2",
                ordinal: 1,
                itemKey: "check-session",
                subjectKey: "Session controls",
                coverage: {
                  status: "inconclusive",
                  requested: ["implementation", "tests"],
                  completed: ["implementation"],
                  gaps: ["tests unavailable"],
                },
                updatedAt: active.updatedAt,
              },
            ],
            page: { hasMore: false },
          });
        }
        throw new Error(`unexpected ${request.method} ${path}`);
      }),
    );
    renderApplication(
      api,
      "/projects/project_example/audits/audit_example/coverage",
    );

    expect(
      await screen.findByRole("heading", { name: "Coverage matrix" }),
    ).toBeVisible();
    const rows = screen.getAllByRole("row");
    expect(within(rows[1]!).getByText("violated")).toBeVisible();
    expect(within(rows[2]!).getByText("inconclusive")).toBeVisible();
    expect(screen.getByText("tests unavailable")).toBeVisible();
    const readsBeforePoll = auditReads;
    await vi.advanceTimersByTimeAsync(1_100);
    await vi.waitFor(() => expect(auditReads).toBeGreaterThan(readsBeforePoll));
    vi.useRealTimers();
  });

  it("recovers a stale pause from the authoritative revision", async () => {
    let current = auditAt("active", 2);
    let pauseRequested = false;
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const path = new URL(request.url).pathname;
        if (path === "/v1/auth/session") return jsonResponse(session);
        if (path === "/v1/projects/project_example") {
          return jsonResponse(project, { headers: { ETag: '"1"' } });
        }
        if (path === "/v1/audits/audit_example/pause") {
          expect(request.headers.get("If-Match")).toBe('"2"');
          pauseRequested = true;
          current = auditAt("paused", 3);
          return jsonResponse(
            {
              code: "precondition_failed",
              message: "Audit revision changed",
              retryable: false,
              requestId: "request_stale",
            },
            { status: 412 },
          );
        }
        if (path === "/v1/audits/audit_example") {
          return jsonResponse(current, {
            headers: { ETag: `"${current.revision}"` },
          });
        }
        throw new Error(`unexpected ${request.method} ${path}`);
      }),
    );
    renderApplication(api, "/projects/project_example/audits/audit_example");
    const user = userEvent.setup();

    await user.click(
      await screen.findByRole("button", { name: "Pause new Runs" }),
    );
    expect(
      within(await screen.findByRole("alert")).getByText(
        "Audit revision changed",
      ),
    ).toBeVisible();
    await vi.waitFor(() =>
      expect(screen.getByRole("button", { name: "Resume" })).toBeVisible(),
    );
    expect(pauseRequested).toBe(true);
    expect(screen.getByText("revision 3")).toBeVisible();
  });
});
