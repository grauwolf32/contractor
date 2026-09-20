import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter } from "react-router";
import { describe, expect, it, vi } from "vitest";

import type { RuntimeConfig } from "../../../config/runtime-config";
import { PublicAPI } from "../../../api/client";
import type {
  Audit,
  AuditCoverageRow,
  AuditFinding,
  AuditItem,
  AuditProfile,
  AuditReviewRequest,
} from "../../../api/audits";
import { Application } from "../../../app/application";
import * as queryClientFactory from "../../../app/query-client";
import { applicationRoutes } from "../../../app/router";
import { queryKeys } from "../../../api/query-keys";

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
    name: "owasp-top10-2025-source-risk",
    version: "1",
    digest: `sha256:${"2".repeat(64)}`,
  },
  mode: "risk-assessment",
  standards: [{ scheme: "owasp-web-top10", version: "2025" }],
  inputs: { source: { required: true, mediaTypes: ["application/zip"] } },
  inventory: {
    implementation: "standard-mappings@1",
    itemWorkflowRole: "check",
  },
  execution: {
    roundMode: "fixed-barrier",
    maxRounds: 1,
    batchSize: 1,
    maxItemsPerRound: 10,
    maxItemsTotal: 10,
    maxSubmittedRuns: 30,
    maxItemRunAttempts: 3,
    deadlineSeconds: 600,
    maxEvidenceBytes: 1_048_576,
    incompleteRound: "assess-with-gaps",
  },
  interaction: {
    activeChecks: "prohibited",
    findingConfirmation: "human-required",
    notApplicable: "human-required",
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
      source: {
        ref: { ...sourceArtifact.artifact },
        digest: `sha256:${"1".repeat(64)}`,
        mediaType: sourceArtifact.mediaType,
        sizeBytes: sourceArtifact.size,
      },
    },
    scope: { objective: "Review source controls" },
    runtimeLabels: ["debug"],
    state,
    phase: state === "draft" ? "not-started" : "rounds",
    ...(state === "draft" ? {} : { currentRoundId: "round_example" }),
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

function top10Baseline(audit: Audit): NonNullable<Audit["baseline"]> {
  const exactStandard = {
    artifact: {
      namespace: "audit-audit_example",
      name: "standard-owasp-web-top10-2025",
      revision: "standard-r1",
    },
    digest: `sha256:${"c".repeat(64)}`,
    mediaType: "application/vnd.contractor.audit-standard+zip" as const,
    sizeBytes: 4096,
  };
  return {
    inputs: audit.inputs,
    scope: audit.scope,
    runtimeLabels: [],
    runtimeConfig: {
      default: {
        label: "default",
        explicit: false,
        bindingRevision: 1,
        config: {
          name: "default-runtime",
          version: "1",
          digest: `sha256:${"d".repeat(64)}`,
        },
      },
      labels: [],
    },
    skills: [],
    standards: [
      {
        reference: { scheme: "owasp-web-top10", version: "2025" },
        title: "OWASP Top 10:2025",
        source: {
          name: "OWASP Top 10:2025",
          url: "https://owasp.org/Top10/2025/",
          revision: "66ebc4798d2ca72973967a20264bdeb70dcf0a13",
        },
        license: {
          id: "CC-BY-SA-4.0",
          url: "https://creativecommons.org/licenses/by-sa/4.0/",
          attribution: "OWASP Foundation, OWASP Top 10:2025.",
          disclosure: "full",
        },
        catalog: {
          ...exactStandard,
          artifact: {
            namespace: "audit-standards",
            name: "std-owasp-web-top10-2025",
            revision: "catalog-r1",
          },
        },
        retained: exactStandard,
      },
    ],
    inventory: {
      sourceContentDigest: exactStandard.digest,
      canonicalInventoryDigest: `sha256:${"e".repeat(64)}`,
      gaps: [],
      worklist: {
        ref: {
          namespace: "audit-audit_example",
          name: "round-1-worklist",
          revision: "worklist-r1",
        },
        digest: `sha256:${"f".repeat(64)}`,
        mediaType: "application/zip",
        sizeBytes: 2048,
      },
    },
  };
}

function findingAt(
  state: AuditFinding["state"],
  revision: number,
): AuditFinding {
  return {
    findingId: "finding_example",
    auditId: "audit_example",
    state,
    firstProposal: {
      receiptId: "receipt_example",
      proposalId: "proposal_example",
      requestDigest: `sha256:${"3".repeat(64)}`,
      clientKey: "candidate-authz",
      proposal: {
        ref: {
          namespace: "audit-findings",
          name: "candidate-authz",
          revision: "proposal-r1",
        },
        digest: `sha256:${"4".repeat(64)}`,
        mediaType: "application/json",
        sizeBytes: 512,
      },
      document: {
        schema: "contractor.audit.finding-proposal.v1",
        client_key: "candidate-authz",
        title: "Missing object authorization",
        description:
          "The **order endpoint** may read another owner's record.\n\n- Check `ownerId` before reading.",
        subject: { kind: "component", key: "orders" },
        preconditions: [],
        standard_refs: [],
        evidence_ids: [],
        proposed_checks: [],
        severity_suggestion: "high",
        limitations: [],
      },
      evidence: [],
      origin: {
        runId: "run_source",
        stageExecutionId: "stage_source",
        allocationId: "allocation_source",
        invocationId: "invocation_source",
        logicalAgentName: "reviewer",
        workflow: {
          name: "source-review",
          version: "1",
          schemaVersion: "contractor/v1alpha1",
          configurationRef: { name: "source-review", version: "1" },
          closureDigest: `sha256:${"5".repeat(64)}`,
        },
        runDeleted: true,
      },
      retention: "audit-held",
      auditHolds: [],
      createdAt: "2026-09-06T10:00:00Z",
    },
    revision,
    createdAt: "2026-09-06T10:00:00Z",
    updatedAt: "2026-09-06T10:00:00Z",
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

async function openAuditCreateForm({
  profiles = [profile],
  loadArtifacts,
}: {
  profiles?: AuditProfile[];
  loadArtifacts: (url: URL) => Response | Promise<Response>;
}) {
  const api = new PublicAPI(
    runtimeConfig,
    vi.fn(async (input) => {
      const request = input instanceof Request ? input : new Request(input);
      const url = new URL(request.url);
      if (url.pathname === "/v1/auth/session") return jsonResponse(session);
      if (url.pathname === "/v1/projects/project_example") {
        return jsonResponse(project, { headers: { ETag: '"1"' } });
      }
      if (url.pathname === "/v1/projects/project_example/audits") {
        return jsonResponse({ items: [], page: { hasMore: false } });
      }
      if (url.pathname === "/v1/projects/project_example/artifacts") {
        return loadArtifacts(url);
      }
      if (url.pathname === "/v1/audit-profiles") {
        return jsonResponse({ items: profiles, page: { hasMore: false } });
      }
      const exactProfile = profiles.find(
        (candidate) =>
          url.pathname ===
          `/v1/audit-profiles/${candidate.ref.name}/versions/${candidate.ref.version}`,
      );
      if (exactProfile !== undefined) {
        return jsonResponse(exactProfile, {
          headers: { ETag: `"${exactProfile.ref.digest}"` },
        });
      }
      throw new Error(`unexpected ${request.method} ${url.pathname}`);
    }),
  );
  renderApplication(api, "/projects/project_example/audits");
  const user = userEvent.setup();
  await user.click(await screen.findByRole("button", { name: "New Audit" }));
  await screen.findByLabelText("Input source");
  return user;
}

describe("Project Audit routes", () => {
  it.each(["checks", "report"] as const)(
    "refreshes the final %s projection when the parent stops polling",
    async (section) => {
      let current = auditAt("active", 2);
      const queryClient = queryClientFactory.createApplicationQueryClient();
      vi.spyOn(
        queryClientFactory,
        "createApplicationQueryClient",
      ).mockReturnValue(queryClient);
      const item: AuditItem = {
        itemId: "item_final",
        roundId: "round_example",
        itemKey: "check_final",
        ordinal: 0,
        kind: "check",
        subjectKey: "Final retained check",
        task: current.inputs.source!,
        origin: {
          schema: "contractor.audit.item-origin.v1",
          entryKey: "check_final",
          sourceRef: current.inputs.source!.ref,
          sourceContentDigest: current.inputs.source!.digest,
          sourceMediaType: "application/zip",
          canonicalInventoryDigest: `sha256:${"c".repeat(64)}`,
        },
        workflowRole: "check",
        state: "settled",
        approvalKind: "none",
        attempts: [],
        createdAt: current.createdAt,
        updatedAt: current.updatedAt,
      };
      const reads = vi.fn();
      const api = new PublicAPI(
        runtimeConfig,
        vi.fn(async (input) => {
          const path = new URL((input as Request).url).pathname;
          if (path === "/v1/auth/session") return jsonResponse(session);
          if (path === "/v1/projects/project_example")
            return jsonResponse(project, { headers: { ETag: '"1"' } });
          if (path === "/v1/audits/audit_example")
            return jsonResponse(current, {
              headers: { ETag: `"${current.revision}"` },
            });
          if (path.endsWith("/items")) {
            reads();
            return jsonResponse({
              items: current.state === "active" ? [] : [item],
              page: { hasMore: false },
            });
          }
          if (path.endsWith("/report")) {
            reads();
            return jsonResponse(
              current.state === "active"
                ? { status: "pending" }
                : {
                    status: "ready",
                    summary: "Final retained report",
                    summaryArtifact: {
                      ref: {
                        namespace: "audit-example",
                        name: "report.md",
                        revision: "report-r1",
                      },
                      digest: `sha256:${"1".repeat(64)}`,
                      mediaType: "text/markdown",
                      sizeBytes: 21,
                    },
                  },
            );
          }
          return jsonResponse({ items: [], page: { hasMore: false } });
        }),
      );
      renderApplication(
        api,
        `/projects/project_example/audits/audit_example/${section}`,
      );
      await screen.findByText(
        section === "checks"
          ? "No checks materialized"
          : /The Audit has not reached report generation/,
      );
      current = auditAt("completed", 3);
      act(() =>
        queryClient.setQueryData(
          queryKeys.audits.detail(current.auditId),
          current,
        ),
      );
      await screen.findByText(
        section === "checks" ? "Final retained check" : "Final retained report",
      );
      expect(reads).toHaveBeenCalledTimes(2);
    },
  );

  it.each(["text/markdown", "text/plain"])(
    "only previews and downloads current Markdown reports: %s",
    async (mediaType) => {
      const markdown = mediaType === "text/markdown";
      const summary =
        "# Coverage summary\n\n| Status | Count |\n| --- | ---: |\n| satisfied | 2 |\n";
      const api = new PublicAPI(
        runtimeConfig,
        vi.fn(async (input) => {
          const request = input instanceof Request ? input : new Request(input);
          const path = new URL(request.url).pathname;
          if (path === "/v1/auth/session") return jsonResponse(session);
          if (path === "/v1/projects/project_example")
            return jsonResponse(project, { headers: { ETag: '"1"' } });
          if (path === "/v1/audits/audit_example")
            return jsonResponse(auditAt("completed", 3), {
              headers: { ETag: '"3"' },
            });
          if (path === "/v1/audits/audit_example/report")
            return jsonResponse({
              status: "ready",
              summary,
              summaryArtifact: {
                ref: {
                  namespace: "audit-example",
                  name: markdown ? "report.md" : "report.txt",
                  revision: "report-r1",
                },
                digest: `sha256:${"1".repeat(64)}`,
                mediaType,
                sizeBytes: summary.length,
              },
            });
          if (path.endsWith("/coverage") || path.endsWith("/reviews"))
            return jsonResponse({ items: [], page: { hasMore: false } });
          throw new Error(`unexpected ${request.method} ${path}`);
        }),
      );
      renderApplication(
        api,
        "/projects/project_example/audits/audit_example/report",
      );
      if (!markdown) {
        expect(
          await screen.findByText("Server returned an invalid Audit response"),
        ).toBeVisible();
        expect(
          screen.queryByRole("button", { name: "Download exact summary" }),
        ).not.toBeInTheDocument();
        expect(
          screen.queryByRole("heading", { name: "Coverage summary" }),
        ).not.toBeInTheDocument();
        return;
      }
      const button = await screen.findByRole("button", {
        name: "Download exact summary",
      });
      expect(
        await screen.findByRole("heading", { name: "Coverage summary" }),
      ).toBeVisible();
      expect(screen.getByRole("table")).toHaveTextContent("satisfied");
      let downloadedName = "";
      const click = vi
        .spyOn(HTMLAnchorElement.prototype, "click")
        .mockImplementation(function (this: HTMLAnchorElement) {
          downloadedName = this.download;
        });
      const createURL = vi.fn<(blob: Blob) => string>(() => "blob:report-test");
      const originalCreate = URL.createObjectURL;
      const originalRevoke = URL.revokeObjectURL;
      URL.createObjectURL = createURL;
      URL.revokeObjectURL = vi.fn();
      try {
        await userEvent.setup().click(button);
        expect(downloadedName).toBe("audit_example-report.md");
        expect(createURL.mock.calls[0]?.[0].type).toBe(mediaType);
      } finally {
        click.mockRestore();
        URL.createObjectURL = originalCreate;
        URL.revokeObjectURL = originalRevoke;
      }
    },
  );

  it("creates a draft with the automatically selected unique compatible Project Artifact", async () => {
    const requests: Request[] = [];
    const draft = auditAt("draft", 1);
    const selectedProfile: AuditProfile = {
      ...profile,
      inventory: {
        ...profile.inventory,
        standardSelection: {
          scope: "ASVS 5.0 Level 1 source pilot",
          levels: ["1"],
          entryIds: ["v5.0.0-1.2.4", "v5.0.0-1.2.5"],
        },
      },
    };
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
          return jsonResponse({
            items: [selectedProfile],
            page: { hasMore: false },
          });
        }
        if (
          path === "/v1/audit-profiles/owasp-top10-2025-source-risk/versions/1"
        ) {
          return jsonResponse(selectedProfile, {
            headers: { ETag: `"${selectedProfile.ref.digest}"` },
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
        if (path.endsWith("/coverage") || path.endsWith("/reviews"))
          return jsonResponse({ items: [], page: { hasMore: false } });
        throw new Error(`unexpected ${request.method} ${path}`);
      }),
    );
    const { router } = renderApplication(
      api,
      "/projects/project_example/audits",
    );
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: "New Audit" }));
    await screen.findByRole("heading", { name: "New Audit" });
    const createButton = screen.getByRole("button", {
      name: "Create Audit draft",
    });
    expect(
      await screen.findByText(
        "Exact standards pinned at start: owasp-web-top10@2025",
      ),
    ).toBeVisible();
    expect(
      await screen.findByTestId("audit-profile-standard-selection"),
    ).toHaveTextContent("ASVS 5.0 Level 1 source pilot");
    expect(
      screen.getByTestId("audit-profile-standard-selection"),
    ).toHaveTextContent("2 exact requirements");
    await waitFor(() =>
      expect(screen.getByLabelText("Input source")).toHaveValue(
        JSON.stringify(sourceArtifact.artifact),
      ),
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
      await screen.findByRole("heading", {
        name: "OWASP Top 10 · Source risks",
      }),
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
      profile: { name: "owasp-top10-2025-source-risk", version: "1" },
      inputs: { source: sourceArtifact.artifact },
      scope: { objective: "Map attack surface" },
    });
  });

  it("selects the highest loaded numeric profile version and preserves an explicit older version", async () => {
    const older = { ...profile, ref: { ...profile.ref, version: "2" } };
    const latest = { ...profile, ref: { ...profile.ref, version: "10" } };
    const user = await openAuditCreateForm({
      profiles: [older, latest],
      loadArtifacts: () =>
        jsonResponse({ items: [sourceArtifact], page: { hasMore: false } }),
    });
    const select = screen.getByLabelText("Exact Audit profile");
    expect(select).toHaveValue(JSON.stringify([profile.ref.name, "10"]));
    await user.selectOptions(select, JSON.stringify([profile.ref.name, "2"]));
    await user.type(
      screen.getByLabelText("Objective"),
      "Use the retained contract",
    );
    expect(select).toHaveValue(JSON.stringify([profile.ref.name, "2"]));
  });

  it.each(["application/zip", "text/plain"])(
    "waits for all Project pages before selecting when a later artifact is %s",
    async (mediaType) => {
      const laterArtifact = {
        ...sourceArtifact,
        artifact: { ...sourceArtifact.artifact, name: "another-artifact" },
        mediaType,
      };
      let resolveLaterPage!: (response: Response) => void;
      const laterPage = new Promise<Response>((resolve) => {
        resolveLaterPage = resolve;
      });
      const loadArtifacts = vi.fn((url: URL) =>
        url.searchParams.get("cursor") === "later-page"
          ? laterPage
          : jsonResponse({
              items: [sourceArtifact],
              page: { hasMore: true, nextCursor: "later-page" },
            }),
      );
      const user = await openAuditCreateForm({ loadArtifacts });
      const source = screen.getByLabelText("Input source");
      const create = screen.getByRole("button", { name: "Create Audit draft" });
      await waitFor(() => expect(loadArtifacts).toHaveBeenCalledTimes(2));
      expect(source).toHaveValue("");
      expect(create).toBeDisabled();
      resolveLaterPage(
        jsonResponse({ items: [laterArtifact], page: { hasMore: false } }),
      );
      await waitFor(() =>
        expect(screen.queryByText("Loading Project Artifacts…")).toBeNull(),
      );
      if (mediaType === "application/zip") {
        expect(source).toHaveValue("");
        expect(create).toBeDisabled();
        await user.selectOptions(
          source,
          JSON.stringify(laterArtifact.artifact),
        );
        await user.type(screen.getByLabelText("Objective"), "Manual choice");
        expect(source).toHaveValue(JSON.stringify(laterArtifact.artifact));
        expect(create).toBeEnabled();
      } else {
        expect(source).toHaveValue(JSON.stringify(sourceArtifact.artifact));
        expect(create).toBeEnabled();
        await user.selectOptions(source, "");
        await user.type(screen.getByLabelText("Objective"), "Choose later");
        expect(source).toHaveValue("");
        expect(create).toBeDisabled();
      }
    },
  );

  it("matches each input of a newly selected profile and respects an optional input cleared by the user", async () => {
    const otherProfile: AuditProfile = {
      ...profile,
      ref: { ...profile.ref, name: "custom-audit" },
      inputs: {
        source: { required: true, mediaTypes: ["application/json"] },
        notes: { required: false, mediaTypes: ["text/*"] },
        attachment: { required: false, mediaTypes: ["*/*"] },
        diagram: { required: false, mediaTypes: ["image/png"] },
      },
    };
    const jsonArtifact = {
      ...sourceArtifact,
      artifact: { ...sourceArtifact.artifact, name: "api-schema" },
      mediaType: "application/json",
    };
    const notesArtifact = {
      ...sourceArtifact,
      artifact: { ...sourceArtifact.artifact, name: "notes" },
      mediaType: "text/markdown",
    };
    const user = await openAuditCreateForm({
      profiles: [profile, otherProfile],
      loadArtifacts: () =>
        jsonResponse({
          items: [sourceArtifact, jsonArtifact, notesArtifact],
          page: { hasMore: false },
        }),
    });
    await waitFor(() =>
      expect(screen.getByLabelText("Input source")).toHaveValue(
        JSON.stringify(sourceArtifact.artifact),
      ),
    );
    await user.selectOptions(
      screen.getByLabelText("Exact Audit profile"),
      JSON.stringify([otherProfile.ref.name, otherProfile.ref.version]),
    );
    await waitFor(() =>
      expect(screen.getByLabelText("Input source")).toHaveValue(
        JSON.stringify(jsonArtifact.artifact),
      ),
    );
    expect(screen.getByLabelText("Input notes")).toHaveValue(
      JSON.stringify(notesArtifact.artifact),
    );
    expect(screen.getByLabelText("Input attachment")).toHaveValue("");
    expect(screen.getByLabelText("Input diagram")).toHaveValue("");
    await user.selectOptions(screen.getByLabelText("Input notes"), "");
    await user.type(screen.getByLabelText("Objective"), "Review schema");
    expect(screen.getByLabelText("Input notes")).toHaveValue("");
    expect(
      screen.getByRole("button", { name: "Create Audit draft" }),
    ).toBeEnabled();
  });

  it("does not treat a partial Project inventory as unique when a later page fails", async () => {
    let retry = false;
    const loadArtifacts = vi.fn((url: URL) => {
      if (url.searchParams.get("cursor") === "later-page") {
        return retry
          ? jsonResponse({ items: [], page: { hasMore: false } })
          : jsonResponse(
              { code: "unavailable", message: "Artifact listing unavailable" },
              { status: 503 },
            );
      }
      return jsonResponse({
        items: [sourceArtifact],
        page: { hasMore: true, nextCursor: "later-page" },
      });
    });
    const user = await openAuditCreateForm({ loadArtifacts });
    const retryButton = await screen.findByRole("button", {
      name: "Retry loading Project Artifacts",
    });
    expect(screen.getByLabelText("Input source")).toHaveValue("");
    expect(
      screen.getByRole("button", { name: "Create Audit draft" }),
    ).toBeDisabled();
    expect(loadArtifacts).toHaveBeenCalledTimes(2);
    retry = true;
    await user.click(retryButton);
    await waitFor(() =>
      expect(screen.getByLabelText("Input source")).toHaveValue(
        JSON.stringify(sourceArtifact.artifact),
      ),
    );
    expect(loadArtifacts).toHaveBeenCalledTimes(3);
  });

  it("shows the exact retained standard identity on the Audit baseline", async () => {
    const current = auditAt("completed", 3);
    current.baseline = top10Baseline(current);
    current.baseline.inventory!.standardSelection = {
      scope: "ASVS 5.0 Level 1 source pilot",
      levels: ["1"],
      entryIds: ["v5.0.0-1.2.4", "v5.0.0-2.1.1"],
    };
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
          return jsonResponse(current, { headers: { ETag: '"3"' } });
        }
        if (path.endsWith("/coverage") || path.endsWith("/reviews"))
          return jsonResponse({ items: [], page: { hasMore: false } });
        throw new Error(`unexpected ${request.method} ${path}`);
      }),
    );
    renderApplication(api, "/projects/project_example/audits/audit_example");

    await userEvent
      .setup()
      .click(await screen.findByText("Baseline and exact standards"));
    const standards = await screen.findByTestId("audit-baseline-standards");
    expect(within(standards).getByText("OWASP Top 10:2025")).toBeVisible();
    expect(within(standards).getByText("owasp-web-top10@2025")).toBeVisible();
    expect(standards).toHaveTextContent("CC-BY-SA-4.0");
    expect(
      within(standards).getByRole("link", { name: "source" }),
    ).toHaveAttribute("href", "https://owasp.org/Top10/2025/");
    const selection = screen.getByTestId("audit-baseline-standard-selection");
    expect(selection).toHaveTextContent("ASVS 5.0 Level 1 source pilot");
    expect(selection).toHaveTextContent("v5.0.0-1.2.4");
    expect(selection).toHaveTextContent("v5.0.0-2.1.1");
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
        if (path.endsWith("/coverage") || path.endsWith("/reviews"))
          return jsonResponse({ items: [], page: { hasMore: false } });
        throw new Error(`unexpected ${request.method} ${path}`);
      }),
    );
    renderApplication(
      api,
      "/projects/project_example/audits/audit_example/coverage",
    );

    expect(
      await screen.findByRole("heading", { name: "Coverage and results" }),
    ).toBeVisible();
    const rows = screen.getAllByRole("article");
    expect(within(rows[0]!).getByText("Issue found")).toBeVisible();
    expect(within(rows[1]!).getByText("Inconclusive")).toBeVisible();
    expect(
      await screen.findByText("Missing authorization check"),
    ).toBeVisible();
    fireEvent.click(within(rows[1]!).getByText("Read task and result"));
    expect(screen.getByText("tests unavailable")).toBeVisible();
    const readsBeforePoll = auditReads;
    await vi.advanceTimersByTimeAsync(1_100);
    await vi.waitFor(() => expect(auditReads).toBeGreaterThan(readsBeforePoll));
    vi.useRealTimers();
  });

  it("loads every coverage page and searches the actual task, conclusion and evidence", async () => {
    const completed = auditAt("completed", 4);
    completed.currentRoundId = "round_example";
    const cursors: Array<string | null> = [];
    const row = (
      itemId: string,
      status: AuditCoverageRow["coverage"]["status"],
    ): AuditCoverageRow => ({
      roundId: "round_example",
      itemId,
      ordinal: 0,
      itemKey: itemId,
      subjectKey: itemId,
      coverage: {
        status,
        requested: ["observation"],
        completed: ["observation"],
        gaps: [],
      },
      updatedAt: completed.updatedAt,
      details: {
        objective: "Verify that expired invitations cannot be reused.",
        methods: ["custom-method"],
        taskDocument: {
          schema: "contractor.audit.item-task.v1",
          checklist: {
            statement: "Verify that expired invitations cannot be reused.",
          },
        },
        resultSummary: "An expired invitation can still be accepted.",
        evidence: [
          {
            id: "proof",
            kind: "observation",
            summary: "The invitation endpoint accepts an expired token.",
          },
        ],
      },
    });
    const first = row("custom-check", "violated");
    const second = {
      ...row("trace-check", "traced-complete"),
      details: undefined,
    };
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const url = new URL(request.url);
        if (url.pathname === "/v1/auth/session") return jsonResponse(session);
        if (url.pathname === "/v1/projects/project_example")
          return jsonResponse(project, { headers: { ETag: '"1"' } });
        if (url.pathname === "/v1/audits/audit_example")
          return jsonResponse(completed, { headers: { ETag: '"4"' } });
        if (url.pathname.endsWith("/coverage")) {
          expect(url.searchParams.get("round")).toBe("round_example");
          const cursor = url.searchParams.get("cursor");
          cursors.push(cursor);
          return jsonResponse(
            cursor === null
              ? {
                  items: [first],
                  page: { hasMore: true, nextCursor: "next-check" },
                }
              : { items: [second], page: { hasMore: false } },
          );
        }
        throw new Error(`unexpected ${url.pathname}`);
      }),
    );
    const { router } = renderApplication(
      api,
      "/projects/project_example/audits/audit_example/coverage",
    );
    const user = userEvent.setup();
    await user.click(
      within(
        await screen.findByRole("article", { name: "custom-check" }),
      ).getByText("Read task and result"),
    );
    expect(
      await screen.findByText(
        "Verify that expired invitations cannot be reused.",
      ),
    ).toBeVisible();
    expect(
      within(
        screen
          .getByRole("article", { name: "custom-check" })
          .querySelector(".audit-result-reading") as HTMLElement,
      ).getByText("An expired invitation can still be accepted."),
    ).toBeVisible();
    expect(screen.getByText("Showing 2 of 2 checks")).toBeVisible();
    expect(cursors).toEqual([null, "next-check"]);
    await user.click(screen.getByRole("button", { name: /Issues found/u }));
    expect(screen.getByText("Showing 1 of 2 checks")).toBeVisible();
    expect(screen.queryByRole("article", { name: "trace-check" })).toBeNull();
    expect(router.state.location.search).toContain("result=issues");
    await user.click(screen.getByRole("button", { name: "Clear filters" }));
    await user.type(
      screen.getByRole("searchbox", { name: "Search checks" }),
      "endpoint",
    );
    expect(screen.getByRole("article", { name: "custom-check" })).toBeVisible();
    expect(screen.queryByRole("article", { name: "trace-check" })).toBeNull();
    await user.click(screen.getByText("Full task & evidence (1)"));
    expect(
      await screen.findByText(
        "The invitation endpoint accepts an expired token.",
      ),
    ).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Clear filters" }));
    expect(screen.getByText("Fully traced")).toBeVisible();
    expect(
      screen
        .getAllByText(
          "All requested parts of this operation were traced. This is not a security verdict.",
        )
        .some((element) => element.classList.contains("audit-result-excerpt")),
    ).toBe(true);
    expect(
      within(
        screen.getByRole("navigation", { name: "Audit sections" }),
      ).getByRole("link", { name: "Findings" }),
    ).toHaveAttribute(
      "href",
      "/projects/project_example/audits/audit_example/findings",
    );
  });

  it("shows audit cards and deletion in the Project Audits section", async () => {
    const completed = auditAt("completed", 4);
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const path = new URL(request.url).pathname;
        if (path === "/v1/auth/session") return jsonResponse(session);
        if (path === "/v1/projects/project_example")
          return jsonResponse(project, { headers: { ETag: '"1"' } });
        if (path === "/v1/projects/project_example/audits")
          return jsonResponse({ items: [completed], page: { hasMore: false } });
        return jsonResponse({ items: [], page: { hasMore: false } });
      }),
    );
    renderApplication(api, "/projects/project_example/audits");
    const user = userEvent.setup();
    expect(
      await screen.findByRole("link", { name: "View checks & results →" }),
    ).toHaveAttribute(
      "href",
      "/projects/project_example/audits/audit_example/coverage",
    );
    expect(
      screen.getByRole("heading", { name: "OWASP Top 10 · Source risks" }),
    ).toBeVisible();
    expect(screen.getByRole("button", { name: "New Audit" })).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Delete Audit" }));
    expect(
      screen.getByRole("alertdialog", { name: "Delete this Audit?" }),
    ).toBeVisible();
    await user.click(
      screen.getByRole("button", { name: "Keep Audit unchanged" }),
    );
    expect(screen.queryByRole("alertdialog")).toBeNull();
  });

  it("combines every audit and finding page, filters results, and keeps duplicate reviews within their audit", async () => {
    const firstAudit = auditAt("paused", 3);
    const otherAudit = {
      ...firstAudit,
      auditId: "audit_trace",
      profile: { ...profile.ref, name: "openapi-operation-trace" },
    };
    const first = findingAt("proposed", 1);
    const sibling = {
      ...findingAt("proposed", 1),
      findingId: "finding_sibling",
    };
    sibling.firstProposal.document.title = "Missing rate limit";
    sibling.firstProposal.document.severity_suggestion = "medium";
    const other = { ...findingAt("proposed", 1), auditId: otherAudit.auditId };
    other.firstProposal.document.title = "Trace information exposure";
    other.firstProposal.document.severity_suggestion = "low";
    const review: AuditReviewRequest = {
      requestId: "review_duplicate",
      auditId: firstAudit.auditId,
      findingId: first.findingId,
      subjectId: first.findingId,
      subjectKind: "finding",
      kind: "finding-triage",
      subjectRevision: first.revision,
      subjectDigest: `sha256:${"6".repeat(64)}`,
      requestedActions: [
        "true_positive",
        "false_positive",
        "duplicate",
        "reopen",
        "needs_evidence",
      ],
      state: "pending",
      revision: 1,
      createdAt: firstAudit.createdAt,
      updatedAt: firstAudit.updatedAt,
    };
    const requested: string[] = [];
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const url = new URL(request.url);
        const path = url.pathname;
        expect(request.method).toBe("GET");
        requested.push(`${path}:${url.searchParams.get("cursor") ?? "first"}`);
        if (path === "/v1/auth/session") return jsonResponse(session);
        if (path === "/v1/projects/project_example")
          return jsonResponse(project, { headers: { ETag: '"1"' } });
        if (path === "/v1/projects/project_example/audits")
          return jsonResponse(
            url.searchParams.has("cursor")
              ? { items: [otherAudit], page: { hasMore: false } }
              : {
                  items: [firstAudit],
                  page: { hasMore: true, nextCursor: "other-audits" },
                },
          );
        if (path === "/v1/audits/audit_example/findings")
          return jsonResponse(
            url.searchParams.has("cursor")
              ? { items: [sibling], page: { hasMore: false } }
              : {
                  items: [first],
                  page: { hasMore: true, nextCursor: "more-findings" },
                },
          );
        if (path === "/v1/audits/audit_trace/findings")
          return jsonResponse({ items: [other], page: { hasMore: false } });
        if (path === "/v1/audits/audit_example/reviews")
          return jsonResponse(
            url.searchParams.has("cursor")
              ? { items: [review], page: { hasMore: false } }
              : {
                  items: [],
                  page: { hasMore: true, nextCursor: "more-reviews" },
                },
          );
        if (path === "/v1/audits/audit_trace/reviews")
          return jsonResponse({ items: [], page: { hasMore: false } });
        throw new Error(`unexpected ${path}`);
      }),
    );
    const { container, router } = renderApplication(
      api,
      "/projects/project_example/findings",
    );
    const user = userEvent.setup();
    expect(await screen.findByText("3 of 3 findings · 2 audits")).toBeVisible();
    const filters = within(
      screen.getByRole("region", { name: "Finding filters" }),
    );
    expect(requested).toEqual(
      expect.arrayContaining([
        "/v1/projects/project_example/audits:other-audits",
        "/v1/audits/audit_example/findings:more-findings",
        "/v1/audits/audit_example/reviews:more-reviews",
      ]),
    );
    const cards = container.querySelectorAll(".audit-finding-card");
    expect(cards).toHaveLength(3);
    expect(new Set([...cards].map((card) => card.id)).size).toBe(3);
    const firstCard = within(
      screen
        .getByRole("heading", { name: first.firstProposal.document.title })
        .closest(".audit-finding-card") as HTMLElement,
    );
    await user.selectOptions(
      await firstCard.findByLabelText("Decision"),
      "duplicate",
    );
    expect(
      within(firstCard.getByLabelText("Canonical finding"))
        .getAllByRole("option")
        .map((option) => option.textContent),
    ).toEqual(["Missing rate limit · finding_sibling"]);
    expect(
      screen.getByRole("link", {
        name: "OpenAPI · Operation trace · it_trace",
      }),
    ).toHaveAttribute(
      "href",
      "/projects/project_example/audits/audit_trace/findings",
    );

    await user.selectOptions(filters.getByLabelText("Audit"), "audit_trace");
    expect(screen.getByText("1 of 3 findings · 2 audits")).toBeVisible();
    expect(
      screen.queryByRole("heading", {
        name: first.firstProposal.document.title,
      }),
    ).not.toBeInTheDocument();
    expect(router.state.location.search).toBe("?audit=audit_trace");
    await user.click(screen.getByRole("button", { name: "Clear filters" }));
    await user.selectOptions(filters.getByLabelText("Severity"), "medium");
    expect(
      screen.getByRole("heading", { name: "Missing rate limit" }),
    ).toBeVisible();
    expect(
      screen.queryByRole("heading", { name: "Trace information exposure" }),
    ).not.toBeInTheDocument();
    await user.selectOptions(filters.getByLabelText("Severity"), "");
    await user.type(
      screen.getByLabelText("Search findings"),
      "TRACE INFORMATION",
    );
    expect(screen.getByText("1 of 3 findings · 2 audits")).toBeVisible();
    await user.type(screen.getByLabelText("Search findings"), " missing");
    expect(
      screen.getByRole("heading", { name: "No matching findings" }),
    ).toBeVisible();
  });

  it("keeps available findings visible when another audit fails and retries the missing audit", async () => {
    const firstAudit = auditAt("paused", 3);
    const otherAudit = {
      ...firstAudit,
      auditId: "audit_trace",
      profile: { ...profile.ref, name: "openapi-operation-trace" },
    };
    let failed = true;
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const path = new URL(request.url).pathname;
        if (path === "/v1/auth/session") return jsonResponse(session);
        if (path === "/v1/projects/project_example")
          return jsonResponse(project, { headers: { ETag: '"1"' } });
        if (path === "/v1/projects/project_example/audits")
          return jsonResponse({
            items: [firstAudit, otherAudit],
            page: { hasMore: false },
          });
        if (path === "/v1/audits/audit_example/findings")
          return jsonResponse({
            items: [findingAt("proposed", 1)],
            page: { hasMore: false },
          });
        if (path === "/v1/audits/audit_trace/findings" && failed)
          return jsonResponse(
            {
              code: "unavailable",
              message: "Findings unavailable",
              retryable: true,
              requestId: "request_failed",
            },
            { status: 503 },
          );
        if (
          path.endsWith("/reviews") ||
          path === "/v1/audits/audit_trace/findings"
        )
          return jsonResponse({ items: [], page: { hasMore: false } });
        throw new Error(`unexpected ${path}`);
      }),
    );
    renderApplication(api, "/projects/project_example/findings");
    const user = userEvent.setup();
    expect(
      await screen.findByRole("heading", {
        name: "Missing object authorization",
      }),
    ).toBeVisible();
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Findings unavailable: OpenAPI · Operation trace",
    );
    expect(screen.getByText("1 of 1 findings loaded · 2 audits")).toBeVisible();
    expect(screen.queryByText("No findings yet")).not.toBeInTheDocument();
    failed = false;
    await user.click(screen.getByRole("button", { name: "Retry findings" }));
    expect(await screen.findByText("1 of 1 findings · 2 audits")).toBeVisible();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it.each(["audit", "project"] as const)(
    "reviews a finding with exact revisions from the %s view and renders immutable history",
    async (scope) => {
      const requests: Request[] = [];
      let currentAudit = auditAt("completed", 2);
      let currentFinding = findingAt("proposed", 1);
      currentFinding.firstProposal.document.description +=
        "\n\n" +
        "Retained evidence must remain readable without another action. ".repeat(
          8,
        ) +
        "\n\nThe final paragraph contains the complete remediation context.";
      let reviews: AuditReviewRequest[] = [];
      const pendingReview: AuditReviewRequest = {
        requestId: "review_example",
        auditId: currentAudit.auditId,
        findingId: currentFinding.findingId,
        subjectKind: "finding",
        subjectId: currentFinding.findingId,
        kind: "finding-triage",
        subjectRevision: 1,
        subjectDigest: `sha256:${"6".repeat(64)}`,
        requestedActions: [
          "true_positive",
          "false_positive",
          "duplicate",
          "reopen",
          "needs_evidence",
        ],
        state: "pending",
        revision: 1,
        createdAt: currentAudit.createdAt,
        updatedAt: currentAudit.updatedAt,
      };
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
          if (path === "/v1/projects/project_example/audits") {
            return jsonResponse({
              items: [currentAudit],
              page: { hasMore: false },
            });
          }
          if (path === "/v1/audits/audit_example") {
            return jsonResponse(currentAudit, {
              headers: { ETag: `"${currentAudit.revision}"` },
            });
          }
          if (path === "/v1/audits/audit_example/findings") {
            return jsonResponse({
              items: [currentFinding],
              page: { hasMore: false },
            });
          }
          if (path === "/v1/audits/audit_example/reviews") {
            return jsonResponse({ items: reviews, page: { hasMore: false } });
          }
          if (
            path ===
            "/v1/audits/audit_example/findings/finding_example/provenance"
          ) {
            return jsonResponse({
              auditRevision: currentAudit.revision,
              findingRevision: currentFinding.revision,
              items: [
                {
                  recordId: "attempt:receipt_example:execution_item_example",
                  kind: "check-attempt",
                  receiptId: "receipt_example",
                  relation: "verification",
                  proposal: currentFinding.firstProposal.proposal,
                  origin: currentFinding.firstProposal.origin,
                  supportsCurrentAssessment: true,
                  createdAt: currentAudit.createdAt,
                  assessment: {
                    assessmentId: "assessment_example",
                    semanticAssessment: "supported",
                    result: {
                      ref: {
                        namespace: "audit-results",
                        name: "check-one",
                        revision: "result-r2",
                      },
                      digest: `sha256:${"8".repeat(64)}`,
                    },
                    receiptId: "receipt_example",
                    directVerification: false,
                    acceptedAt: currentAudit.createdAt,
                  },
                  attempt: {
                    executionItemId: "execution_item_example",
                    executionId: "execution_example",
                    itemId: "item_example",
                    itemAttempt: 2,
                    role: "check",
                    workflowRole: "authorization-check",
                    state: "settled",
                    collectionDisposition: "accepted-result",
                    terminalOutcome: "succeeded",
                    runId: "run_verification",
                    runDeleted: true,
                    runProvenance: {
                      schema: "contractor.audit.run-provenance.v1",
                      runId: "run_verification",
                      workflow: {
                        name: "verify-authorization",
                        version: "1",
                        schemaVersion: "contractor/v1alpha1",
                        configurationRef: {
                          name: "verify-authorization",
                          version: "1",
                        },
                        closureDigest: `sha256:${"9".repeat(64)}`,
                      },
                    },
                    task: {
                      ref: {
                        namespace: "audit-task-packages",
                        name: "check-one",
                        revision: "task-r1",
                      },
                      digest: `sha256:${"a".repeat(64)}`,
                    },
                    itemOrigin: {
                      schema: "contractor.audit.item-origin.v1",
                      entryKey: "check-one",
                      sourceRef: {
                        namespace: "inputs",
                        name: "checks",
                        revision: "checks-r1",
                      },
                      sourceContentDigest: `sha256:${"b".repeat(64)}`,
                      sourceMediaType: "application/json",
                      canonicalInventoryDigest: `sha256:${"c".repeat(64)}`,
                    },
                    result: {
                      ref: {
                        namespace: "audit-results",
                        name: "check-one",
                        revision: "result-r2",
                      },
                      digest: `sha256:${"8".repeat(64)}`,
                    },
                    createdAt: currentAudit.createdAt,
                    collectedAt: currentAudit.updatedAt,
                  },
                },
              ],
              page: { hasMore: false },
            });
          }
          if (
            path ===
              "/v1/audits/audit_example/findings/finding_example/reviews" &&
            request.method === "POST"
          ) {
            expect(request.headers.get("If-Match")).toBe('"1"');
            reviews = [pendingReview];
            currentAudit = auditAt("completed", 3);
            return jsonResponse(pendingReview, {
              status: 201,
              headers: { ETag: '"1"' },
            });
          }
          if (
            path ===
              "/v1/audits/audit_example/reviews/review_example/decisions" &&
            request.method === "POST"
          ) {
            expect(request.headers.get("If-Match")).toBe('"1"');
            const body = (await request.json()) as Record<string, unknown>;
            expect(body).toEqual({
              verdict: "true_positive",
              severity: "high",
              rationale: "Confirmed from exact source evidence.",
            });
            const decision = {
              decisionId: "decision_example",
              requestId: pendingReview.requestId,
              auditId: currentAudit.auditId,
              findingId: currentFinding.findingId,
              actorId: session.principal.userId,
              verdict: "true_positive" as const,
              severity: "high" as const,
              rationale: "Confirmed from exact source evidence.",
              subjectRevision: 1,
              subjectDigest: pendingReview.subjectDigest,
              createdAt: currentAudit.updatedAt,
            };
            currentFinding = {
              ...currentFinding,
              state: "confirmed",
              revision: 2,
              analystVerdict: "true_positive",
              analystSeverity: "high",
              analystDecision: decision,
            };
            reviews = [
              { ...pendingReview, state: "decided", revision: 2, decision },
            ];
            currentAudit = auditAt("completed", 4);
            return jsonResponse({
              finding: currentFinding,
              request: reviews[0],
              decision,
              replayed: false,
            });
          }
          if (path.endsWith("/coverage") || path.endsWith("/reviews"))
            return jsonResponse({ items: [], page: { hasMore: false } });
          throw new Error(`unexpected ${request.method} ${path}`);
        }),
      );
      renderApplication(
        api,
        scope === "audit"
          ? "/projects/project_example/audits/audit_example/findings"
          : "/projects/project_example/findings",
      );
      const user = userEvent.setup();

      expect(
        await screen.findByRole("heading", {
          name: "Missing object authorization",
        }),
      ).toBeVisible();
      expect(screen.getByText("Unreviewed", { selector: "dd" })).toBeVisible();
      expect(
        await screen.findByText(
          "The final paragraph contains the complete remediation context.",
        ),
      ).toBeVisible();
      expect(
        await screen.findByText("order endpoint", { selector: "strong" }),
      ).toBeVisible();
      expect(screen.getByText("ownerId", { selector: "code" })).toBeVisible();
      const sourceLink = within(
        screen.getByRole("region", { name: "Source artifacts" }),
      ).getByRole("link");
      expect(sourceLink).toHaveAttribute(
        "href",
        `/projects/project_example/artifacts/sources/${sourceArtifact.artifact.name}?revision=${sourceArtifact.artifact.revision}`,
      );

      await user.click(screen.getByRole("button", { name: "Show provenance" }));
      expect(
        await screen.findByText(
          "attempt 2 · check/authorization-check · settled",
        ),
      ).toBeVisible();
      expect(screen.getByText("verify-authorization@1")).toBeVisible();
      expect(screen.getByText("inventory entry check-one")).toBeVisible();
      await user.click(screen.getByRole("button", { name: "Review finding" }));
      const findingCard = within(
        screen
          .getByRole("heading", { name: "Missing object authorization" })
          .closest(".audit-finding-card") as HTMLElement,
      );
      await user.selectOptions(
        await findingCard.findByLabelText("Severity"),
        "high",
      );
      await user.type(
        screen.getByLabelText("Analyst rationale"),
        "Confirmed from exact source evidence.",
      );
      await user.click(screen.getByRole("button", { name: "Record decision" }));

      expect(await screen.findByText("true_positive · high")).toBeVisible();
      const mutationRequests = requests.filter(
        (request) => request.method === "POST",
      );
      expect(mutationRequests).toHaveLength(2);
      expect(mutationRequests[0]?.headers.get("Idempotency-Key")).toMatch(
        /^audit-finding-review-ui-/u,
      );
      expect(mutationRequests[1]?.headers.get("Idempotency-Key")).toMatch(
        /^audit-finding-review-ui-/u,
      );

      if (scope === "project") {
        await user.click(
          screen.getByRole("link", {
            name: "OWASP Top 10 · Source risks · _example",
          }),
        );
      }
      await user.click(await screen.findByRole("link", { name: "Reviews" }));
      expect(
        await screen.findByRole("heading", { name: "Human reviews" }),
      ).toBeVisible();
      expect(
        screen.getByText("Confirmed from exact source evidence."),
      ).toBeVisible();
    },
  );

  it("approves an exact non-finding review without treating model text as authority", async () => {
    let currentAudit = auditAt("waiting_review", 3);
    const item: AuditItem = {
      itemId: "item_active_check",
      roundId: "round_example",
      itemKey: "request-authorization",
      ordinal: 50,
      kind: "check",
      subjectKey: "POST /orders/{id} · Authorization check",
      task: currentAudit.inputs.source!,
      origin: {
        schema: "contractor.audit.item-origin.v1",
        entryKey: "request-authorization",
        sourceRef: currentAudit.inputs.source!.ref,
        sourceContentDigest: currentAudit.inputs.source!.digest,
        sourceMediaType: "application/zip",
        canonicalInventoryDigest: `sha256:${"c".repeat(64)}`,
      },
      workflowRole: "check",
      state: "awaiting_review",
      approvalKind: "active-check-approval",
      attempts: [],
      createdAt: currentAudit.createdAt,
      updatedAt: currentAudit.updatedAt,
    };
    let review: AuditReviewRequest = {
      requestId: "review_active_check",
      auditId: currentAudit.auditId,
      subjectKind: "audit-item-action",
      subjectId: "item_active_check",
      kind: "active-check-approval",
      subjectRevision: 1,
      subjectDigest: `sha256:${"b".repeat(64)}`,
      requestedActions: ["approve", "reject"],
      state: "pending",
      revision: 1,
      createdAt: currentAudit.createdAt,
      updatedAt: currentAudit.updatedAt,
    };
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
          return jsonResponse(currentAudit, {
            headers: { ETag: `"${currentAudit.revision}"` },
          });
        }
        if (path === "/v1/audits/audit_example/items") {
          return jsonResponse(
            new URL(request.url).searchParams.has("cursor")
              ? { items: [item], page: { hasMore: false } }
              : {
                  items: [],
                  page: { hasMore: true, nextCursor: "later-checks" },
                },
          );
        }
        if (
          path === "/v1/audits/audit_example/reviews" &&
          request.method === "GET"
        ) {
          return jsonResponse({ items: [review], page: { hasMore: false } });
        }
        if (
          path ===
            "/v1/audits/audit_example/reviews/review_active_check/decisions" &&
          request.method === "POST"
        ) {
          expect(request.headers.get("If-Match")).toBe('"1"');
          expect(request.headers.get("Idempotency-Key")).toMatch(
            /^audit-action-review-ui-/u,
          );
          expect(await request.json()).toEqual({
            action: "approve",
            rationale: "The target and exact active request are **approved**.",
          });
          const decision = {
            decisionId: "decision_active_check",
            requestId: review.requestId,
            auditId: review.auditId,
            action: "approve" as const,
            actorId: session.principal.userId,
            rationale: "The target and exact active request are **approved**.",
            subjectRevision: review.subjectRevision,
            subjectDigest: review.subjectDigest,
            createdAt: currentAudit.updatedAt,
          };
          review = { ...review, state: "decided", revision: 2, decision };
          currentAudit = auditAt("active", 4);
          return jsonResponse({ request: review, decision, replayed: false });
        }
        if (path.endsWith("/coverage") || path.endsWith("/reviews"))
          return jsonResponse({ items: [], page: { hasMore: false } });
        throw new Error(`unexpected ${request.method} ${path}`);
      }),
    );
    renderApplication(
      api,
      "/projects/project_example/audits/audit_example/reviews",
    );
    const user = userEvent.setup();
    expect(await screen.findByText("Active check approval")).toBeVisible();
    expect(
      await screen.findByRole("heading", { name: item.subjectKey }),
    ).toBeVisible();
    expect(screen.getByRole("link", { name: "View check →" })).toHaveAttribute(
      "href",
      "/projects/project_example/audits/audit_example/checks#check-item_active_check",
    );
    await user.type(
      screen.getByLabelText("Rationale"),
      "The target and exact active request are **approved**.",
    );
    await user.click(
      screen.getByRole("button", { name: "Approve exact subject" }),
    );
    expect(
      await screen.findByText("approve", { exact: false, selector: "span" }),
    ).toBeVisible();
    expect(
      await screen.findByText("approved", { selector: "strong" }),
    ).toBeVisible();
  });

  it("marks only an authorized applicability review as not applicable", async () => {
    const currentAudit = auditAt("active", 3);
    const review: AuditReviewRequest = {
      requestId: "review_applicability",
      auditId: currentAudit.auditId,
      subjectKind: "audit-item-action",
      subjectId: "item_asvs_documentation",
      kind: "requirement-applicability",
      subjectRevision: 1,
      subjectDigest: `sha256:${"c".repeat(64)}`,
      requestedActions: ["approve", "reject", "not_applicable"],
      state: "pending",
      revision: 1,
      createdAt: currentAudit.createdAt,
      updatedAt: currentAudit.updatedAt,
    };
    let decided = false;
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
          return jsonResponse(currentAudit, { headers: { ETag: '"3"' } });
        }
        if (
          path === "/v1/audits/audit_example/reviews" &&
          request.method === "GET"
        ) {
          return jsonResponse({
            items: decided
              ? [
                  {
                    ...review,
                    state: "decided",
                    revision: 2,
                    decision: {
                      decisionId: "decision_not_applicable",
                      requestId: review.requestId,
                      auditId: review.auditId,
                      action: "not_applicable",
                      actorId: session.principal.userId,
                      rationale:
                        "Documentation is outside this exact application scope.",
                      subjectRevision: review.subjectRevision,
                      subjectDigest: review.subjectDigest,
                      createdAt: currentAudit.updatedAt,
                    },
                  },
                ]
              : [review],
            page: { hasMore: false },
          });
        }
        if (
          path ===
            "/v1/audits/audit_example/reviews/review_applicability/decisions" &&
          request.method === "POST"
        ) {
          expect(request.headers.get("If-Match")).toBe('"1"');
          expect(await request.json()).toEqual({
            action: "not_applicable",
            rationale: "Documentation is outside this exact application scope.",
          });
          decided = true;
          return jsonResponse({
            request: review,
            decision: {
              decisionId: "decision_not_applicable",
              requestId: review.requestId,
              auditId: review.auditId,
              action: "not_applicable",
              actorId: session.principal.userId,
              rationale:
                "Documentation is outside this exact application scope.",
              subjectRevision: review.subjectRevision,
              subjectDigest: review.subjectDigest,
              createdAt: currentAudit.updatedAt,
            },
            replayed: false,
          });
        }
        if (path === "/v1/audits/audit_example/items") {
          return jsonResponse({ items: [], page: { hasMore: false } });
        }
        if (path === "/v1/audits/audit_example/report") {
          return jsonResponse({ status: "pending" });
        }
        if (path.endsWith("/coverage") || path.endsWith("/reviews"))
          return jsonResponse({ items: [], page: { hasMore: false } });
        throw new Error(`unexpected ${request.method} ${path}`);
      }),
    );
    renderApplication(
      api,
      "/projects/project_example/audits/audit_example/reviews",
    );
    const user = userEvent.setup();
    await user.type(
      await screen.findByLabelText("Rationale"),
      "Documentation is outside this exact application scope.",
    );
    await user.click(
      screen.getByRole("button", { name: "Mark not applicable" }),
    );
    expect(
      await screen.findByText("not_applicable", {
        exact: false,
        selector: "span",
      }),
    ).toBeVisible();
    expect(decided).toBe(true);
  });

  it.each(["draft", "paused"] as const)(
    "chooses unlimited time before starting or continuing a %s Audit",
    async (state) => {
      let current = auditAt(state, 2);
      if (state === "paused")
        current = {
          ...current,
          stopReason: {
            code: "deadline_exhausted",
            message: "The Audit wall-time deadline was reached",
          },
        };
      const writes: Request[] = [];
      const api = new PublicAPI(
        runtimeConfig,
        vi.fn(async (input) => {
          const request = input instanceof Request ? input : new Request(input);
          const path = new URL(request.url).pathname;
          if (path === "/v1/auth/session") return jsonResponse(session);
          if (path === "/v1/projects/project_example")
            return jsonResponse(project, { headers: { ETag: '"1"' } });
          if (path === "/v1/audits/audit_example")
            return jsonResponse(current, {
              headers: { ETag: `"${current.revision}"` },
            });
          if (path.endsWith("/start") || path.endsWith("/resume")) {
            writes.push(request.clone());
            expect(await request.json()).toEqual({ deadlineSeconds: 0 });
            expect(request.headers.get("If-Match")).toBe('"2"');
            current = auditAt("active", 3);
            return jsonResponse(
              path.endsWith("/start")
                ? {
                    audit: current,
                    round: { roundId: "round_example" },
                    items: [],
                  }
                : current,
              { headers: { ETag: '"3"' } },
            );
          }
          if (path.endsWith("/coverage") || path.endsWith("/reviews"))
            return jsonResponse({ items: [], page: { hasMore: false } });
          throw new Error(`unexpected ${request.method} ${path}`);
        }),
      );
      renderApplication(api, "/projects/project_example/audits/audit_example");
      const user = userEvent.setup();
      const name = state === "draft" ? "Start Audit" : "Continue Audit";
      await user.click(await screen.findByRole("button", { name }));
      const dialog = screen.getByRole("dialog", { name });
      expect(writes).toHaveLength(0);
      expect(within(dialog).getByLabelText("Audit time limit")).toHaveFocus();
      expect(within(dialog).getByLabelText("Audit time limit")).toHaveValue(
        "604800",
      );
      await user.selectOptions(
        within(dialog).getByLabelText("Audit time limit"),
        "0",
      );
      await user.click(within(dialog).getByRole("button", { name }));
      expect(
        await screen.findByRole("button", { name: "Pause new Audit Runs" }),
      ).toBeVisible();
      expect(writes).toHaveLength(1);
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    },
  );

  it.each(["completed", "failed"] as const)(
    "keeps a terminal %s Audit with an old deadline reason final",
    async (state) => {
      const current = {
        ...auditAt(state, 2),
        stopReason: {
          code: "deadline_exhausted",
          message: "The Audit wall-time deadline was reached",
        },
      };
      const api = new PublicAPI(
        runtimeConfig,
        vi.fn(async (input) => {
          const request = input instanceof Request ? input : new Request(input);
          const path = new URL(request.url).pathname;
          expect(request.method).toBe("GET");
          if (path === "/v1/auth/session") return jsonResponse(session);
          if (path === "/v1/projects/project_example")
            return jsonResponse(project, { headers: { ETag: '"1"' } });
          if (path === "/v1/audits/audit_example")
            return jsonResponse(current, { headers: { ETag: '"2"' } });
          if (path.endsWith("/coverage") || path.endsWith("/reviews"))
            return jsonResponse({ items: [], page: { hasMore: false } });
          throw new Error(`unexpected ${request.method} ${path}`);
        }),
      );
      renderApplication(api, "/projects/project_example/audits/audit_example");
      expect(
        await screen.findByRole("button", { name: "Delete Audit" }),
      ).toBeVisible();
      expect(
        screen.queryByRole("button", { name: "Continue Audit" }),
      ).not.toBeInTheDocument();
      expect(screen.getByText(current.stopReason.message)).toBeVisible();
      expect(
        screen.queryByText(/Continue with a longer limit/),
      ).not.toBeInTheDocument();
    },
  );

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
        if (path.endsWith("/workspace"))
          return jsonResponse({
            auditId: current.auditId,
            auditRevision: current.revision,
            asOf: current.updatedAt,
            executionState: current.state,
            outstandingRuns: 0,
            totalChecks: 0,
            completedChecks: 0,
            issues: 0,
            gaps: 0,
            unchecked: 0,
            findings: 0,
            unreviewedFindings: 0,
            pendingReviews: 0,
          });
        if (path.endsWith("/coverage") || path.endsWith("/reviews"))
          return jsonResponse({ items: [], page: { hasMore: false } });
        throw new Error(`unexpected ${request.method} ${path}`);
      }),
    );
    renderApplication(api, "/projects/project_example/audits/audit_example");
    const user = userEvent.setup();

    await user.click(
      await screen.findByRole("button", { name: "Pause new Audit Runs" }),
    );
    expect(
      within(await screen.findByRole("alert")).getByText(
        "Audit revision changed",
      ),
    ).toBeVisible();
    await vi.waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Continue Audit" }),
      ).toBeVisible(),
    );
    expect(pauseRequested).toBe(true);
    expect(screen.getByText("revision 3")).toBeVisible();
  });

  it("confirms Audit cancellation and deletion without duplicate mutations", async () => {
    let current = auditAt("active", 2);
    const mutations: Request[] = [];
    let releaseCancel: (() => void) | undefined;
    const cancelGate = new Promise<void>((resolve) => {
      releaseCancel = resolve;
    });
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const path = new URL(request.url).pathname;
        if (path === "/v1/auth/session") return jsonResponse(session);
        if (path === "/v1/projects/project_example") {
          return jsonResponse(project, { headers: { ETag: '"1"' } });
        }
        if (path === "/v1/audits/audit_example/cancel") {
          mutations.push(request.clone());
          await cancelGate;
          current = auditAt("cancelled", 3);
          return jsonResponse(current, {
            status: 202,
            headers: { ETag: '"3"' },
          });
        }
        if (
          path === "/v1/audits/audit_example" &&
          request.method === "DELETE"
        ) {
          mutations.push(request.clone());
          current = auditAt("deleting", 4);
          return jsonResponse(current, {
            status: 202,
            headers: { ETag: '"4"' },
          });
        }
        if (path === "/v1/audits/audit_example") {
          return jsonResponse(current, {
            headers: { ETag: `"${current.revision}"` },
          });
        }
        if (path.endsWith("/coverage") || path.endsWith("/reviews"))
          return jsonResponse({ items: [], page: { hasMore: false } });
        throw new Error(`unexpected ${request.method} ${path}`);
      }),
    );
    renderApplication(api, "/projects/project_example/audits/audit_example");
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: "Cancel" }));
    let dialog = screen.getByRole("alertdialog", {
      name: "Cancel this Audit?",
    });
    expect(within(dialog).getByText(/Payment service/u)).toBeVisible();
    expect(within(dialog).getByText("audit_example")).toBeVisible();
    expect(
      within(dialog).getByText("owasp-top10-2025-source-risk@1"),
    ).toBeVisible();
    expect(
      within(dialog).getByRole("button", { name: "Keep Audit unchanged" }),
    ).toHaveFocus();
    fireEvent.click(dialog.parentElement!);
    expect(dialog).toBeVisible();
    expect(mutations).toHaveLength(0);

    await user.keyboard("{Escape}");
    expect(
      screen.queryByRole("alertdialog", { name: "Cancel this Audit?" }),
    ).toBeNull();
    expect(mutations).toHaveLength(0);

    await user.click(screen.getByRole("button", { name: "Cancel" }));
    dialog = screen.getByRole("alertdialog", {
      name: "Cancel this Audit?",
    });
    await user.dblClick(
      within(dialog).getByRole("button", { name: "Confirm cancellation" }),
    );
    await vi.waitFor(() => expect(mutations).toHaveLength(1));
    expect(mutations[0]?.headers.get("If-Match")).toBe('"2"');
    expect(mutations[0]?.headers.get("Idempotency-Key")).toMatch(
      /^mutate-audit-ui-/u,
    );
    releaseCancel?.();
    await vi.waitFor(() =>
      expect(screen.getByText("cancelled", { exact: true })).toBeVisible(),
    );

    await user.click(screen.getByRole("button", { name: "Delete Audit" }));
    dialog = screen.getByRole("alertdialog", {
      name: "Delete this Audit?",
    });
    expect(
      within(dialog).getByText(/Permanently delete this audit/u),
    ).toBeVisible();
    await user.click(
      within(dialog).getByRole("button", { name: "Begin Audit deletion" }),
    );
    await vi.waitFor(() => expect(mutations).toHaveLength(2));
    expect(mutations[1]?.method).toBe("DELETE");
    expect(mutations[1]?.headers.get("If-Match")).toBe('"3"');
    await vi.waitFor(() =>
      expect(screen.getByText("deleting", { exact: true })).toBeVisible(),
    );
  });

  it("refreshes a stale destructive confirmation before an explicit retry", async () => {
    let current = auditAt("active", 2);
    const cancelRevisions: string[] = [];
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const path = new URL(request.url).pathname;
        if (path === "/v1/auth/session") return jsonResponse(session);
        if (path === "/v1/projects/project_example") {
          return jsonResponse(project, { headers: { ETag: '"1"' } });
        }
        if (path === "/v1/audits/audit_example/cancel") {
          cancelRevisions.push(request.headers.get("If-Match") ?? "");
          if (cancelRevisions.length === 1) {
            current = auditAt("paused", 3);
            return jsonResponse(
              {
                code: "precondition_failed",
                message: "Audit revision changed",
                retryable: false,
                requestId: "request_stale_cancel",
              },
              { status: 412 },
            );
          }
          current = auditAt("cancelled", 4);
          return jsonResponse(current, {
            status: 202,
            headers: { ETag: '"4"' },
          });
        }
        if (path === "/v1/audits/audit_example") {
          return jsonResponse(current, {
            headers: { ETag: `"${current.revision}"` },
          });
        }
        if (path.endsWith("/coverage") || path.endsWith("/reviews"))
          return jsonResponse({ items: [], page: { hasMore: false } });
        throw new Error(`unexpected ${request.method} ${path}`);
      }),
    );
    renderApplication(api, "/projects/project_example/audits/audit_example");
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: "Cancel" }));
    await user.click(
      screen.getByRole("button", { name: "Confirm cancellation" }),
    );
    expect(
      await screen.findByText("The Audit revision changed.", { exact: false }),
    ).toBeVisible();
    await vi.waitFor(() =>
      expect(screen.getByText(/paused · revision 3/u)).toBeVisible(),
    );
    expect(cancelRevisions).toEqual(['"2"']);

    await user.click(
      screen.getByRole("button", { name: "Confirm cancellation" }),
    );
    await vi.waitFor(() => expect(cancelRevisions).toEqual(['"2"', '"3"']));
    await vi.waitFor(() =>
      expect(screen.getByText("cancelled", { exact: true })).toBeVisible(),
    );
  });
  it("links complete current-round progress to filtered checks and pending decisions", async () => {
    const previousScroll = HTMLElement.prototype.scrollIntoView;
    const scroll = vi.fn();
    HTMLElement.prototype.scrollIntoView = scroll;
    const completed = auditAt("completed", 4);
    completed.currentRoundId = "round_example";
    const coverageCursors: Array<string | null> = [];
    const reviewCursors: Array<string | null> = [];
    const review: AuditReviewRequest = {
      requestId: "review_report",
      auditId: completed.auditId,
      subjectKind: "audit-report",
      subjectId: "report_example",
      kind: "report-acceptance",
      subjectRevision: 1,
      subjectDigest: `sha256:${"b".repeat(64)}`,
      requestedActions: ["approve", "reject"],
      state: "pending",
      revision: 1,
      createdAt: completed.createdAt,
      updatedAt: completed.updatedAt,
    };
    const row = (
      itemId: string,
      status: AuditCoverageRow["coverage"]["status"],
    ): AuditCoverageRow => ({
      roundId: "round_example",
      itemId,
      ordinal: 0,
      itemKey: itemId,
      subjectKey: itemId,
      coverage: { status, requested: [], completed: [], gaps: [] },
      updatedAt: completed.updatedAt,
    });
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const url = new URL(request.url),
          cursor = url.searchParams.get("cursor");
        if (url.pathname === "/v1/auth/session") return jsonResponse(session);
        if (url.pathname === "/v1/projects/project_example")
          return jsonResponse(project, { headers: { ETag: '"1"' } });
        if (url.pathname === "/v1/audits/audit_example")
          return jsonResponse(completed, { headers: { ETag: '"4"' } });
        if (url.pathname.endsWith("/workspace"))
          return jsonResponse({
            auditId: completed.auditId,
            auditRevision: 4,
            asOf: completed.updatedAt,
            roundId: "round_example",
            executionState: completed.state,
            outstandingRuns: 0,
            totalChecks: 3,
            completedChecks: 1,
            issues: 0,
            gaps: 1,
            unchecked: 1,
            findings: 51,
            unreviewedFindings: 51,
            pendingReviews: 1,
          });
        if (url.pathname.endsWith("/coverage")) {
          expect(url.searchParams.get("round")).toBe("round_example");
          coverageCursors.push(cursor);
          return jsonResponse(
            cursor === null
              ? {
                  items: [row("met", "satisfied")],
                  page: { hasMore: true, nextCursor: "coverage-next" },
                }
              : {
                  items: [row("gap", "blocked"), row("waiting", "not-tested")],
                  page: { hasMore: false },
                },
          );
        }
        if (url.pathname.endsWith("/reviews")) {
          reviewCursors.push(cursor);
          return jsonResponse({
            items: [review],
            page: { hasMore: false },
            auditRevision: 4,
            asOf: completed.updatedAt,
            total: 1,
          });
        }
        throw new Error(`unexpected ${url.pathname}`);
      }),
    );
    const { router } = renderApplication(
      api,
      "/projects/project_example/audits/audit_example",
    );
    const progress = await screen.findByRole("region", {
      name: "Audit progress",
    });
    expect(await within(progress).findByText("Partial coverage")).toBeVisible();
    expect(
      within(progress).getByRole("progressbar", { name: "Concluded checks" }),
    ).toHaveAttribute("value", "1");
    expect(
      await within(progress).findByRole("link", {
        name: "Completed / total checks: 1 / 3",
      }),
    ).toHaveAttribute(
      "href",
      "/projects/project_example/audits/audit_example/coverage?auditRevision=4",
    );
    expect(
      within(progress).getByRole("link", { name: "Need follow-up: 1" }),
    ).toHaveAttribute(
      "href",
      "/projects/project_example/audits/audit_example/coverage?result=uncertain&auditRevision=4",
    );
    expect(coverageCursors).toEqual([]);
    expect(reviewCursors).toEqual([]);
    const user = userEvent.setup();
    await user.click(
      await within(progress).findByRole("link", {
        name: "Pending decisions: 1",
      }),
    );
    expect(
      await screen.findByRole("combobox", { name: "Review state" }),
    ).toHaveValue("pending");
    expect(router.state.location.search).toBe("?state=pending&auditRevision=4");
    expect(router.state.location.hash).toBe("");
    expect(document.getElementById("review-review_report")).toBeVisible();
    expect(reviewCursors).toEqual([null]);
    expect(
      screen.queryByRole("button", { name: "Next page" }),
    ).not.toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Refresh context" }),
    ).toBeEnabled();
    HTMLElement.prototype.scrollIntoView = previousScroll;
    await user.selectOptions(
      screen.getByRole("combobox", { name: "Review state" }),
      "all",
    );
    expect(router.state.location.search).toBe("");
  });
});

describe("Audit workspace snapshot navigation", () => {
  it("uses whole-filter totals, pins continuations, and resets stale cursors on filter changes", async () => {
    const audit = auditAt("completed", 5);
    const queries: URLSearchParams[] = [];
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const url = new URL(request.url);
        if (url.pathname === "/v1/auth/session") return jsonResponse(session);
        if (url.pathname === "/v1/projects/project_example")
          return jsonResponse(project, { headers: { ETag: '"1"' } });
        if (url.pathname === "/v1/audits/audit_example")
          return jsonResponse(audit, { headers: { ETag: '"5"' } });
        if (url.pathname.endsWith("/findings")) {
          queries.push(url.searchParams);
          if (url.searchParams.has("cursor"))
            return jsonResponse(
              {
                code: "conflict",
                message: "Audit changed",
                retryable: false,
                requestId: "stale",
              },
              { status: 409 },
            );
          return jsonResponse({
            items: [findingAt("proposed", 1)],
            page: { hasMore: true, nextCursor: "next-exact" },
            total: url.searchParams.has("severity") ? 7 : 65,
            auditRevision: 5,
            asOf: audit.updatedAt,
          });
        }
        if (url.pathname.endsWith("/reviews"))
          return jsonResponse({
            items: [],
            page: { hasMore: false },
            total: 0,
            auditRevision: 5,
            asOf: audit.updatedAt,
          });
        throw new Error(`unexpected ${url}`);
      }),
    );
    const { router } = renderApplication(
      api,
      "/projects/project_example/audits/audit_example/findings?verdict=unreviewed",
    );
    const user = userEvent.setup();
    expect(
      await screen.findByText(/Showing 1 of 65 matching records/),
    ).toBeVisible();
    expect(queries).toHaveLength(1);
    await user.click(screen.getByRole("button", { name: "Next page" }));
    expect(await screen.findByText(/This Audit changed/)).toBeVisible();
    expect(queries.at(-1)?.get("cursor")).toBe("next-exact");
    expect(queries.at(-1)?.get("auditRevision")).toBe("5");
    expect(queries.at(-1)?.get("verdict")).toBe("unreviewed");
    await user.selectOptions(screen.getByLabelText("Analyst severity"), "high");
    expect(
      await screen.findByText(/Showing 1 of 7 matching records/),
    ).toBeVisible();
    expect(router.state.location.search).toBe(
      "?verdict=unreviewed&severity=high",
    );
    expect(queries.at(-1)?.has("cursor")).toBe(false);
    expect(queries.at(-1)?.has("auditRevision")).toBe(false);
  });

  it("opens the exact report from a paged review queue, preserves return context and blocks a different review", async () => {
    const audit = auditAt("waiting_review", 5);
    const review: AuditReviewRequest = {
      requestId: "report-review",
      auditId: audit.auditId,
      subjectKind: "audit-report",
      subjectId: audit.auditId,
      kind: "report-acceptance",
      subjectRevision: 4,
      subjectDigest: `sha256:${"b".repeat(64)}`,
      requestedActions: ["approve", "reject"],
      state: "pending",
      revision: 1,
      createdAt: audit.createdAt,
      updatedAt: audit.updatedAt,
    };
    const mutations: string[] = [];
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const url = new URL(request.url);
        if (request.method !== "GET") mutations.push(request.method);
        if (url.pathname === "/v1/auth/session") return jsonResponse(session);
        if (url.pathname === "/v1/projects/project_example")
          return jsonResponse(project, { headers: { ETag: '"1"' } });
        if (url.pathname === "/v1/audits/audit_example")
          return jsonResponse(audit, { headers: { ETag: '"5"' } });
        if (url.pathname.endsWith("/reviews"))
          return jsonResponse({
            items: [review],
            page: { hasMore: false },
            total: 51,
            auditRevision: 5,
            asOf: audit.updatedAt,
          });
        if (url.pathname.endsWith("/report"))
          return jsonResponse({
            status: "proposed",
            summary: "Exact retained evidence for owner acceptance.",
            summaryArtifact: {
              ref: {
                namespace: "audit-example",
                name: "report.md",
                revision: "report-r1",
              },
              digest: `sha256:${"1".repeat(64)}`,
              mediaType: "text/markdown",
              sizeBytes: 44,
            },
            review,
          });
        throw new Error(`unexpected ${url}`);
      }),
    );
    const queue =
      "/projects/project_example/audits/audit_example/reviews?state=pending&cursor=page-two&auditRevision=5";
    const { router } = renderApplication(api, queue);
    const user = userEvent.setup();
    await user.click(
      await screen.findByRole("link", { name: "Review report →" }),
    );
    expect(
      await screen.findByText("Exact retained evidence for owner acceptance."),
    ).toBeVisible();
    expect(
      screen.getByRole("button", { name: "Approve exact subject" }),
    ).toBeDisabled();
    await user.click(screen.getByRole("link", { name: "← Audit reviews" }));
    expect(router.state.location.pathname + router.state.location.search).toBe(
      queue,
    );
    await router.navigate(
      "/projects/project_example/audits/audit_example/report?review=removed-review",
    );
    expect(
      await screen.findByText(/requested report review is unavailable/),
    ).toBeVisible();
    expect(
      screen.queryByRole("button", { name: "Approve exact subject" }),
    ).not.toBeInTheDocument();
    expect(mutations).toEqual([]);
  });
});
