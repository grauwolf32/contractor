import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter } from "react-router";
import { describe, expect, it, vi } from "vitest";

import { PublicAPI } from "../api/client";
import { Application } from "../app/application";
import { applicationRoutes } from "../app/router";
import type { RuntimeConfig } from "../config/runtime-config";

const runtimeConfig: RuntimeConfig = {
  uiVersion: "0.1.0",
  supportedApiVersions: ["contractor.public.v1"],
  apiBaseUrl: "http://127.0.0.1:8080",
};

const session = {
  principal: {
    userId: "user_local",
    username: "owner",
    capabilities: ["user", "operations"] as const,
  },
  csrfToken: "a".repeat(43),
  idleExpiresAt: "2026-09-05T20:00:00Z",
  absoluteExpiresAt: "2026-09-06T12:00:00Z",
};

const evaluation = {
  projectId: "evaluation-openapi",
  kind: "evaluation",
  name: "OpenAPI regression",
  description: "A/B evaluation workspace",
  revision: "1",
  createdAt: "2026-09-05T08:00:00Z",
  updatedAt: "2026-09-05T08:01:00Z",
};

function apiResponse(value: unknown, options: ResponseInit = {}): Response {
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

describe("Evals and global Skills routes", () => {
  it("filters evaluation Projects and groups their ordinary Runs by eval metadata", async () => {
    const requests: Request[] = [];
    const runs = [
      {
        runId: "run-eval-a",
        projectId: evaluation.projectId,
        workflow: "openapi-from-workspace@4",
        state: "succeeded",
        labels: {
          purpose: "eval",
          "eval.name": "openapi-regression",
          "eval.id": "eval-2026-09-05",
          "eval.leg": "a",
          "eval.fixture": "payment-service",
          "eval.case": "routes",
          "eval.sample": "1",
        },
        createdAt: "2026-09-05T08:10:00Z",
        updatedAt: "2026-09-05T08:12:00Z",
        finishedAt: "2026-09-05T08:12:00Z",
      },
      {
        runId: "run-eval-b",
        projectId: evaluation.projectId,
        workflow: "openapi-from-workspace@4",
        state: "running",
        labels: {
          purpose: "eval",
          "eval.name": "openapi-regression",
          "eval.id": "eval-2026-09-05",
          "eval.leg": "b",
          "eval.case": "routes",
          "eval.sample": "1",
        },
        createdAt: "2026-09-05T08:11:00Z",
        updatedAt: "2026-09-05T08:13:00Z",
      },
      {
        runId: "run-eval-unassigned",
        projectId: evaluation.projectId,
        workflow: "likec4-from-workspace@4",
        state: "failed",
        labels: { purpose: "eval" },
        createdAt: "2026-09-05T08:09:00Z",
        updatedAt: "2026-09-05T08:10:00Z",
        finishedAt: "2026-09-05T08:10:00Z",
      },
    ];
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        requests.push(request);
        const url = new URL(request.url);
        if (url.pathname === "/v1/auth/session") {
          return apiResponse(session);
        }
        if (url.pathname === "/v1/projects") {
          return apiResponse({ items: [evaluation], page: { hasMore: false } });
        }
        if (url.pathname === `/v1/projects/${evaluation.projectId}`) {
          return apiResponse(evaluation, { headers: { ETag: '"1"' } });
        }
        if (url.pathname === `/v1/projects/${evaluation.projectId}/artifacts`) {
          return apiResponse({ items: [], page: { hasMore: false } });
        }
        if (url.pathname === `/v1/projects/${evaluation.projectId}/runs`) {
          return apiResponse({ items: runs, page: { hasMore: false } });
        }
        if (url.pathname === "/v1/workflows") {
          return apiResponse({ items: [], page: { hasMore: false } });
        }
        throw new Error(`unexpected ${request.method} ${url.pathname}`);
      }),
    );
    const view = renderApplication(api, "/evals");

    const evalLink = await screen.findByRole("link", {
      name: "OpenAPI regression",
    });
    expect(evalLink).toHaveAttribute("href", "/evals/evaluation-openapi");
    const listRequest = requests.find(
      (request) => new URL(request.url).pathname === "/v1/projects",
    );
    expect(new URL(listRequest!.url).searchParams.get("kind")).toBe(
      "evaluation",
    );

    await userEvent.setup().click(evalLink);
    expect(
      await screen.findByRole("heading", { name: "Eval Runs" }),
    ).toBeInTheDocument();
    expect(screen.getByText("eval-2026-09-05")).toBeInTheDocument();
    expect(screen.getByText("Runs without eval.id")).toBeInTheDocument();
    const grouped =
      view.container.querySelector<HTMLElement>(".eval-run-group");
    expect(grouped).not.toBeNull();
    expect(within(grouped!).getByText("run-eval-a")).toBeInTheDocument();
    expect(within(grouped!).getByText("run-eval-b")).toBeInTheDocument();
    expect(within(grouped!).getByText("payment-service")).toBeInTheDocument();
    expect(within(grouped!).getAllByText("routes")).toHaveLength(2);
    expect(within(grouped!).getByText("a")).toBeInTheDocument();
    expect(within(grouped!).getByText("b")).toBeInTheDocument();
    expect(screen.getByText(/ordinary isolated Workflow Run/i)).toBeVisible();
  });

  it("uses only UserScope Artifact APIs for global Skill packages", async () => {
    const requests: Request[] = [];
    let created = false;
    const existing = {
      artifact: {
        namespace: "skills",
        name: "architecture-review",
        revision: "skill-r1",
      },
      mediaType: "application/vnd.contractor.agent-skill+zip",
      size: 128,
      current: true,
      frozen: false,
      createdAt: "2026-09-05T08:00:00Z",
    };
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        requests.push(request);
        const url = new URL(request.url);
        if (url.pathname === "/v1/auth/session") {
          return apiResponse(session);
        }
        if (url.pathname === "/v1/artifacts" && request.method === "GET") {
          return apiResponse({
            items: [
              existing,
              ...(created
                ? [
                    {
                      ...existing,
                      artifact: {
                        namespace: "skills",
                        name: "new-skill",
                        revision: "skill-r2",
                      },
                    },
                  ]
                : []),
            ],
            page: { hasMore: false },
          });
        }
        if (
          url.pathname === "/v1/artifacts/skills/new-skill" &&
          request.method === "PUT"
        ) {
          created = true;
          return apiResponse(
            {
              artifact: {
                namespace: "skills",
                name: "new-skill",
                revision: "skill-r2",
              },
              mediaType: "application/vnd.contractor.agent-skill+zip",
              size: 3,
            },
            { status: 201, headers: { ETag: '"skill-r2"' } },
          );
        }
        throw new Error(`unexpected ${request.method} ${url.pathname}`);
      }),
    );
    renderApplication(api, "/skills");

    expect(
      await screen.findByRole("link", { name: "architecture-review" }),
    ).toHaveAttribute("href", "/artifacts/skills/architecture-review");
    const listRequest = requests.find(
      (request) =>
        request.method === "GET" &&
        new URL(request.url).pathname === "/v1/artifacts",
    );
    expect(new URL(listRequest!.url).searchParams.get("namespace")).toBe(
      "skills",
    );

    const user = userEvent.setup();
    await user.click(screen.getByText("Upload Skill package"));
    expect(screen.getByLabelText("Namespace")).toBeDisabled();
    expect(screen.getByLabelText("Namespace")).toHaveValue("skills");
    expect(screen.getByLabelText("Media type")).toBeDisabled();
    expect(screen.getByLabelText("Media type")).toHaveValue(
      "application/vnd.contractor.agent-skill+zip",
    );
    await user.type(screen.getByLabelText("Name"), "new-skill");
    await user.upload(
      screen.getByLabelText(/Local file/),
      new File(["zip"], "new-skill.zip", { type: "application/zip" }),
    );
    await user.click(screen.getByRole("button", { name: "Create binding" }));

    expect(
      await screen.findByRole("link", {
        name: /Open skills\/new-skill@skill-r2/,
      }),
    ).toHaveAttribute("href", "/artifacts/skills/new-skill?revision=skill-r2");
    const upload = requests.find((request) => request.method === "PUT")!;
    expect(new URL(upload.url).pathname).toBe("/v1/artifacts/skills/new-skill");
    expect(upload.headers.get("Content-Type")).toBe(
      "application/vnd.contractor.agent-skill+zip",
    );
    expect(upload.headers.get("If-None-Match")).toBe("*");
    expect(upload.headers.get("X-CSRF-Token")).toBe(session.csrfToken);
    expect(await upload.text()).toBe("zip");
    expect(
      requests.every((request) => !request.url.includes("/projects/")),
    ).toBe(true);
  });
});
