import { describe, expect, it, vi } from "vitest";

import type { RuntimeConfig } from "../config/runtime-config";
import { PublicAPI } from "./client";
import {
  createProject,
  getProject,
  listProjectRuns,
  listProjects,
  updateProject,
  type Project,
} from "./projects";

const runtimeConfig: RuntimeConfig = {
  uiVersion: "0.1.0",
  supportedApiVersions: ["contractor.public.v1"],
  apiBaseUrl: "http://127.0.0.1:8080",
};

const project: Project = {
  projectId: "project_example",
  kind: "project",
  name: "Example",
  description: "Reusable inputs",
  revision: "1",
  createdAt: "2026-09-01T10:00:00Z",
  updatedAt: "2026-09-01T10:00:00Z",
};

function response(value: unknown, options: ResponseInit = {}): Response {
  const headers = new Headers(options.headers);
  headers.set("content-type", "application/json");
  headers.set("X-Contractor-API-Version", "contractor.public.v1");
  return new Response(JSON.stringify(value), { ...options, headers });
}

describe("Project API", () => {
  it("lists only the requested kind and validates a Project detail ETag", async () => {
    const requests: Request[] = [];
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        requests.push(request);
        const url = new URL(request.url);
        if (url.pathname === "/v1/projects") {
          return response({ items: [project], page: { hasMore: false } });
        }
        return response(project, { headers: { ETag: '"1"' } });
      }),
    );

    await expect(
      listProjects(api, { kind: "project", cursor: "next-page" }),
    ).resolves.toMatchObject({ items: [project] });
    await expect(getProject(api, project.projectId)).resolves.toEqual(project);

    const listURL = new URL(requests[0]!.url);
    expect(listURL.searchParams.get("kind")).toBe("project");
    expect(listURL.searchParams.get("cursor")).toBe("next-page");
    expect(listURL.searchParams.get("limit")).toBe("50");
  });

  it("creates idempotently and updates with the exact Project revision", async () => {
    const requests: Request[] = [];
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        requests.push(request);
        if (request.method === "POST") {
          return response(project, {
            status: 201,
            headers: { ETag: '"1"' },
          });
        }
        return response(
          { ...project, name: "Renamed", revision: "2" },
          { headers: { ETag: '"2"' } },
        );
      }),
    );
    api.csrf.replace("a".repeat(43));

    await createProject(api, {
      request: {
        kind: "project",
        name: "  Example  ",
        description: "  Reusable inputs  ",
      },
      idempotencyKey: "create-project-example",
    });
    await updateProject(api, {
      projectId: project.projectId,
      expectedRevision: "1",
      request: { name: " Renamed " },
    });

    expect(requests[0]?.headers.get("Idempotency-Key")).toBe(
      "create-project-example",
    );
    expect(requests[1]?.headers.get("If-Match")).toBe('"1"');
    expect(requests[0]?.headers.get("X-CSRF-Token")).toBe("a".repeat(43));
    await expect(requests[0]?.json()).resolves.toEqual({
      kind: "project",
      name: "Example",
      description: "Reusable inputs",
    });
    await expect(requests[1]?.json()).resolves.toEqual({ name: "Renamed" });
  });

  it("normalizes absent-safe Run labels in the Project history", async () => {
    let captured: Request | undefined;
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        captured = input instanceof Request ? input : new Request(input);
        return response({
          items: [
            {
              runId: "run_example",
              projectId: project.projectId,
              workflow: "openapi-from-source@1",
              state: "running",
              labels: undefined,
              createdAt: "2026-09-01T10:01:00Z",
              updatedAt: "2026-09-01T10:02:00Z",
            },
          ],
          page: { hasMore: false },
        });
      }),
    );
    const page = await listProjectRuns(api, { projectId: project.projectId });
    expect(page.items[0]?.labels).toEqual({});
    expect(new URL(captured!.url).pathname).toBe(
      `/v1/projects/${project.projectId}/runs`,
    );
  });
});
