import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, it, vi } from "vitest";

import { PublicAPI } from "../../api/client";
import { PublicAPIProvider } from "../../api/context";
import type { Project } from "../../api/projects";
import { ProjectActiveChecksEditor } from "./active-checks-editor";

const project: Project = {
  projectId: "policy-project",
  kind: "project",
  name: "Policy",
  description: "",
  activeChecks: "automatic",
  lifecycle: "active",

  revision: "4",
  createdAt: "2026-10-15T10:00:00Z",
  updatedAt: "2026-10-15T10:00:00Z",
};

it.each([false, true])(
  "saves Project policy with the edited revision; conflict=%s",
  async (conflict) => {
    const requests: Request[] = [];
    const api = new PublicAPI(
      {
        apiBaseUrl: "https://api.example.test",
        uiVersion: "0.6.5",
        supportedApiVersions: ["contractor.public.v1"],
      },
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const request = new Request(input, init);
        requests.push(request);
        return conflict
          ? Response.json(
              {
                error: {
                  code: "precondition_failed",
                  message: "Project changed",
                  retryable: false,
                },
              },
              {
                status: 412,
                headers: { "X-Contractor-API-Version": "contractor.public.v1" },
              },
            )
          : Response.json(
              { ...project, activeChecks: "approval-required", revision: "5" },
              {
                headers: {
                  ETag: '"5"',
                  "X-Contractor-API-Version": "contractor.public.v1",
                },
              },
            );
      }),
    );
    api.csrf.replace("a".repeat(43));
    const client = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });
    const view = render(
      <QueryClientProvider client={client}>
        <PublicAPIProvider api={api}>
          <ProjectActiveChecksEditor project={project} />
        </PublicAPIProvider>
      </QueryClientProvider>,
    );
    const user = userEvent.setup();
    expect(screen.getByText("Automatically")).toBeVisible();
    await user.click(
      screen.getByRole("button", { name: "Edit active checks" }),
    );
    await user.selectOptions(
      screen.getByRole("combobox", { name: "Active checks" }),
      "approval-required",
    );
    view.rerender(
      <QueryClientProvider client={client}>
        <PublicAPIProvider api={api}>
          <ProjectActiveChecksEditor project={{ ...project, revision: "6" }} />
        </PublicAPIProvider>
      </QueryClientProvider>,
    );
    await user.click(
      screen.getByRole("button", { name: "Save active checks" }),
    );
    await waitFor(() => expect(requests).toHaveLength(1));
    expect(requests[0]?.headers.get("If-Match")).toBe('"4"');
    await expect(requests[0]?.clone().json()).resolves.toEqual({
      activeChecks: "approval-required",
    });
    if (conflict)
      await waitFor(() =>
        expect(screen.getByRole("combobox")).toHaveValue("approval-required"),
      );
    else await waitFor(() => expect(screen.queryByRole("combobox")).toBeNull());
  },
);
