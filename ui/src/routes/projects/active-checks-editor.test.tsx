import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, it } from "vitest";

import { PublicAPIProvider } from "../../api/context";
import * as build from "../../test/builders";
import { fakeServer, json, preconditionFailed } from "../../test/fake-server";
import { testAPI } from "../../test/render";
import { ProjectActiveChecksEditor } from "./active-checks-editor";

const project = build.project("policy-project", {
  name: "Policy",
  revision: "4",
  createdAt: "2026-10-15T10:00:00Z",
  updatedAt: "2026-10-15T10:00:00Z",
});

it.each([false, true])(
  "saves Project policy with the edited revision; conflict=%s",
  async (conflict) => {
    const { api, server } = testAPI(
      fakeServer({
        "PATCH /v1/projects/{projectId}": () =>
          conflict
            ? preconditionFailed("Project changed")
            : json(
                {
                  ...project,
                  activeChecks: "approval-required",
                  revision: "5",
                },
                200,
                { ETag: '"5"' },
              ),
      }),
    );
    const client = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });
    // The page re-renders the editor with a newer project while it edits.
    const editor = (current: typeof project) => (
      <QueryClientProvider client={client}>
        <PublicAPIProvider api={api}>
          <ProjectActiveChecksEditor project={current} />
        </PublicAPIProvider>
      </QueryClientProvider>
    );
    const view = render(editor(project));
    const user = userEvent.setup();
    expect(screen.getByText("Automatically")).toBeVisible();
    await user.click(
      screen.getByRole("button", { name: "Edit active checks" }),
    );
    await user.selectOptions(
      screen.getByRole("combobox", { name: "Active checks" }),
      "approval-required",
    );
    view.rerender(editor({ ...project, revision: "6" }));
    await user.click(
      screen.getByRole("button", { name: "Save active checks" }),
    );
    await waitFor(() => expect(server.requests).toHaveLength(1));
    expect(server.requests[0]?.headers.get("If-Match")).toBe('"4"');
    await expect(server.requests[0]?.clone().json()).resolves.toEqual({
      activeChecks: "approval-required",
    });
    if (conflict)
      await waitFor(() =>
        expect(screen.getByRole("combobox")).toHaveValue("approval-required"),
      );
    else await waitFor(() => expect(screen.queryByRole("combobox")).toBeNull());
  },
);
