import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import type { Audit } from "../../../api/audits";
import type { Project } from "../../../api/projects";
import {
  fakeAPI,
  jsonResponse,
  makeAudit,
  makeProject,
  renderApplication,
  workspaceOf,
} from "../../projects/audits/check-test-support";

const projectA = makeProject("project_a", "crapi-workshop");
const projectB = makeProject("project_b", "Payment service");

function fixture() {
  const audits: Record<string, Audit> = {
    a_running: makeAudit("a_running", "project_a", "active", {
      updatedAt: "2026-10-05T10:50:00Z",
      deadlineAt: "2026-10-06T10:00:00Z",
    }),
    b_waiting: makeAudit("b_waiting", "project_b", "waiting_review", {
      profileName: "owasp-top10-2025-source-risk",
      revision: 5,
      updatedAt: "2026-10-05T10:40:00Z",
    }),
    b_draft: makeAudit("b_draft", "project_b", "draft", {
      profileName: "owasp-asvs-5-0-l1-source-review",
      revision: 1,
      updatedAt: "2026-10-05T10:30:00Z",
    }),
    a_failed: makeAudit("a_failed", "project_a", "failed", {
      profileName: "owasp-wstg-4-2-source-review",
      updatedAt: "2026-10-05T10:20:00Z",
      stopReason: {
        code: "role_execution_not_retryable",
        message: "role check exited",
      },
    }),
    b_done: makeAudit("b_done", "project_b", "completed", {
      updatedAt: "2026-10-05T10:10:00Z",
    }),
  };
  return audits;
}

function serve(
  audits: Record<string, Audit>,
  options: {
    projects?: Project[];
    failProject?: string;
    extra?: Record<string, Audit>;
  } = {},
) {
  const projects = options.projects ?? [projectA, projectB];
  return fakeAPI(async (request, url) => {
    const path = url.pathname;
    if (path === "/v1/projects")
      return jsonResponse({ items: projects, page: { hasMore: false } });
    let match = /^\/v1\/projects\/([^/]+)$/.exec(path);
    if (match)
      return jsonResponse(
        projects.find((project) => project.projectId === match![1]),
        { headers: { ETag: '"1"' } },
      );
    match = /^\/v1\/projects\/([^/]+)\/audits$/.exec(path);
    if (match) {
      if (match[1] === options.failProject)
        return jsonResponse(
          {
            code: "unavailable",
            message: "Checks unavailable",
            retryable: true,
            requestId: "request_failed",
          },
          { status: 503 },
        );
      const state = url.searchParams.get("state");
      return jsonResponse({
        items: Object.values(audits).filter(
          (audit) =>
            audit.projectId === match![1] &&
            (state === null || audit.state === state),
        ),
        page: { hasMore: false },
      });
    }
    match = /^\/v1\/audits\/([^/]+)(\/[a-z]+)?$/.exec(path);
    if (match) {
      const audit = audits[match[1]!] ?? options.extra?.[match[1]!];
      if (audit === undefined)
        return jsonResponse(
          {
            code: "not_found",
            message: "Check not found",
            retryable: false,
            requestId: "request_missing",
          },
          { status: 404 },
        );
      if (match[2] === undefined) {
        if (request.method === "DELETE") {
          audits[audit.auditId] = {
            ...audit,
            state: "deleting",
            revision: audit.revision + 1,
          };
          return jsonResponse(audits[audit.auditId], {
            status: 202,
            headers: { ETag: `"${audit.revision + 1}"` },
          });
        }
        return jsonResponse(audit, {
          headers: { ETag: `"${audit.revision}"` },
        });
      }
      if (match[2] === "/workspace")
        return jsonResponse(
          workspaceOf(audit, {
            totalChecks: 3,
            completedChecks: audit.state === "completed" ? 3 : 1,
            issues: audit.state === "completed" ? 1 : 0,
            gaps: audit.state === "completed" ? 0 : 1,
            unchecked: audit.state === "completed" ? 0 : 1,
            findings: 2,
            unreviewedFindings: audit.auditId === "b_waiting" ? 2 : 0,
            pendingReviews: audit.state === "waiting_review" ? 1 : 0,
          }),
        );
      if (match[2] === "/cancel") {
        const cancelled = {
          ...audit,
          state: "cancelled" as const,
          revision: audit.revision + 1,
        };
        audits[audit.auditId] = cancelled;
        return jsonResponse(cancelled, {
          status: 202,
          headers: { ETag: `"${cancelled.revision}"` },
        });
      }
    }
    return undefined;
  });
}

describe("Checks list", () => {
  it("lists every project's checks and filters them by state and project in the URL", async () => {
    const { api } = serve(fixture());
    const { router } = renderApplication(api, "/checks");
    const user = userEvent.setup();
    const list = await screen.findByRole("region", { name: "Checks" });
    expect(
      within(list).getByRole("heading", { level: 1, name: "Checks" }),
    ).toBeVisible();
    await within(list).findByRole("link", {
      name: "OpenAPI · Operation trace · crapi-workshop",
    });
    const rows = () =>
      within(list)
        .getAllByRole("listitem")
        .filter((row) => row.id.startsWith("check-"));
    await waitFor(() => expect(rows()).toHaveLength(5));
    // Newest first, with project and state.
    expect(rows()[0]).toHaveTextContent("crapi-workshop");
    expect(rows()[0]).toHaveTextContent("Running");
    expect(rows()[1]).toHaveTextContent("Waiting for you");
    expect(
      within(list).getByText("1 running, 1 waiting for you."),
    ).toBeVisible();
    // A progress line where the Server's counts are read.
    expect(
      await within(rows()[0]!).findByRole("img", {
        name: "1 of 3 done: 1 done, 1 need follow-up, 1 not checked yet",
      }),
    ).toBeVisible();
    const filters = within(list).getByRole("group", {
      name: "Filter checks by state",
    });
    expect(
      within(filters).getByRole("button", { name: "All 5" }),
    ).toHaveAttribute("aria-pressed", "true");
    expect(
      within(filters).getByRole("button", { name: "Stopped / failed 1" }),
    ).toBeVisible();
    await user.click(within(filters).getByRole("button", { name: "Drafts 1" }));
    expect(router.state.location.search).toBe("?state=drafts");
    expect(rows()).toHaveLength(1);
    expect(rows()[0]).toHaveTextContent("ASVS 5.0 · Level 1 source review");
    await user.click(within(filters).getByRole("button", { name: "All 5" }));
    await user.selectOptions(
      within(list).getByRole("combobox", { name: "Project" }),
      "project_a",
    );
    expect(router.state.location.search).toBe("?project=project_a");
    expect(rows()).toHaveLength(2);
    expect(
      within(filters).getByRole("button", { name: "Waiting for you 0" }),
    ).toBeVisible();
    expect(
      within(list).getByRole("link", { name: "Start a check" }),
    ).toHaveAttribute("href", "/checks/new?project=project_a");
  });

  it("takes a check state as the state filter", async () => {
    const { api } = serve(fixture());
    renderApplication(api, "/checks?state=waiting_review");
    const filters = await screen.findByRole("group", {
      name: "Filter checks by state",
    });
    expect(
      within(filters).getByRole("button", { name: /^Waiting for you/u }),
    ).toHaveAttribute("aria-pressed", "true");
  });

  it("shows the selected check with progress, what waits for the user and the way in", async () => {
    const { api } = serve(fixture());
    const { router } = renderApplication(api, "/checks");
    const user = userEvent.setup();
    const list = await screen.findByRole("region", { name: "Checks" });
    await user.click(
      await within(list).findByRole("link", {
        name: "OWASP Top 10 · Source risks · Payment service",
      }),
    );
    expect(router.state.location.search).toBe("?check=b_waiting");
    const detail = screen.getByRole("region", { name: "Selected check" });
    expect(
      await within(detail).findByRole("heading", {
        name: "OWASP Top 10 · Source risks",
      }),
    ).toBeVisible();
    expect(within(detail).getByText("Waiting for you")).toBeVisible();
    expect(
      within(detail).getByRole("link", { name: "Payment service" }),
    ).toHaveAttribute("href", "/projects/project_b");
    expect(
      within(detail).getByRole("button", { name: "Copy check ID" }),
    ).toBeVisible();
    expect(
      within(detail).getByRole("link", { name: "Open check" }),
    ).toHaveAttribute("href", "/projects/project_b/audits/b_waiting");
    expect(await within(detail).findByText("1 of 3 done")).toBeVisible();
    // Waiting checks send decisions to the Inbox; possible issues to Issues.
    expect(
      within(detail).getByRole("link", { name: "1 decision waiting for you" }),
    ).toHaveAttribute("href", "/");
    expect(
      within(detail).getByRole("link", { name: "2 possible issues to review" }),
    ).toHaveAttribute("href", "/issues?project=project_b&state=proposed");
    expect(
      within(detail).getByRole("link", { name: "Need follow-up: 1" }),
    ).toHaveAttribute(
      "href",
      "/projects/project_b/audits/b_waiting/coverage?result=uncertain&auditRevision=5",
    );
    await user.click(
      within(list).getByRole("link", {
        name: "WSTG 4.2 · Source review · crapi-workshop",
      }),
    );
    expect(
      await within(
        screen.getByRole("region", { name: "Selected check" }),
      ).findByText(
        "The check stopped because a step failed in a way that cannot be retried.",
      ),
    ).toBeVisible();
    expect(screen.queryByText("role_execution_not_retryable")).toBeNull();
  });

  it("moves the selection with J and K and opens the check with Enter", async () => {
    const { api } = serve(fixture());
    const { router } = renderApplication(api, "/checks?state=all");
    const user = userEvent.setup();
    const list = await screen.findByRole("region", { name: "Checks" });
    await within(list).findByRole("link", {
      name: "WSTG 4.2 · Source review · crapi-workshop",
    });
    // Every row declares the keys; the footer hint is for sighted users.
    const rows = within(within(list).getByRole("list")).getAllByRole(
      "listitem",
    );
    expect(rows.length).toBeGreaterThan(1);
    for (const row of rows)
      expect(within(row).getByRole("link")).toHaveAttribute(
        "aria-keyshortcuts",
        "J K ArrowDown ArrowUp Home End Enter",
      );
    expect(list.querySelector(".checks-list-rows")).not.toHaveAttribute(
      "aria-keyshortcuts",
    );
    await user.keyboard("j");
    await waitFor(() =>
      expect(router.state.location.search).toBe("?check=a_running"),
    );
    await user.keyboard("j");
    await waitFor(() =>
      expect(router.state.location.search).toBe("?check=b_waiting"),
    );
    await user.keyboard("k");
    await waitFor(() =>
      expect(router.state.location.search).toBe("?check=a_running"),
    );
    within(list)
      .getByRole("link", {
        name: "OpenAPI · Operation trace · crapi-workshop",
        current: true,
      })
      .focus();
    await user.keyboard("{Enter}");
    await waitFor(() =>
      expect(router.state.location.pathname).toBe(
        "/projects/project_a/audits/a_running",
      ),
    );
  });

  it("stops and deletes the selected check only after confirmation", async () => {
    const audits = fixture();
    const { api, requests } = serve(audits);
    renderApplication(api, "/checks?check=a_running");
    const user = userEvent.setup();
    const detail = await screen.findByRole("region", {
      name: "Selected check",
    });
    await user.click(
      await within(detail).findByRole("button", { name: "Stop check" }),
    );
    let dialog = screen.getByRole("alertdialog", { name: "Stop this check?" });
    expect(within(dialog).getByText(/crapi-workshop/u)).toBeVisible();
    expect(within(dialog).getByText("a_running")).toBeVisible();
    expect(within(dialog).getByText("openapi-operation-trace@1")).toBeVisible();
    expect(within(dialog).getByText("Running · revision 3")).toBeVisible();
    expect(
      within(dialog).getByRole("button", { name: "Keep the check running" }),
    ).toHaveFocus();
    await user.click(
      within(dialog).getByRole("button", { name: "Keep the check running" }),
    );
    expect(screen.queryByRole("alertdialog")).toBeNull();
    await user.click(
      within(detail).getByRole("button", { name: "Stop check" }),
    );
    dialog = screen.getByRole("alertdialog", { name: "Stop this check?" });
    await user.click(
      within(dialog).getByRole("button", { name: "Stop check" }),
    );
    await waitFor(() =>
      expect(
        within(
          screen.getByRole("region", { name: "Selected check" }),
        ).getByText("Stopped"),
      ).toBeVisible(),
    );
    const cancel = requests.find((request) =>
      request.url.endsWith("/v1/audits/a_running/cancel"),
    );
    expect(cancel?.headers.get("If-Match")).toBe('"3"');
    expect(cancel?.headers.get("Idempotency-Key")).toMatch(
      /^mutate-audit-ui-/u,
    );
    // The list follows the Server, not the click.
    const list = screen.getByRole("region", { name: "Checks" });
    await waitFor(() =>
      expect(
        within(list).getByRole("button", { name: "Stopped / failed 2" }),
      ).toBeVisible(),
    );
    await user.click(within(detail).getByLabelText("Check actions"));
    await user.click(
      within(detail).getByRole("button", { name: "Delete check" }),
    );
    dialog = screen.getByRole("alertdialog", { name: "Delete this check?" });
    expect(
      within(dialog).getByText(/Permanently delete this check/u),
    ).toBeVisible();
    await user.click(
      within(dialog).getByRole("button", { name: "Delete check" }),
    );
    await waitFor(() =>
      expect(
        requests.some(
          (request) =>
            request.method === "DELETE" &&
            request.url.endsWith("/v1/audits/a_running"),
        ),
      ).toBe(true),
    );
    expect(
      await within(
        screen.getByRole("region", { name: "Selected check" }),
      ).findByText("Deleting"),
    ).toBeVisible();
  });

  it("starts a draft through the time limit dialog", async () => {
    const { api, requests } = serve(fixture());
    renderApplication(api, "/checks?state=drafts&check=b_draft");
    const user = userEvent.setup();
    const detail = await screen.findByRole("region", {
      name: "Selected check",
    });
    expect(
      await within(detail).findByText(/This check is a draft/u),
    ).toBeVisible();
    await user.click(
      await within(detail).findByRole("button", { name: "Start check" }),
    );
    const dialog = screen.getByRole("dialog", { name: "Start check" });
    expect(within(dialog).getByLabelText("Time limit")).toHaveValue("604800");
    expect(
      within(dialog)
        .getAllByRole("option")
        .map((option) => option.textContent),
    ).toEqual(["7 days", "24 hours", "No time limit", "Custom duration"]);
    await user.click(within(dialog).getByRole("button", { name: "Close" }));
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(requests.some((request) => request.method === "POST")).toBe(false);
  });

  it("opens a check beyond the listed ones and says when it is gone", async () => {
    const extra = makeAudit("old_check", "project_a", "completed", {
      updatedAt: "2026-09-01T10:00:00Z",
    });
    const { api } = serve(fixture(), { extra: { old_check: extra } });
    const { router } = renderApplication(api, "/checks?check=old_check");
    const detail = await screen.findByRole("region", {
      name: "Selected check",
    });
    expect(
      await within(detail).findByRole("link", { name: "crapi-workshop" }),
    ).toHaveAttribute("href", "/projects/project_a");
    await router.navigate("/checks?check=missing_check");
    expect(
      await screen.findByText("This check no longer exists"),
    ).toBeVisible();
  });

  it("sends the decisions of a check beyond the list to its own Decisions", async () => {
    // Not among the checks the list (and the Inbox) reads.
    const extra = makeAudit("old_wait", "project_a", "waiting_review", {
      updatedAt: "2026-09-01T10:00:00Z",
    });
    const { api } = serve(fixture(), { extra: { old_wait: extra } });
    renderApplication(api, "/checks?check=old_wait");
    const detail = await screen.findByRole("region", {
      name: "Selected check",
    });
    expect(
      await within(detail).findByRole("link", {
        name: "1 decision waiting for you",
      }),
    ).toHaveAttribute(
      "href",
      "/projects/project_a/audits/old_wait/reviews?state=pending",
    );
  });

  it("keeps listing what it could read when a project fails", async () => {
    const { api } = serve(fixture(), { failProject: "project_a" });
    renderApplication(api, "/checks");
    expect(
      await screen.findByText(
        "Checks of crapi-workshop could not be loaded; they are missing here.",
      ),
    ).toBeVisible();
    const list = screen.getByRole("region", { name: "Checks" });
    expect(
      within(list).getByRole("link", {
        name: "OWASP Top 10 · Source risks · Payment service",
      }),
    ).toBeVisible();
  });
});
