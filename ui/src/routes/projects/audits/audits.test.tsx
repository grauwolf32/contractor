import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeAll, describe, expect, it } from "vitest";

import type { ProjectFinding } from "../../../api/audits";
import type { PublicAPI } from "../../../api/client";
import { renderApplication } from "../../../test/application";
import * as build from "../../../test/builders";
import { fakeServer } from "../../../test/fake-server";
import { testAPI, testSession } from "../../../test/render";
import { json } from "../../../test/responses";

/** A PublicAPI over `handle` whose answers the fake Server checks. */
function checkedAPI(
  handle: (input: Request) => Response | Promise<Response>,
): PublicAPI {
  return testAPI(fakeServer({}, (request) => handle(request))).api;
}

const session = testSession(["user", "operations"]);

const project = build.project("project_example", {
  name: "Payment service",
  description: "Reusable service analysis",
  createdAt: "2026-09-06T10:00:00Z",
  updatedAt: "2026-09-06T10:00:00Z",
});

function findingAt(
  state: ProjectFinding["state"],
  revision: number,
): ProjectFinding {
  return build.finding(
    project.projectId,
    "finding_example",
    { state, revision },
    {
      title: "Missing object authorization",
      subject: { kind: "component", key: "orders" },
    },
  );
}

// The routes these tests open (the check page, the project's possible
// issues, Issues) and the Markdown preview of a possible issue's description
// load lazily. Without this, the first test to use each loads it inside its
// own 5 s budget: alone the first test took 265–285 ms instead of 96–99 ms
// and the first decision on Issues 582–672 ms instead of 250 ms; in the full
// parallel suite they took 770 ms and 867 ms (vitest JSON reports,
// 2026-10-23).
beforeAll(async () => {
  await Promise.all([
    import("./detail"),
    import("./findings"),
    import("../detail"),
    import("../../issues"),
    import("../../artifacts/previews/markdown"),
  ]);
});

describe("Project Audit routes", () => {
  it("lists canonical Project records without requiring their source checks and filters loaded rows", async () => {
    const first = findingAt(null, 1);
    const rated = build.finding(
      project.projectId,
      "finding_rated",
      {
        state: "confirmed",
        analystVerdict: "true_positive",
        analystSeverity: "low",
        rationale: "Checked",
        actorId: "analyst",
        assessedAt: first.createdAt,
      },
      { title: "Trace information exposure", severity_suggestion: "medium" },
    );
    const reads: string[] = [];
    const api = checkedAPI((request) => {
      const url = new URL(request.url);
      reads.push(url.pathname);
      if (url.pathname === "/v1/auth/session") return json(session);
      if (url.pathname === "/v1/projects/project_example")
        return json(project, 200, { ETag: '"1"' });
      if (url.pathname === "/v1/projects/project_example/findings")
        return json({
          items: [first, rated].map(build.findingSummary),
          projectRevision: 4,
          page: { hasMore: false },
        });
      throw new Error(`unexpected ${url}`);
    });
    const { router } = renderApplication("/projects/project_example/findings", {
      api,
    });
    const user = userEvent.setup();
    expect(
      await screen.findByText("2 of 2 possible issues loaded"),
    ).toBeVisible();
    expect(screen.getByRole("link", { name: first.title })).toHaveAttribute(
      "href",
      "/issues/project_example/finding_example?state=all&project=project_example",
    );
    expect(
      reads.some(
        (path) => path.startsWith("/v1/audits/") && path.endsWith("/findings"),
      ),
    ).toBe(false);
    await user.selectOptions(screen.getByLabelText("Severity"), "medium");
    expect(screen.getByText("No matching possible issues")).toBeVisible();
    await user.selectOptions(screen.getByLabelText("Severity"), "low");
    expect(screen.getByRole("link", { name: rated.title })).toBeVisible();
    expect(router.state.location.search).toBe("?severity=low");
    await user.click(screen.getByRole("button", { name: "Clear filters" }));
    await user.click(screen.getByRole("button", { name: "Not assessed 1" }));
    expect(screen.queryByRole("link", { name: rated.title })).toBeNull();
    expect(router.state.location.search).toBe("?state=unassessed");
  });
});

describe("Bounded Project finding collections", () => {
  it("loads five pages per explicit batch, retains a failed tail and refreshes from the head", async () => {
    const cursors: (string | null)[] = [];
    let fail = false;
    const api = checkedAPI((request) => {
      const url = new URL(request.url);
      if (url.pathname === "/v1/auth/session") return json(session);
      if (url.pathname === "/v1/projects/project_example")
        return json(project, 200, { ETag: '"1"' });
      if (url.pathname === "/v1/projects/project_example/findings") {
        const cursor = url.searchParams.get("cursor");
        cursors.push(cursor);
        if (fail && cursor !== null)
          return json(
            {
              code: "conflict",
              message: "Project changed",
              retryable: false,
              requestId: "stale",
            },
            409,
          );
        const ordinal = Number(cursor ?? 0);
        const row = build.finding(
          project.projectId,
          `finding_${ordinal}`,
          {},
          { title: `Finding ${ordinal}` },
        );
        return json({
          projectRevision: 1,
          items: [build.findingSummary(row)],
          page:
            ordinal < 5
              ? { hasMore: true, nextCursor: String(ordinal + 1) }
              : { hasMore: false },
        });
      }
      throw new Error(`unexpected ${url}`);
    });
    renderApplication("/projects/project_example/findings", { api });
    const user = userEvent.setup();
    expect(
      await screen.findByText("5 of 5 possible issues loaded"),
    ).toBeVisible();
    expect(cursors).toEqual([null, "1", "2", "3", "4"]);
    fail = true;
    await user.click(
      screen.getByRole("button", { name: "Load more possible issues" }),
    );
    expect(
      await screen.findByText(/Loading more failed: Project changed/),
    ).toBeVisible();
    expect(screen.getByRole("link", { name: "Finding 0" })).toBeVisible();
    fail = false;
    await user.click(screen.getByRole("button", { name: "Retry" }));
    expect(
      await screen.findByText("6 of 6 possible issues loaded"),
    ).toBeVisible();
    const before = cursors.length;
    await user.click(screen.getByRole("button", { name: "Refresh" }));
    await waitFor(() =>
      expect(cursors.slice(before)).toEqual([null, "1", "2", "3", "4"]),
    );
    expect(
      await screen.findByText("5 of 5 possible issues loaded"),
    ).toBeVisible();
  });
});
