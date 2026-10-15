import { fireEvent, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { renderApplication } from "../test/application";
import * as build from "../test/builders";
import { json, page } from "../test/fake-server";
import { CSRF_TOKEN } from "../test/render";

const evaluation = build.project("evaluation-openapi", {
  kind: "evaluation",
  name: "OpenAPI regression",
  description: "A/B evaluation workspace",
  createdAt: "2026-09-05T08:00:00Z",
  updatedAt: "2026-09-05T08:01:00Z",
});

describe("Evals and global Skills routes", () => {
  it("filters evaluation Projects and groups their ordinary Runs by eval metadata", async () => {
    const runs = [
      build.conforming("RunSummary", {
        runId: "run-eval-a",
        projectId: evaluation.projectId,
        workflow: "openapi-from-workspace@5",
        state: "succeeded",
        deletable: true,
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
      }),
      build.conforming("RunSummary", {
        runId: "run-eval-b",
        projectId: evaluation.projectId,
        workflow: "openapi-from-workspace@5",
        state: "running",
        deletable: false,
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
      }),
      build.conforming("RunSummary", {
        runId: "run-eval-unassigned",
        projectId: evaluation.projectId,
        workflow: "likec4-from-workspace@5",
        state: "failed",
        deletable: true,
        labels: { purpose: "eval" },
        createdAt: "2026-09-05T08:09:00Z",
        updatedAt: "2026-09-05T08:10:00Z",
        finishedAt: "2026-09-05T08:10:00Z",
      }),
    ];
    const view = renderApplication("/evals/legacy", {
      server: {
        "GET /v1/projects": () => json(page([evaluation])),
        "GET /v1/projects/{projectId}": () =>
          json(evaluation, 200, { ETag: '"1"' }),
        "GET /v1/projects/{projectId}/artifacts": () => json(page([])),
        "GET /v1/projects/{projectId}/runs": () => json(page(runs)),
        "GET /v1/workflows": () => json(page([])),
      },
    });
    const requests = view.server.requests;

    const evalLink = await screen.findByRole("link", {
      name: "OpenAPI regression",
    });
    expect(evalLink).toHaveAttribute("href", "/evals/evaluation-openapi");
    // The shell's Inbox badge lists kind=project on every page.
    const listRequest = requests.find((request) => {
      const url = new URL(request.url);
      return (
        url.pathname === "/v1/projects" &&
        url.searchParams.get("kind") !== "project"
      );
    });
    expect(new URL(listRequest!.url).searchParams.get("kind")).toBe(
      "evaluation",
    );

    await userEvent.setup().click(evalLink);
    expect(
      await screen.findByRole("heading", { name: "Eval Runs" }),
    ).toBeInTheDocument();
    expect(await screen.findByText("eval-2026-09-05")).toBeInTheDocument();
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
    expect(screen.queryByText(/ordinary Workflow Run/i)).toBeNull();
    expect(
      screen.getByText(/Succeeded describes execution only/u),
    ).toBeVisible();
    expect(screen.getByRole("link", { name: "All Runs →" })).toHaveAttribute(
      "href",
      "/runs",
    );

    // The legacy workspace keeps execution history and a read-only artifact
    // list; project-only regions (shortcuts, uploads, workflow launch) are gone.
    const sections = screen.getByRole("navigation", {
      name: "Project sections",
    });
    expect(
      within(sections)
        .getAllByRole("link")
        .map((link) => link.textContent),
    ).toEqual(["Runs", "Artifacts", "Workspace settings"]);
    expect(
      screen.getByRole("heading", { name: "Artifacts" }),
    ).toBeInTheDocument();
    expect(
      screen.getByText("No Artifact bindings in this workspace."),
    ).toBeVisible();
    expect(screen.queryByRole("button", { name: "Add sources" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Add artifact" })).toBeNull();
    expect(screen.queryByText("Artifact library")).toBeNull();
    expect(screen.queryByText("Format matches")).toBeNull();
    expect(screen.queryByText("All workflows")).toBeNull();
    expect(screen.queryByText("Missing inputs")).toBeNull();
    expect(
      requests.some(
        (request) => new URL(request.url).pathname === "/v1/workflows",
      ),
    ).toBe(false);
    const runsHeading = screen.getByRole("heading", { name: "Eval Runs" });
    const artifactsHeading = screen.getByRole("heading", { name: "Artifacts" });
    const settings = screen.getByText("Workspace settings", {
      selector: "summary",
    });
    expect(
      runsHeading.compareDocumentPosition(artifactsHeading) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    expect(
      artifactsHeading.compareDocumentPosition(settings) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });

  it("uses only UserScope Artifact APIs for global Skill packages", async () => {
    let created = false;
    const existing = build.conforming("ArtifactMetadata", {
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
    });
    const { server } = renderApplication("/catalog/skills", {
      server: {
        "GET /v1/artifacts": () =>
          json(
            page([
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
            ]),
          ),
        "PUT /v1/artifacts/{namespace}/{name}": () => {
          created = true;
          return json(
            build.conforming("ArtifactWriteResponse", {
              artifact: {
                namespace: "skills",
                name: "new-skill",
                revision: "skill-r2",
              },
              mediaType: "application/vnd.contractor.agent-skill+zip",
              size: 3,
            }),
            201,
            { ETag: '"skill-r2"' },
          );
        },
      },
    });
    const requests = server.requests;

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
    const uploadButton = screen.getByRole("button", { name: "Upload Skills" });
    expect(screen.queryByRole("dialog")).toBeNull();
    await user.click(uploadButton);
    const dialog = screen.getByRole("dialog", { name: "Upload Skills" });
    expect(within(dialog).queryByLabelText("Namespace")).toBeNull();
    expect(within(dialog).queryByLabelText("Media type")).toBeNull();
    expect(within(dialog).getByLabelText("Name")).toHaveFocus();
    await user.tab({ shift: true });
    expect(
      within(dialog).getByRole("button", { name: "Close upload dialog" }),
    ).toHaveFocus();
    await user.tab({ shift: true });
    expect(
      within(dialog).getByRole("button", { name: "Upload" }),
    ).toHaveFocus();
    await user.tab();
    expect(
      within(dialog).getByRole("button", { name: "Close upload dialog" }),
    ).toHaveFocus();
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(uploadButton).toHaveFocus();
    await user.click(uploadButton);
    fireEvent.drop(
      screen.getByLabelText("Drop a file here").closest(".artifact-file-drop")!,
      {
        dataTransfer: {
          files: [
            new File(["zip"], "suggested-skill.zip", {
              type: "application/zip",
            }),
          ],
        },
      },
    );
    expect(screen.getByLabelText("Name")).toHaveValue("suggested-skill");
    await user.click(screen.getByRole("button", { name: "Cancel" }));
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(uploadButton).toHaveFocus();
    await user.click(uploadButton);
    expect(screen.getByLabelText("Name")).toHaveValue("");
    await user.type(screen.getByLabelText("Name"), "new-skill");
    await user.click(screen.getByRole("button", { name: "Upload" }));
    expect(screen.getByRole("alert")).toHaveTextContent(
      "Choose a Skill ZIP package",
    );
    await user.upload(
      screen.getByLabelText("Drop a file here"),
      new File(["zip"], "new-skill.zip", { type: "application/zip" }),
    );
    await user.click(screen.getByRole("button", { name: "Upload" }));

    expect(
      await screen.findByRole("link", {
        name: /Open skills\/new-skill@skill-r2/,
      }),
    ).toHaveAttribute("href", "/artifacts/skills/new-skill?revision=skill-r2");
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(uploadButton).toHaveFocus();
    const upload = requests.find((request) => request.method === "PUT")!;
    expect(new URL(upload.url).pathname).toBe("/v1/artifacts/skills/new-skill");
    expect(upload.headers.get("Content-Type")).toBe(
      "application/vnd.contractor.agent-skill+zip",
    );
    expect(upload.headers.get("If-None-Match")).toBe("*");
    expect(upload.headers.get("X-CSRF-Token")).toBe(CSRF_TOKEN);
    expect(await upload.text()).toBe("zip");
    expect(
      requests.every((request) => !request.url.includes("/projects/")),
    ).toBe(true);
  });
});
