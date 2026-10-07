import {
  act,
  fireEvent,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { AuditProfile } from "../../../api/audits";
import { queryKeys } from "../../../api/query-keys";
import {
  asvsPilot,
  asvsReview,
  checklist,
  draftAudit,
  materialFixture,
  nuclei,
  openapiYaml,
  profileFixture,
  projectFixture,
  sourceZip,
  startedAudit,
  startResponse,
  top10,
  trace,
  unsupported,
  wstgLive,
  wstgLiveDetail,
} from "./test-fixtures";
import {
  CSRF_TOKEN,
  failure,
  gatewayFailure,
  json,
  page,
  renderStart,
} from "./test-support";

const START = "/checks/new?project=project_shop";

function list() {
  return screen.getByRole("region", { name: "Check types" });
}

function setup() {
  return screen.getByRole("region", { name: "Set up the check" });
}

/** Waits until the check types are grouped by readiness. */
async function waitForTypes() {
  const groups = await screen.findAllByRole("region", {
    name: /^(Ready with your materials|Needs more materials|Can't run on this server|All check types)$/,
  });
  return groups[0]!;
}

function startButton() {
  return screen.getByRole("button", { name: "Start check" });
}

async function bodyOf(request: Request | undefined): Promise<unknown> {
  if (request === undefined) throw new Error("request was not sent");
  return request.clone().json();
}

describe("Start a check: project", () => {
  it("asks for a project first and carries the other parameters over", async () => {
    const { user, router } = renderStart(
      "/checks/new?objective=Find+IDOR&type=owasp-top10-2025-source-risk",
      {
        projects: [
          projectFixture(),
          projectFixture({ projectId: "project_other", name: "Other" }),
        ],
        profiles: [top10],
        materials: [sourceZip],
      },
    );
    const projects = await screen.findByRole("region", { name: "Projects" });
    expect(
      within(projects).getByRole("heading", {
        level: 1,
        name: "Start a check",
      }),
    ).toBeVisible();
    expect(screen.getByText("Choose a project")).toBeVisible();
    await user.click(
      await within(projects).findByRole("link", { name: "Shop service" }),
    );
    await waitFor(() =>
      expect(
        new URLSearchParams(router.state.location.search).get("project"),
      ).toBe("project_shop"),
    );
    const params = new URLSearchParams(router.state.location.search);
    expect(params.get("objective")).toBe("Find IDOR");
    expect(params.get("type")).toBe("owasp-top10-2025-source-risk");
    expect(
      await screen.findByRole("heading", {
        level: 1,
        name: "Start a check on Shop service",
      }),
    ).toBeVisible();
    expect(
      await screen.findByLabelText("Your objective, in your own words"),
    ).toHaveValue("Find IDOR");
  });

  it("returns to the picker when the project does not exist", async () => {
    renderStart("/checks/new?project=project_gone", {
      project: null,
      projects: [projectFixture()],
      profiles: [top10],
    });
    expect(await screen.findByText(/This project was not found/)).toBeVisible();
    expect(
      await screen.findByRole("link", { name: "Shop service" }),
    ).toHaveAttribute("href", "/checks/new?project=project_shop");
  });
});

describe("Start a check: check types", () => {
  const catalog: AuditProfile[] = [
    trace,
    top10,
    checklist,
    wstgLive,
    unsupported,
  ];

  it("groups check types by readiness and links to what is missing", async () => {
    // A YAML API spec would also match the checklist's format, so the API
    // spec here is a ZIP bundle (the API spec input takes ZIP too).
    renderStart(START, {
      profiles: catalog,
      details: { "owasp-wstg-4-2-active-http@1": wstgLiveDetail },
      materials: [
        sourceZip,
        materialFixture("openapi-bundle", "application/zip"),
      ],
    });
    const ready = await screen.findByRole("region", {
      name: "Ready with your materials",
    });
    expect(
      within(ready)
        .getAllByRole("listitem")
        .map((row) => within(row).getAllByRole("link")[0]?.textContent),
    ).toEqual(["API endpoint trace", "OWASP Top 10 (2025) review"]);
    expect(within(ready).getByText("by file format")).toBeVisible();
    expect(
      within(ready).getByText("Uses API spec and source code"),
    ).toBeVisible();

    const needs = screen.getByRole("region", { name: "Needs more materials" });
    const live = within(needs)
      .getByRole("link", { name: "OWASP WSTG 4.2 live testing" })
      .closest("li")!;
    expect(live).toHaveTextContent(
      "Missing: Context brief (text or Markdown), Live target URL",
    );
    expect(
      within(live).getByRole("link", { name: "Add materials" }),
    ).toHaveAttribute("href", "/projects/project_shop/artifacts?add=artifact");
    expect(
      within(live).getByRole("link", { name: "Set the live target" }),
    ).toHaveAttribute("href", "/projects/project_shop/settings");
    const custom = within(needs)
      .getByRole("link", { name: "Custom checklist" })
      .closest("li")!;
    expect(custom).toHaveTextContent("Missing: Checklist (JSON or YAML)");
    expect(
      within(custom).queryByRole("link", { name: "Set the live target" }),
    ).toBeNull();

    const unavailable = screen.getByRole("region", {
      name: "Can't run on this server",
    });
    expect(
      within(unavailable).getByRole("link", { name: "Legacy multi round" }),
    ).toBeVisible();
  });

  it("shows why a check type cannot run when it is chosen", async () => {
    renderStart(`${START}&type=legacy-multi-round`, {
      profiles: catalog,
      materials: [sourceZip],
    });
    const pane = await screen.findByRole("region", {
      name: "Set up the check",
    });
    expect(
      await within(pane).findByText(
        "The server does not support more than one round.",
      ),
    ).toBeVisible();
    expect(startButton()).toBeDisabled();
    expect(
      screen.getByRole("button", { name: "Save as draft" }),
    ).toBeDisabled();
  });

  it("folds scope variants into one row and switches them under Scope", async () => {
    const { user, router } = renderStart(START, {
      profiles: [asvsReview, asvsPilot],
      materials: [sourceZip],
    });
    await waitForTypes();
    expect(
      within(list()).getAllByRole("link", { name: /OWASP ASVS/ }),
    ).toHaveLength(1);
    const scope = within(setup()).getByRole("group", { name: "Scope" });
    const full = within(scope).getByRole("radio", { name: /70 requirements/ });
    const pilot = within(scope).getByRole("radio", { name: /5 requirements/ });
    expect(full).toBeChecked();
    await user.click(pilot);
    await waitFor(() =>
      expect(
        new URLSearchParams(router.state.location.search).get("type"),
      ).toBe("owasp-asvs-5-0-l1-source-pilot"),
    );
    expect(
      await within(setup()).findByText("owasp-asvs-5-0-l1-source-pilot@1"),
    ).toBeVisible();
    expect(
      within(scope).getByRole("radio", { name: /5 requirements/ }),
    ).toBeChecked();
  });

  it("suggests a check type from the objective with a fixed reason", async () => {
    const { user, router } = renderStart(
      `${START}&objective=${encodeURIComponent("Check the shop APIs for authorization flaws")}`,
      { profiles: [trace, top10], materials: [sourceZip, openapiYaml] },
    );
    await waitForTypes();
    const row = within(list())
      .getByRole("link", { name: "API endpoint trace" })
      .closest("li")!;
    expect(within(row).getByText("Suggested")).toBeVisible();
    const pane = setup();
    expect(
      within(pane).getByRole("heading", {
        level: 2,
        name: "API endpoint trace",
      }),
    ).toBeVisible();
    expect(within(pane).getByText("openapi-operation-trace@1")).toBeVisible();
    expect(
      within(pane).getByText(/Why this fits:/).parentElement,
    ).toHaveTextContent(
      "Your objective mentions APIs, endpoints or access control, and this check type traces each endpoint of your API spec through the source code. Your project has a material whose format matches each input it needs.",
    );
    // "flaws" also fits the broad rule: one alternative as a text link.
    await user.click(
      within(pane).getByRole("link", { name: "OWASP Top 10 (2025) review" }),
    );
    await waitFor(() =>
      expect(
        new URLSearchParams(router.state.location.search).get("type"),
      ).toBe("owasp-top10-2025-source-risk"),
    );
    expect(
      await within(setup()).findByRole("heading", {
        level: 2,
        name: "OWASP Top 10 (2025) review",
      }),
    ).toBeVisible();
    expect(within(setup()).queryByText(/Why this fits:/)).toBeNull();
    expect(
      within(setup()).getByRole("link", { name: "Use API endpoint trace" }),
    ).toBeVisible();
  });

  it("follows the type parameter and says when it names nothing", async () => {
    renderStart(`${START}&type=missing-type`, {
      profiles: [trace, top10],
      materials: [sourceZip, openapiYaml],
    });
    await waitForTypes();
    expect(
      screen.getByText(/This server lists no check type named/),
    ).toBeVisible();
    const selected = within(list())
      .getAllByRole("link")
      .find((link) => link.getAttribute("aria-current") === "true");
    expect(selected).toHaveTextContent("API endpoint trace");
  });

  it("moves through check types with J and K", async () => {
    const { user, router } = renderStart(START, {
      profiles: [trace, top10, checklist],
      materials: [sourceZip, openapiYaml],
    });
    await waitForTypes();
    await user.keyboard("j");
    await waitFor(() =>
      expect(
        new URLSearchParams(router.state.location.search).get("type"),
      ).toBe("owasp-top10-2025-source-risk"),
    );
    await user.keyboard("j");
    await waitFor(() =>
      expect(
        new URLSearchParams(router.state.location.search).get("type"),
      ).toBe("source-checklist"),
    );
    await user.keyboard("k");
    await waitFor(() =>
      expect(
        new URLSearchParams(router.state.location.search).get("type"),
      ).toBe("owasp-top10-2025-source-risk"),
    );
  });

  it("keeps the check types and Start check when refreshing them fails", async () => {
    let unavailable = false;
    const { user } = renderStart(`${START}&type=owasp-top10-2025-source-risk`, {
      projects: [projectFixture()],
      profiles: [top10],
      profileList: () =>
        unavailable ? gatewayFailure(503) : json(page([top10])),
      materials: [sourceZip],
    });
    await waitForTypes();
    unavailable = true;
    // Through the project picker and back: the list is read again.
    await user.click(
      within(list()).getByRole("link", { name: "Change project" }),
    );
    await user.click(await screen.findByRole("link", { name: "Shop service" }));
    expect(
      await screen.findByText(
        "Check types could not be refreshed. The list shows the ones read before.",
      ),
    ).toBeVisible();
    expect(
      within(list()).getByRole("link", { name: "OWASP Top 10 (2025) review" }),
    ).toBeVisible();
    expect(startButton()).toBeEnabled();
    expect(screen.getByRole("button", { name: "Save as draft" })).toBeEnabled();
    unavailable = false;
    await user.click(within(list()).getByRole("button", { name: "Try again" }));
    await waitFor(() =>
      expect(screen.queryByText(/could not be refreshed/)).toBeNull(),
    );
  });

  it("looks for a named check type on pages not read yet", async () => {
    const { user } = renderStart(`${START}&type=source-checklist`, {
      profiles: [top10, checklist],
      profileList: (_request, url) =>
        url.searchParams.get("cursor") === null
          ? json(page([top10], { nextCursor: "2" }))
          : json(page([checklist])),
      materials: [sourceZip],
    });
    await waitForTypes();
    expect(
      screen.getByText(/is among those loaded so far\. Load more check types/),
    ).toBeVisible();
    await user.click(
      screen.getByRole("button", { name: "Load more check types" }),
    );
    expect(
      await within(setup()).findByRole("heading", {
        level: 2,
        name: "Custom checklist",
      }),
    ).toBeVisible();
    expect(screen.queryByText(/is among those loaded so far/)).toBeNull();
  });

  it("keeps the check types when loading more of them fails", async () => {
    let unavailable = true;
    const { user } = renderStart(`${START}&type=owasp-top10-2025-source-risk`, {
      profiles: [top10, checklist],
      profileList: (_request, url) =>
        url.searchParams.get("cursor") === null
          ? json(page([top10], { nextCursor: "2" }))
          : unavailable
            ? gatewayFailure(502)
            : json(page([checklist])),
      materials: [sourceZip],
    });
    await waitForTypes();
    await user.click(
      screen.getByRole("button", { name: "Load more check types" }),
    );
    expect(
      await screen.findByText("More check types could not be loaded."),
    ).toBeVisible();
    expect(
      within(list()).getByRole("link", { name: "OWASP Top 10 (2025) review" }),
    ).toBeVisible();
    expect(startButton()).toBeEnabled();
    unavailable = false;
    await user.click(within(list()).getByRole("button", { name: "Try again" }));
    expect(
      await within(list()).findByRole("link", { name: "Custom checklist" }),
    ).toBeVisible();
    expect(
      screen.queryByText("More check types could not be loaded."),
    ).toBeNull();
  });

  it("lets the user choose a scope the server can't run and shows why", async () => {
    const pilot: AuditProfile = {
      ...asvsPilot,
      serverCompatible: false,
      compatibilityReasons: ["multiple_rounds_unsupported"],
    };
    const { user, router } = renderStart(START, {
      profiles: [asvsReview, pilot],
      materials: [sourceZip],
    });
    await waitForTypes();
    const scope = within(setup()).getByRole("group", { name: "Scope" });
    const option = within(scope).getByRole("radio", {
      name: /5 requirements.*not supported by this server/,
    });
    expect(option).toBeEnabled();
    await user.click(option);
    await waitFor(() =>
      expect(
        new URLSearchParams(router.state.location.search).get("type"),
      ).toBe("owasp-asvs-5-0-l1-source-pilot"),
    );
    expect(
      await within(setup()).findByText(
        "The server does not support more than one round.",
      ),
    ).toBeVisible();
    expect(startButton()).toBeDisabled();
    expect(
      screen.getByRole("button", { name: "Save as draft" }),
    ).toBeDisabled();
  });

  it("links to every check type in the Library", async () => {
    renderStart(START, { profiles: [top10], materials: [sourceZip] });
    await waitForTypes();
    expect(
      within(list()).getByRole("link", { name: "See all check types" }),
    ).toHaveAttribute("href", "/catalog/audit-presets");
  });

  it("keeps the typed objective and the chosen type when changing the project", async () => {
    const { user } = renderStart(`${START}&type=owasp-top10-2025-source-risk`, {
      projects: [projectFixture()],
      profiles: [top10],
      materials: [sourceZip],
    });
    await waitForTypes();
    await user.type(
      screen.getByLabelText("Your objective, in your own words"),
      "Review sessions",
    );
    const change = within(list()).getByRole("link", { name: "Change project" });
    expect(change).toHaveAttribute(
      "href",
      "/checks/new?type=owasp-top10-2025-source-risk&objective=Review+sessions",
    );
    await user.click(change);
    await user.click(await screen.findByRole("link", { name: "Shop service" }));
    expect(
      await screen.findByLabelText("Your objective, in your own words"),
    ).toHaveValue("Review sessions");
    expect(
      await within(setup()).findByRole("heading", {
        level: 2,
        name: "OWASP Top 10 (2025) review",
      }),
    ).toBeVisible();
  });
});

describe("Start a check: materials", () => {
  it("attaches the only match and asks when several materials match", async () => {
    const second = materialFixture("shop-openapi-v2", "application/json");
    const { user } = renderStart(`${START}&type=openapi-operation-trace`, {
      profiles: [trace],
      materials: [sourceZip, openapiYaml, second],
    });
    await waitForTypes();
    const materials = within(setup()).getByRole("region", {
      name: "Materials",
    });
    const source = within(materials).getByText("Source code").closest("li")!;
    expect(source).toHaveTextContent("sources/shop-source@shop-source-r1");
    expect(source).toHaveTextContent(
      "Attached: the only material in a matching format.",
    );
    // The API spec input also takes ZIP, so three materials match by format.
    const spec = within(materials).getByRole("combobox", {
      name: "Material for API spec",
    });
    expect(spec).toHaveValue("");
    expect(within(spec).getAllByRole("option")).toHaveLength(4);
    expect(startButton()).toBeDisabled();
    expect(screen.getByText("Choose the API spec material.")).toBeVisible();
    await user.selectOptions(
      spec,
      within(spec).getByRole("option", {
        name: "sources/shop-openapi-v2@shop-openapi-v2-r1 · application/json",
      }),
    );
    expect(startButton()).toBeEnabled();
  });

  it("asks when one material is the only match of several inputs", async () => {
    const spec = materialFixture("openapi", "application/json");
    const { user } = renderStart(`${START}&type=openapi-nuclei-scan`, {
      profiles: [nuclei],
      materials: [spec],
    });
    await waitForTypes();
    const needs = screen.getByRole("region", { name: "Needs more materials" });
    expect(
      within(needs).getByRole("link", { name: "Nuclei scan" }).closest("li"),
    ).toHaveTextContent(
      "Missing: Another material for API spec or scan settings",
    );
    const materials = within(setup()).getByRole("region", {
      name: "Materials",
    });
    const specSelect = within(materials).getByRole("combobox", {
      name: "Material for API spec",
    });
    const settingsSelect = within(materials).getByRole("combobox", {
      name: "Material for Scan settings",
    });
    expect(specSelect).toHaveValue("");
    expect(settingsSelect).toHaveValue("");
    expect(startButton()).toBeDisabled();
    const option = within(specSelect).getByRole("option", {
      name: /^sources\/openapi@openapi-r1/,
    });
    await user.selectOptions(specSelect, option);
    await user.selectOptions(
      settingsSelect,
      within(settingsSelect).getByRole("option", {
        name: /^sources\/openapi@openapi-r1/,
      }),
    );
    expect(
      within(materials).getByText(
        /The same material is attached to API spec and scan settings\./,
      ),
    ).toBeVisible();
    expect(startButton()).toBeEnabled();
  });

  it("lets the user drop an optional material and keeps that choice", async () => {
    const notes = materialFixture("notes", "text/markdown");
    const profile = profileFixture("notes-review", {
      inputs: {
        source: { required: true, mediaTypes: ["application/zip"] },
        notes: { required: false, mediaTypes: ["text/*"] },
      },
    });
    const { user, sent } = renderStart(START, {
      profiles: [profile],
      materials: [sourceZip, notes],
    });
    await waitForTypes();
    await user.click(
      within(setup()).getByRole("button", { name: "Don't use Notes" }),
    );
    expect(
      within(setup()).getByRole("combobox", { name: "Material for Notes" }),
    ).toHaveValue("");
    await user.type(
      screen.getByLabelText("Your objective, in your own words"),
      "Review",
    );
    expect(
      within(setup()).getByRole("combobox", { name: "Material for Notes" }),
    ).toHaveValue("");
    await user.click(screen.getByRole("button", { name: "Save as draft" }));
    await waitFor(() => expect(sent("POST", "/audits")).toHaveLength(1));
    await expect(bodyOf(sent("POST", "/audits")[0])).resolves.toMatchObject({
      inputs: { source: sourceZip.artifact },
    });
  });

  it("reads a bounded number of pages and never treats a partial inventory as unique", async () => {
    let reads = 0;
    const { user } = renderStart(`${START}&type=owasp-top10-2025-source-risk`, {
      profiles: [top10],
      materials: (_request, url) => {
        reads += 1;
        const index = Number(url.searchParams.get("cursor") ?? "0");
        return json({
          items: [
            materialFixture(
              `source-${index}`,
              index === 0 ? "application/zip" : "text/plain",
            ),
          ],
          page: { hasMore: index < 4, nextCursor: String(index + 1) },
        });
      },
    });
    await waitForTypes();
    const materials = within(setup()).getByRole("region", {
      name: "Materials",
    });
    expect(reads).toBe(4);
    expect(
      within(materials).getByText(/Showing the first 4 materials/),
    ).toBeVisible();
    const select = within(materials).getByRole("combobox", {
      name: "Material for Source code",
    });
    expect(select).toHaveValue("");
    await user.click(
      within(materials).getByRole("button", { name: "Load more materials" }),
    );
    await waitFor(() => expect(reads).toBe(5));
    // Every page is read now, so the only ZIP is attached.
    expect(
      await within(materials).findByText("sources/source-0@source-0-r1"),
    ).toBeVisible();
  });

  it("reads a failed further page of materials again on retry", async () => {
    const reads: string[] = [];
    let unavailable = true;
    const { user } = renderStart(`${START}&type=owasp-top10-2025-source-risk`, {
      profiles: [top10],
      materials: (_request, url) => {
        const cursor = url.searchParams.get("cursor") ?? "0";
        reads.push(cursor);
        const index = Number(cursor);
        if (index === 4 && unavailable) return gatewayFailure(503);
        return json({
          items: [
            materialFixture(
              `source-${index}`,
              index === 0 ? "application/zip" : "text/plain",
            ),
          ],
          page: { hasMore: index < 4, nextCursor: String(index + 1) },
        });
      },
    });
    await waitForTypes();
    const materials = within(setup()).getByRole("region", {
      name: "Materials",
    });
    await user.click(
      within(materials).getByRole("button", { name: "Load more materials" }),
    );
    expect(
      await within(materials).findByText(
        "More of the project's materials could not be loaded.",
      ),
    ).toBeVisible();
    // A partial inventory still waits for a choice.
    expect(
      within(materials).getByRole("combobox", {
        name: "Material for Source code",
      }),
    ).toHaveValue("");
    unavailable = false;
    await user.click(
      within(materials).getByRole("button", {
        name: "Retry loading materials",
      }),
    );
    expect(
      await within(materials).findByText("sources/source-0@source-0-r1"),
    ).toBeVisible();
    // Only the failed page is read again.
    expect(reads).toEqual(["0", "1", "2", "3", "4", "4"]);
  });

  it("keeps attached materials when refreshing them fails", async () => {
    let unavailable = false;
    const { queryClient } = renderStart(
      `${START}&type=owasp-top10-2025-source-risk`,
      {
        profiles: [top10],
        materials: () =>
          unavailable ? gatewayFailure(503) : json(page([sourceZip])),
      },
    );
    await waitForTypes();
    const materials = within(setup()).getByRole("region", {
      name: "Materials",
    });
    expect(
      within(materials).getByText(
        "Attached: the only material in a matching format.",
      ),
    ).toBeVisible();
    unavailable = true;
    await act(async () => {
      await queryClient.refetchQueries({
        queryKey: queryKeys.projects.artifacts.picker("project_shop"),
      });
    });
    expect(
      await within(materials).findByText(
        "The project's materials could not be refreshed. The page uses the ones read before.",
      ),
    ).toBeVisible();
    expect(
      within(materials).getByText(
        "Attached: the only material in a matching format.",
      ),
    ).toBeVisible();
    expect(within(materials).queryByRole("combobox")).toBeNull();
    expect(startButton()).toBeEnabled();
  });
});

describe("Start a check: options and starting", () => {
  it("sends advanced options in the create request, then starts the check", async () => {
    const newer = { ...top10, ref: { ...top10.ref, version: "2" } };
    const { user, router, sent } = renderStart(
      `${START}&type=owasp-top10-2025-source-risk`,
      { profiles: [top10, newer], materials: [sourceZip] },
    );
    await waitForTypes();
    await user.type(
      screen.getByLabelText("Your objective, in your own words"),
      "Review session handling",
    );
    await user.click(screen.getByText("Advanced options"));
    const versionSelect = screen.getByLabelText("Check type version");
    expect(versionSelect).toHaveValue("2");
    await user.selectOptions(versionSelect, "1");
    expect(
      await screen.findByText("owasp-top10-2025-source-risk@1"),
    ).toBeVisible();
    expect(
      screen.getByText("Standards pinned at start: owasp-web-top10@2025"),
    ).toBeVisible();
    await user.type(screen.getByLabelText("Target"), "checkout service");
    await user.type(
      screen.getByLabelText("Authorization scope"),
      "Staging only",
    );
    await user.type(screen.getByLabelText("Runtime labels"), "debug, caido");
    await user.click(startButton());

    await waitFor(() =>
      expect(router.state.location.pathname).toBe(
        "/projects/project_shop/audits/audit_new",
      ),
    );
    const [create] = sent("POST", "/projects/project_shop/audits");
    expect(create?.headers.get("Idempotency-Key")).toMatch(/^create-audit-ui-/);
    expect(create?.headers.get("X-CSRF-Token")).toBe(CSRF_TOKEN);
    await expect(bodyOf(create)).resolves.toEqual({
      profile: { name: "owasp-top10-2025-source-risk", version: "1" },
      inputs: { source: sourceZip.artifact },
      runtimeLabels: ["debug", "caido"],
      scope: {
        objective: "Review session handling",
        target: "checkout service",
        authorizationScope: "Staging only",
      },
    });
    const [start] = sent("POST", "/v1/audits/audit_new/start");
    expect(start?.headers.get("If-Match")).toBe('"1"');
    expect(start?.headers.get("Idempotency-Key")).toMatch(/^start-audit-ui-/);
    expect(start?.headers.get("X-CSRF-Token")).toBe(CSRF_TOKEN);
    await expect(bodyOf(start)).resolves.toEqual({ deadlineSeconds: 86400 });
  });

  it.each([
    ["24 hours", undefined, 86400],
    ["7 days", undefined, 604800],
    ["No time limit", undefined, 0],
    ["Custom", "1.5", 5400],
  ] as const)(
    "starts with the time limit %s",
    async (choice, hours, seconds) => {
      const { user, sent } = renderStart(
        `${START}&type=owasp-top10-2025-source-risk`,
        {
          profiles: [top10],
          materials: [sourceZip],
        },
      );
      await waitForTypes();
      const select = screen.getByLabelText("Time limit");
      expect(select).toHaveValue("86400");
      await user.selectOptions(select, choice);
      if (hours !== undefined) {
        const field = screen.getByLabelText("Time limit in hours");
        await user.clear(field);
        await user.type(field, hours);
      }
      await user.click(startButton());
      await waitFor(() => expect(sent("POST", "/start")).toHaveLength(1));
      await expect(bodyOf(sent("POST", "/start")[0])).resolves.toEqual({
        deadlineSeconds: seconds,
      });
    },
  );

  it("refuses a custom time limit outside 0.01 to 8760 hours", async () => {
    const { user } = renderStart(`${START}&type=owasp-top10-2025-source-risk`, {
      profiles: [top10],
      materials: [sourceZip],
    });
    await waitForTypes();
    await user.selectOptions(screen.getByLabelText("Time limit"), "Custom");
    const field = screen.getByLabelText("Time limit in hours");
    await user.clear(field);
    await user.type(field, "9000");
    expect(
      screen.getByText(
        "Enter a time limit from 0.01 to 8760 hours (365 days).",
      ),
    ).toHaveAttribute("role", "alert");
    expect(field).toHaveAttribute("aria-invalid", "true");
    expect(startButton()).toBeDisabled();
    // A draft does not need a time limit.
    expect(screen.getByRole("button", { name: "Save as draft" })).toBeEnabled();
  });

  it("saves a draft without starting it", async () => {
    const { user, router, sent } = renderStart(
      `${START}&type=owasp-top10-2025-source-risk`,
      { profiles: [top10], materials: [sourceZip] },
    );
    await waitForTypes();
    await user.click(screen.getByRole("button", { name: "Save as draft" }));
    await waitFor(() =>
      expect(router.state.location.pathname).toBe(
        "/projects/project_shop/audits/audit_new",
      ),
    );
    expect(sent("POST", "/projects/project_shop/audits")).toHaveLength(1);
    expect(sent("POST", "/start")).toHaveLength(0);
  });

  it("starts from the objective with Ctrl+Enter", async () => {
    const { user, router, sent } = renderStart(
      `${START}&type=owasp-top10-2025-source-risk`,
      { profiles: [top10], materials: [sourceZip] },
    );
    await waitForTypes();
    expect(startButton()).toHaveAttribute(
      "aria-keyshortcuts",
      "Control+Enter Meta+Enter",
    );
    await user.type(
      screen.getByLabelText("Your objective, in your own words"),
      "Quick risk review",
    );
    await user.keyboard("{Control>}{Enter}{/Control}");
    await waitFor(() =>
      expect(router.state.location.pathname).toBe(
        "/projects/project_shop/audits/audit_new",
      ),
    );
    expect(sent("POST", "/start")).toHaveLength(1);
  });

  it("starts with Ctrl+Enter from the other fields, not from a link", async () => {
    const { user, router, sent } = renderStart(
      `${START}&type=owasp-top10-2025-source-risk`,
      { profiles: [top10], materials: [sourceZip] },
    );
    await waitForTypes();
    // On a link the key belongs to the link: the page leaves the event
    // alone, so Ctrl/⌘+Enter opens the link in a new tab.
    const link = within(list()).getByRole("link", { name: "Change project" });
    link.focus();
    expect(fireEvent.keyDown(link, { key: "Enter", ctrlKey: true })).toBe(true);
    expect(fireEvent.keyDown(link, { key: "Enter", metaKey: true })).toBe(true);
    await user.keyboard("{Control>}{Enter}{/Control}");
    expect(sent("POST", "/audits")).toHaveLength(0);
    await user.click(screen.getByText("Advanced options"));
    await user.type(screen.getByLabelText("Runtime labels"), "debug");
    await user.keyboard("{Control>}{Enter}{/Control}");
    await waitFor(() =>
      expect(router.state.location.pathname).toBe(
        "/projects/project_shop/audits/audit_new",
      ),
    );
    await expect(bodyOf(sent("POST", "/audits")[0])).resolves.toMatchObject({
      runtimeLabels: ["debug"],
    });
    expect(sent("POST", "/start")).toHaveLength(1);
  });

  it("fills the target from the project's live target and needs the authorization scope", async () => {
    const context = materialFixture("brief", "text/markdown");
    const { user, sent } = renderStart(
      `${START}&type=owasp-wstg-4-2-active-http`,
      {
        project: projectFixture({
          httpTarget: { url: "https://staging.example.com" },
        }),
        profiles: [wstgLive],
        details: { "owasp-wstg-4-2-active-http@1": wstgLiveDetail },
        materials: [context],
      },
    );
    await waitForTypes();
    const live = within(setup()).getByRole("region", { name: "Live target" });
    expect(within(live).getByLabelText("Target")).toHaveValue(
      "https://staging.example.com",
    );
    expect(startButton()).toBeDisabled();
    expect(screen.getByText("Describe the authorization scope.")).toBeVisible();
    await user.type(
      within(live).getByLabelText("Authorization scope"),
      "Staging only",
    );
    await user.click(startButton());
    await waitFor(() => expect(sent("POST", "/audits")).toHaveLength(1));
    await expect(bodyOf(sent("POST", "/audits")[0])).resolves.toMatchObject({
      scope: {
        target: "https://staging.example.com",
        authorizationScope: "Staging only",
      },
    });
  });
});

describe("Start a check: live target", () => {
  it("lets the user enter a target when the project has none", async () => {
    const context = materialFixture("brief", "text/markdown");
    const { user, sent } = renderStart(
      `${START}&type=owasp-wstg-4-2-active-http`,
      {
        profiles: [wstgLive],
        details: { "owasp-wstg-4-2-active-http@1": wstgLiveDetail },
        materials: [context],
      },
    );
    await screen.findByRole("region", { name: "Needs more materials" });
    const pane = setup();
    expect(
      within(pane).getByText("Needs more materials: Live target URL"),
    ).toBeVisible();
    expect(
      within(pane)
        .getAllByRole("link", { name: "Set the live target" })
        .every(
          (link) =>
            link.getAttribute("href") === "/projects/project_shop/settings",
        ),
    ).toBe(true);
    const live = within(pane).getByRole("region", { name: "Live target" });
    expect(within(live).getByLabelText("Target")).toHaveValue("");
    expect(screen.getByText("Enter the target to test.")).toBeVisible();
    await user.type(
      within(live).getByLabelText("Target"),
      "https://test.example.com",
    );
    await user.type(
      within(live).getByLabelText("Authorization scope"),
      "Test host only",
    );
    await user.click(startButton());
    await waitFor(() => expect(sent("POST", "/audits")).toHaveLength(1));
    await expect(bodyOf(sent("POST", "/audits")[0])).resolves.toMatchObject({
      inputs: { context: context.artifact },
      scope: {
        target: "https://test.example.com",
        authorizationScope: "Test host only",
      },
    });
  });
});

describe("Start a check: lost and refused answers", () => {
  it("retries a lost create request with the same key, then starts", async () => {
    let creates = 0;
    const { user, router, sent } = renderStart(
      `${START}&type=owasp-top10-2025-source-risk`,
      {
        profiles: [top10],
        materials: [sourceZip],
        create: () => {
          creates += 1;
          if (creates === 1) return Promise.reject(new TypeError("offline"));
          return json(draftAudit(top10), 201, { ETag: '"1"' });
        },
      },
    );
    await waitForTypes();
    await user.click(startButton());
    expect(
      await screen.findByText("The check may already exist."),
    ).toBeVisible();
    expect(sent("POST", "/start")).toHaveLength(0);
    await user.click(
      screen.getByRole("button", { name: "Retry same request" }),
    );
    await waitFor(() =>
      expect(router.state.location.pathname).toBe(
        "/projects/project_shop/audits/audit_new",
      ),
    );
    const keys = sent("POST", "/projects/project_shop/audits").map((request) =>
      request.headers.get("Idempotency-Key"),
    );
    expect(keys).toHaveLength(2);
    expect(keys[1]).toBe(keys[0]);
    expect(sent("POST", "/start")).toHaveLength(1);
  });

  it("warns that a changed form after a lost answer uses a new key", async () => {
    const { user, sent } = renderStart(
      `${START}&type=owasp-top10-2025-source-risk`,
      {
        profiles: [top10],
        materials: [sourceZip],
        create: () => Promise.reject(new TypeError("offline")),
      },
    );
    await waitForTypes();
    await user.click(screen.getByRole("button", { name: "Save as draft" }));
    await screen.findByText("The check may already exist.");
    await user.type(
      screen.getByLabelText("Your objective, in your own words"),
      "Changed",
    );
    expect(
      screen.getByText(/The form changed after the answer was lost/),
    ).toBeVisible();
    expect(
      screen.queryByRole("button", { name: "Retry same request" }),
    ).toBeNull();
    await user.click(screen.getByRole("button", { name: "Save as draft" }));
    await waitFor(() => expect(sent("POST", "/audits")).toHaveLength(2));
    const [first, second] = sent("POST", "/audits").map((request) =>
      request.headers.get("Idempotency-Key"),
    );
    expect(second).not.toBe(first);
  });

  it("retries a lost start with the same key and never creates twice", async () => {
    let starts = 0;
    const { user, router, sent } = renderStart(
      `${START}&type=owasp-top10-2025-source-risk`,
      {
        profiles: [top10],
        materials: [sourceZip],
        start: () => {
          starts += 1;
          if (starts === 1) return Promise.reject(new TypeError("offline"));
          return json(startResponse(draftAudit(top10)), 200, { ETag: '"2"' });
        },
      },
    );
    await waitForTypes();
    await user.click(startButton());
    expect(
      await screen.findByText("The check may already have started."),
    ).toBeVisible();
    expect(
      screen.getByRole("link", { name: "Open the check" }),
    ).toHaveAttribute("href", "/projects/project_shop/audits/audit_new");
    await user.click(
      screen.getByRole("button", { name: "Retry same request" }),
    );
    await waitFor(() =>
      expect(router.state.location.pathname).toBe(
        "/projects/project_shop/audits/audit_new",
      ),
    );
    expect(sent("POST", "/projects/project_shop/audits")).toHaveLength(1);
    const starts_ = sent("POST", "/start");
    expect(starts_).toHaveLength(2);
    expect(starts_[1]?.headers.get("Idempotency-Key")).toBe(
      starts_[0]?.headers.get("Idempotency-Key"),
    );
    expect(starts_[1]?.headers.get("If-Match")).toBe('"1"');
  });

  it("keeps the draft when the server refuses the start and starts it again", async () => {
    let starts = 0;
    const { user, router, sent } = renderStart(
      `${START}&type=owasp-top10-2025-source-risk`,
      {
        profiles: [top10],
        materials: [sourceZip],
        start: () => {
          starts += 1;
          return starts === 1
            ? failure(422, "invalid_input", "The source archive is empty")
            : json(startResponse(draftAudit(top10)), 200, { ETag: '"2"' });
        },
      },
    );
    await waitForTypes();
    await user.click(startButton());
    expect(
      await screen.findByText(
        "The check was saved as a draft but did not start.",
      ),
    ).toBeVisible();
    expect(screen.getByText("The source archive is empty")).toBeVisible();
    // Said only after a fresh read showed the draft.
    expect(sent("GET", "/v1/audits/audit_new")).toHaveLength(1);
    expect(
      screen.getByRole("link", { name: "Open the draft" }),
    ).toHaveAttribute("href", "/projects/project_shop/audits/audit_new");
    expect(
      screen.getByText(/Start check tries this draft again\./),
    ).toBeVisible();
    await user.click(startButton());
    await waitFor(() =>
      expect(router.state.location.pathname).toBe(
        "/projects/project_shop/audits/audit_new",
      ),
    );
    expect(sent("POST", "/projects/project_shop/audits")).toHaveLength(1);
    expect(sent("POST", "/start")).toHaveLength(2);
  });

  it("reads the check again after a refused start and stops when it already runs", async () => {
    // The first start reached the server, but its answer was lost.
    let starts = 0;
    let running = false;
    const { user, sent } = renderStart(
      `${START}&type=owasp-top10-2025-source-risk`,
      {
        profiles: [top10],
        materials: [sourceZip],
        start: () => {
          starts += 1;
          if (starts === 1) {
            running = true;
            return Promise.reject(new TypeError("offline"));
          }
          return failure(412, "precondition_failed", "Audit revision changed");
        },
        audit: () => {
          const draft = draftAudit(top10);
          const audit = running ? startedAudit(draft) : draft;
          return json(audit, 200, { ETag: `"${audit.revision}"` });
        },
      },
    );
    await waitForTypes();
    await user.click(startButton());
    await screen.findByText("The check may already have started.");
    // A new time limit is a new start request (new key, same revision).
    await user.selectOptions(screen.getByLabelText("Time limit"), "7 days");
    await user.click(startButton());
    expect(
      await screen.findByText("This check has already started."),
    ).toBeVisible();
    expect(screen.getByText(/The server shows it as Running\./)).toBeVisible();
    expect(
      screen.getByRole("link", { name: "Open the check" }),
    ).toHaveAttribute("href", "/projects/project_shop/audits/audit_new");
    expect(screen.queryByText(/did not start/)).toBeNull();
    expect(startButton()).toBeDisabled();
    expect(
      screen.getByRole("button", { name: "Save as draft" }),
    ).toBeDisabled();
    await user.keyboard("{Control>}{Enter}{/Control}");

    const startRequests = sent("POST", "/start");
    expect(startRequests).toHaveLength(2);
    expect(
      startRequests.map((request) => request.headers.get("If-Match")),
    ).toEqual(['"1"', '"1"']);
    expect(startRequests[1]?.headers.get("Idempotency-Key")).not.toBe(
      startRequests[0]?.headers.get("Idempotency-Key"),
    );
    await expect(bodyOf(startRequests[1])).resolves.toEqual({
      deadlineSeconds: 604800,
    });
    expect(sent("GET", "/v1/audits/audit_new")).toHaveLength(1);
    expect(sent("POST", "/projects/project_shop/audits")).toHaveLength(1);
  });

  it("starts the newer revision after the draft changed on the server", async () => {
    let starts = 0;
    const changed = draftAudit(top10, { revision: 3 });
    const { user, router, sent } = renderStart(
      `${START}&type=owasp-top10-2025-source-risk`,
      {
        profiles: [top10],
        materials: [sourceZip],
        start: () => {
          starts += 1;
          return starts === 1
            ? failure(412, "precondition_failed", "Audit revision changed")
            : json(startResponse(changed), 200, { ETag: '"4"' });
        },
        audit: () => json(changed, 200, { ETag: '"3"' }),
      },
    );
    await waitForTypes();
    await user.click(startButton());
    expect(
      await screen.findByText(
        "The check was saved as a draft but did not start.",
      ),
    ).toBeVisible();
    expect(
      screen.getByText(
        /The check changed on the server after this page read it\. The page has read it again; review it before you start it again\./,
      ),
    ).toBeVisible();
    await user.click(startButton());
    await waitFor(() =>
      expect(router.state.location.pathname).toBe(
        "/projects/project_shop/audits/audit_new",
      ),
    );
    expect(
      sent("POST", "/start").map((request) => request.headers.get("If-Match")),
    ).toEqual(['"1"', '"3"']);
  });

  it("does not say a check did not start after a gateway failure", async () => {
    let starts = 0;
    const { user, router, sent } = renderStart(
      `${START}&type=owasp-top10-2025-source-risk`,
      {
        profiles: [top10],
        materials: [sourceZip],
        start: () => {
          starts += 1;
          return starts === 1
            ? gatewayFailure(504)
            : json(startResponse(draftAudit(top10)), 200, { ETag: '"2"' });
        },
      },
    );
    await waitForTypes();
    await user.click(startButton());
    expect(
      await screen.findByText("The check may already have started."),
    ).toBeVisible();
    expect(screen.getByText("The start request failed.")).toBeVisible();
    expect(
      screen.getByText(/The server still shows it as a draft/),
    ).toBeVisible();
    expect(screen.queryByText(/did not start/)).toBeNull();
    expect(sent("GET", "/v1/audits/audit_new")).toHaveLength(1);
    await user.click(
      screen.getByRole("button", { name: "Retry same request" }),
    );
    await waitFor(() =>
      expect(router.state.location.pathname).toBe(
        "/projects/project_shop/audits/audit_new",
      ),
    );
    const startRequests = sent("POST", "/start");
    expect(startRequests).toHaveLength(2);
    expect(startRequests[1]?.headers.get("Idempotency-Key")).toBe(
      startRequests[0]?.headers.get("Idempotency-Key"),
    );
    expect(startRequests[1]?.headers.get("If-Match")).toBe('"1"');
  });

  it("holds Start check after a lost start answer and a changed form", async () => {
    let starts = 0;
    const { user, router, sent } = renderStart(
      `${START}&type=owasp-top10-2025-source-risk`,
      {
        profiles: [top10],
        materials: [sourceZip],
        start: () => {
          starts += 1;
          if (starts === 1) return Promise.reject(new TypeError("offline"));
          return json(startResponse(draftAudit(top10)), 200, { ETag: '"2"' });
        },
      },
    );
    await waitForTypes();
    await user.click(startButton());
    await screen.findByText("The check may already have started.");
    expect(screen.getByText(/it cannot start the check twice/)).toBeVisible();
    await user.type(
      screen.getByLabelText("Your objective, in your own words"),
      "Changed",
    );
    expect(
      screen.getByText(
        /The form changed since, so Start check would create and start a second check while the first one may be running\./,
      ),
    ).toBeVisible();
    expect(startButton()).toBeDisabled();
    expect(
      screen.getByText(
        "The first check may already be running. Retry the first request or open the check before you start another one.",
      ),
    ).toBeVisible();
    await user.keyboard("{Control>}{Enter}{/Control}");
    expect(sent("POST", "/projects/project_shop/audits")).toHaveLength(1);
    expect(sent("POST", "/start")).toHaveLength(1);
    // The first request, unchanged, settles it.
    await user.click(
      screen.getByRole("button", { name: "Retry same request" }),
    );
    await waitFor(() =>
      expect(router.state.location.pathname).toBe(
        "/projects/project_shop/audits/audit_new",
      ),
    );
    expect(sent("POST", "/projects/project_shop/audits")).toHaveLength(1);
    const startRequests = sent("POST", "/start");
    expect(startRequests).toHaveLength(2);
    expect(startRequests[1]?.headers.get("Idempotency-Key")).toBe(
      startRequests[0]?.headers.get("Idempotency-Key"),
    );
  });

  it("shows a refused create with the server's message", async () => {
    const { user, sent } = renderStart(
      `${START}&type=owasp-top10-2025-source-risk`,
      {
        profiles: [top10],
        materials: [sourceZip],
        create: () => failure(409, "conflict", "The project changed"),
      },
    );
    await waitForTypes();
    await user.click(startButton());
    expect(await screen.findByText("The check was not created.")).toBeVisible();
    expect(screen.getByText("The project changed")).toBeVisible();
    expect(sent("POST", "/start")).toHaveLength(0);
  });
});
