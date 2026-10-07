import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter } from "react-router";
import { describe, expect, it, vi } from "vitest";

import type { AuditStandard } from "../../api/audit-presets";
import type { AuditProfile } from "../../api/audits";
import { PublicAPI, type AuthSession } from "../../api/client";
import { Application } from "../../app/application";
import { applicationRoutes } from "../../app/router";
import {
  dynamicPresetFixture,
  newerPresetFixture,
  presetFixture,
  standardFixture,
} from "../../test/audit-presets-fixture";
import { auditPresetLabel } from "../projects/audits/labels";

const session: AuthSession = {
  principal: { userId: "owner", username: "owner", capabilities: ["user"] },
  csrfToken: "a".repeat(43),
  idleExpiresAt: "2099-01-01T00:00:00Z",
  absoluteExpiresAt: "2099-01-02T00:00:00Z",
};

function setup(
  path: string,
  options: {
    profiles?: AuditProfile[];
    standard?: AuditStandard;
    failStandard?: boolean;
    failCatalog?: boolean;
    repeatCursor?: boolean;
  } = {},
) {
  const profiles = options.profiles ?? [
    presetFixture,
    newerPresetFixture,
    dynamicPresetFixture,
  ];
  const fetcher = vi.fn(async (input: RequestInfo | URL) => {
    const request = input instanceof Request ? input : new Request(input);
    const url = new URL(request.url);
    let body: unknown = { items: [], page: { hasMore: false } };
    let status = 200;
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      "X-Contractor-API-Version": "contractor.public.v1",
    };
    if (url.pathname === "/v1/audit-profiles") {
      if (options.failCatalog) {
        status = 503;
        body = {
          code: "unavailable",
          message: "Preset catalog unavailable",
          retryable: true,
        };
      } else {
        const later = url.searchParams.has("cursor");
        body = {
          items: later ? profiles.slice(1) : profiles.slice(0, 1),
          page:
            profiles.length > 1 && (!later || options.repeatCursor)
              ? { hasMore: true, nextCursor: "page-2" }
              : { hasMore: false },
        };
      }
    } else if (url.pathname.startsWith("/v1/audit-profiles/")) {
      const profile = profiles.find(
        (item) =>
          url.pathname ===
          `/v1/audit-profiles/${item.ref.name}/versions/${item.ref.version}`,
      );
      if (profile) {
        body = profile;
        headers.ETag = `"${profile.ref.digest}"`;
      } else {
        status = 404;
        body = {
          code: "not_found",
          message: "Preset not found",
          retryable: false,
        };
      }
    } else if (url.pathname.startsWith("/v1/audit-standards/")) {
      body = {
        apiVersion: "contractor/v1alpha1",
        standard: options.standard ?? standardFixture,
      };
      if (options.failStandard) {
        status = 503;
        body = {
          code: "unavailable",
          message: "Standard unavailable",
          retryable: true,
        };
      }
    } else if (url.pathname.startsWith("/v1/workflows/")) {
      body = {
        ref: { name: "source-check", version: "3" },
        entryStage: "check",
        parameters: {},
        inputs: {},
        outputs: {},
        stages: {},
      };
    }
    return new Response(JSON.stringify(body), { status, headers });
  });
  const publicAPI = new PublicAPI(
    {
      uiVersion: "0.1.0",
      supportedApiVersions: ["contractor.public.v1"],
      apiBaseUrl: "http://127.0.0.1:8080",
    },
    fetcher,
  );
  const router = createMemoryRouter(applicationRoutes(), {
    initialEntries: [path],
  });
  render(
    <Application
      api={{
        getSession: async () => session,
        login: async () => session,
        logout: async () => undefined,
      }}
      publicAPI={publicAPI}
      router={router}
    />,
  );
  return { router, fetcher };
}

describe("Library check types", () => {
  it("keeps Fast WSTG variants separate and describes their limited coverage", async () => {
    const profiles: AuditProfile[] = [
      "owasp-wstg-4-2-source-review",
      "owasp-wstg-4-2-fast-source-review",
      "owasp-wstg-4-2-fast-active-http",
    ].map((name) => ({
      ...presetFixture,
      ref: { ...presetFixture.ref, name },
      inventory: {
        implementation: "standard-mappings@1",
        itemWorkflowRole: "check",
      },
      execution: {
        ...presetFixture.execution,
        maxItemsTotal: name.includes("fast") ? 16 : 94,
      },
    }));
    setup("/catalog/audit-presets", { profiles });
    expect(
      await screen.findByRole("link", {
        name: auditPresetLabel("owasp-wstg-4-2-fast-active-http"),
      }),
    ).toHaveAttribute(
      "href",
      "/catalog/audit-presets/owasp-wstg-4-2-fast-active-http/1",
    );
    expect(screen.getAllByRole("article")).toHaveLength(3);
    expect(
      screen.getByRole("link", {
        name: auditPresetLabel("owasp-wstg-4-2-fast-source-review"),
      }),
    ).toBeVisible();
    expect(
      screen.getByRole("link", {
        name: auditPresetLabel("owasp-wstg-4-2-source-review"),
      }),
    ).toBeVisible();
    expect(
      screen.getAllByText(
        "16 priority WSTG scenarios. Focused first pass; other scenarios are outside scope.",
      ),
    ).toHaveLength(2);
  });

  it("groups all pages by name, selects the highest version and finds later-page check types", async () => {
    const user = userEvent.setup();
    const { router } = setup("/catalog/audit-presets?keep=yes");
    expect(
      await screen.findByLabelText("Version of source-review"),
    ).toHaveValue("10");
    expect(screen.getAllByRole("article")).toHaveLength(2);
    expect(
      within(
        screen.getByRole("navigation", { name: "Library sections" }),
      ).getByRole("link", { name: "Check types" }),
    ).toHaveAttribute("aria-current", "page");
    await user.selectOptions(
      screen.getByLabelText("Version of source-review"),
      "1",
    );
    expect(screen.getByRole("link", { name: "source review" })).toHaveAttribute(
      "href",
      "/catalog/audit-presets/source-review/1",
    );
    expect(screen.getByText("source-review@1")).toBeVisible();
    await user.type(screen.getByLabelText("Search check types"), "OpenAPI");
    await waitFor(() => expect(screen.getAllByRole("article")).toHaveLength(1));
    await user.click(screen.getByRole("link", { name: "View details" }));
    await screen.findByText(
      "The endpoint list depends on the inputs of each check.",
    );
    await user.click(screen.getByRole("link", { name: "All check types" }));
    expect(router.state.location.search).toBe("?keep=yes&q=OpenAPI");
    expect(await screen.findByLabelText("Search check types")).toHaveValue(
      "OpenAPI",
    );
  });

  it("names check type items by their kind and counts them", async () => {
    setup("/catalog/audit-presets");
    const card = (
      await screen.findByRole("link", { name: "source review" })
    ).closest("article")!;
    expect(within(card).getByText("Requirements verification")).toBeVisible();
    expect(within(card).getByText("Available")).toBeVisible();
    expect(
      within(card).getByRole("link", { name: "View requirements" }),
    ).toHaveAttribute("href", "/catalog/audit-presets/source-review/10");
    expect(
      within(card).getByRole("button", { name: "Copy check type version" }),
    ).toBeVisible();
    const dynamic = screen
      .getByRole("link", { name: auditPresetLabel("openapi-operation-trace") })
      .closest("article")!;
    expect(within(dynamic).getByText("Endpoint tracing")).toBeVisible();
    expect(
      within(dynamic).getByText(
        "One endpoint per operation in your OpenAPI document.",
      ),
    ).toBeVisible();
    expect(screen.getByText("2 check types")).toBeVisible();
  });

  it("names the items of a WSTG check type scenarios and counts them", async () => {
    const user = userEvent.setup();
    const name = "owasp-wstg-4-2-source-review";
    const profile: AuditProfile = {
      ...presetFixture,
      ref: { ...presetFixture.ref, name },
      standards: [{ scheme: "owasp-wstg", version: "4.2" }],
      inventory: {
        ...presetFixture.inventory,
        standardSelection: {
          ...presetFixture.inventory.standardSelection!,
          scope: "Selected source review tests",
        },
      },
    };
    setup("/catalog/audit-presets", {
      profiles: [profile],
      standard: {
        ...standardFixture,
        reference: { scheme: "owasp-wstg", version: "4.2" },
        title: "Web Security Testing Guide",
      },
    });
    const card = (
      await screen.findByRole("link", { name: auditPresetLabel(name) })
    ).closest("article")!;
    const view = within(card).getByRole("link", { name: "View scenarios" });
    expect(view).toHaveAttribute("href", `/catalog/audit-presets/${name}/1`);

    await user.click(view);
    await screen.findByText("Check authorization");
    expect(screen.getByRole("heading", { name: "Scenarios" })).toBeVisible();
    expect(screen.getByText("2 scenarios")).toBeVisible();
    expect(screen.getByText("2 scenarios selected · Levels 1")).toBeVisible();
    await user.type(
      screen.getByLabelText("Search scenarios"),
      "trust boundary",
    );
    await waitFor(() =>
      expect(screen.queryByText("Check authorization")).not.toBeInTheDocument(),
    );
    expect(screen.getByText("Check input validation")).toBeVisible();
    expect(screen.getByText("1 of 2 scenarios")).toBeVisible();
    expect(screen.queryByText(/\d requirements?\b/)).not.toBeInTheDocument();
  });

  it("shows the exact selected requirements, reads their details and searches their text", async () => {
    const user = userEvent.setup();
    setup("/catalog/audit-presets/source-review/1");
    await screen.findByText("Check authorization");
    expect(screen.getByText("2 requirements")).toBeVisible();
    expect(
      screen.getByText("2 requirements selected · Levels 1"),
    ).toBeVisible();
    expect(
      screen.queryByText("Check transport security"),
    ).not.toBeInTheDocument();
    await user.click(screen.getByText("Check authorization"));
    expect(
      screen.getByText(
        "Verify authorization before accessing a private record.",
      ),
    ).toBeVisible();
    expect(
      screen.getByText("Trace permission checks from the request handler."),
    ).toBeVisible();
    const item = screen.getByText("Check authorization").closest("details")!;
    expect(within(item).getByText("1–5 evidence items · file")).toBeVisible();
    expect(
      within(item).getByText("Met, Issue found, Inconclusive"),
    ).toBeVisible();
    expect(screen.getByText("Full text")).toBeVisible();
    await user.type(
      screen.getByLabelText("Search requirements"),
      "trust boundary",
    );
    await waitFor(() =>
      expect(screen.queryByText("Check authorization")).not.toBeInTheDocument(),
    );
    expect(screen.getByText("Check input validation")).toBeVisible();
    expect(screen.getByText("1 of 2 requirements")).toBeVisible();
    expect(
      screen.queryByText("Check transport security"),
    ).not.toBeInTheDocument();
  });

  it("starts a check with an available type and explains an unavailable one", async () => {
    const { router } = setup("/catalog/audit-presets/source-review/10");
    const start = await screen.findByRole("link", {
      name: "Start a check with this type",
    });
    expect(start).toHaveAttribute("href", "/checks/new?type=source-review");
    expect(
      screen.getByRole("heading", { level: 2, name: "source review" }),
    ).toBeVisible();
    expect(screen.getByText("Available")).toBeVisible();
    const facts = screen.getByText("Time limit").closest("dl")!;
    expect(within(facts).getByText("10 min")).toBeVisible();
    expect(within(facts).getByText("Never run")).toBeVisible();
    expect(within(facts).getByText("You confirm each one")).toBeVisible();
    expect(within(facts).getByText("Accepted automatically")).toBeVisible();
    await act(async () => {
      await router.navigate("/catalog/audit-presets");
    });
    await waitFor(() =>
      expect(document.title).toBe("Check types · Library · Contractor"),
    );
  });

  it("switches exact versions and preserves requirement searches through workflow visits", async () => {
    const user = userEvent.setup();
    const { router } = setup(
      "/catalog/audit-presets/source-review/1?q=authorization",
    );
    await screen.findByText("Check authorization");
    await user.click(screen.getByRole("link", { name: "source-check@3" }));
    await user.click(
      await screen.findByRole("link", { name: /source review @1/ }),
    );
    expect(router.state.location.search).toBe("?q=authorization");
    await user.selectOptions(
      await screen.findByLabelText("Check type version"),
      "10",
    );
    await screen.findByText("Check transport security");
    expect(screen.getByText("3 requirements")).toBeVisible();
    expect(router.state.location.pathname).toBe(
      "/catalog/audit-presets/source-review/10",
    );
    expect(router.state.location.search).toBe("");
    await act(async () => {
      await router.navigate(-1);
    });
    expect(await screen.findByLabelText("Check type version")).toHaveValue("1");
    expect(
      screen.queryByText("Check transport security"),
    ).not.toBeInTheDocument();
  });

  it.each(["metadata", "identifiers"] as const)(
    "renders %s disclosure without pretending the check type has no requirements",
    async (disclosure) => {
      const standard: AuditStandard = {
        ...standardFixture,
        license: { ...standardFixture.license, disclosure },
        entries:
          disclosure === "identifiers"
            ? standardFixture.entries!.map(({ id, kind }) => ({ id, kind }))
            : [],
      };
      delete standard.mappings;
      delete standard.evidenceContracts;
      setup("/catalog/audit-presets/source-review/1", { standard });
      await screen.findByText(
        new RegExp(`This standard exposes ${disclosure} only`),
      );
      expect(
        screen.getByText(
          disclosure === "metadata" ? "Metadata only" : "Identifiers only",
        ),
      ).toBeVisible();
      expect(screen.queryByText("0 requirements")).not.toBeInTheDocument();
      expect(screen.queryByText("Check authorization")).not.toBeInTheDocument();
      if (disclosure === "identifiers") {
        expect(screen.getByText("REQ-1")).toBeVisible();
        expect(screen.queryByText("REQ-3")).not.toBeInTheDocument();
      }
      expect(
        screen.getByRole("link", { name: "Review standard source" }),
      ).toHaveAttribute("href", standard.source.url);
    },
  );

  it("explains dynamic endpoint lists without fetching an unrelated standard", async () => {
    const { fetcher } = setup(
      "/catalog/audit-presets/openapi-operation-trace/1",
    );
    await screen.findByText(
      "The endpoint list depends on the inputs of each check.",
    );
    expect(screen.getByRole("heading", { name: "Endpoints" })).toBeVisible();
    expect(screen.getByText(/Coverage tab/)).toBeVisible();
    expect(screen.queryByRole("searchbox")).not.toBeInTheDocument();
    expect(
      fetcher.mock.calls.some(([input]) =>
        String(input instanceof Request ? input.url : input).includes(
          "/audit-standards/",
        ),
      ),
    ).toBe(false);
  });

  it("keeps unavailable check types readable and surfaces standard loading errors", async () => {
    setup("/catalog/audit-presets/source-review/1", {
      profiles: [
        {
          ...presetFixture,
          serverCompatible: false,
          compatibilityReasons: ["multiple_rounds_unsupported"],
        },
      ],
      failStandard: true,
    });
    await screen.findByText("This check type cannot run on this server.");
    expect(
      screen.getByText(
        "This type needs more than one round, which this server does not run.",
      ),
    ).toBeVisible();
    expect(screen.getByText("multiple_rounds_unsupported")).toBeVisible();
    expect(screen.getByText("Unavailable on this server")).toBeVisible();
    expect(
      screen.queryByRole("link", { name: "Start a check with this type" }),
    ).not.toBeInTheDocument();
    await screen.findByText("Standard unavailable");
    expect(
      screen.queryByText("This check type defines no requirements."),
    ).not.toBeInTheDocument();
  });

  it("marks unavailable check types in the list with their reasons", async () => {
    setup("/catalog/audit-presets", {
      profiles: [
        {
          ...presetFixture,
          serverCompatible: false,
          compatibilityReasons: ["preparation_unsupported"],
        },
      ],
    });
    const card = (
      await screen.findByRole("link", { name: "source review" })
    ).closest("article")!;
    expect(within(card).getByText("Unavailable on this server")).toBeVisible();
    expect(
      within(card).getByText(
        "This server cannot run the preparation step this type needs yet.",
      ),
    ).toBeVisible();
    expect(
      screen.getByText("1 check type · 1 unavailable on this server"),
    ).toBeVisible();
  });

  it("rejects a standard from a different version", async () => {
    setup("/catalog/audit-presets/source-review/1", {
      standard: {
        ...standardFixture,
        reference: { ...standardFixture.reference, version: "other" },
      },
    });
    await screen.findByText(
      "Server returned an invalid audit standard response",
    );
    expect(screen.queryByText("Check authorization")).not.toBeInTheDocument();
  });

  it("rejects a truncated standard instead of showing an incomplete requirement list", async () => {
    setup("/catalog/audit-presets/source-review/1", {
      standard: {
        ...standardFixture,
        mappings: standardFixture.mappings!.slice(0, 1),
      },
    });
    await screen.findByText(
      "Server returned an invalid audit standard response",
    );
    expect(screen.queryByText("Check authorization")).not.toBeInTheDocument();
  });

  it.each([
    { options: { failCatalog: true }, message: "Preset catalog unavailable" },
    {
      options: { repeatCursor: true },
      message: "The server returned an invalid Audit pagination cursor.",
    },
  ])(
    "does not present an incomplete catalog as an empty result: $message",
    async ({ options, message }) => {
      setup("/catalog/audit-presets", options);
      await screen.findByText(message);
      expect(
        screen.queryByText("No check types published."),
      ).not.toBeInTheDocument();
    },
  );

  it("shows a useful empty state", async () => {
    setup("/catalog/audit-presets", { profiles: [] });
    await screen.findByText("No check types published.");
  });
});
