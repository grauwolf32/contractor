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

describe("Audit presets catalog", () => {
  it("groups all pages by name, selects the highest version and finds later-page presets", async () => {
    const user = userEvent.setup();
    const { router } = setup("/catalog/audit-presets?keep=yes");
    expect(
      await screen.findByLabelText("Version of source-review"),
    ).toHaveValue("10");
    expect(screen.getAllByRole("article")).toHaveLength(2);
    expect(
      within(
        screen.getByRole("navigation", { name: "Catalog navigation" }),
      ).getByRole("link", { name: "Audit presets" }),
    ).toHaveAttribute("aria-current", "page");
    await user.selectOptions(
      screen.getByLabelText("Version of source-review"),
      "1",
    );
    expect(screen.getByRole("link", { name: "source review" })).toHaveAttribute(
      "href",
      "/catalog/audit-presets/source-review/1",
    );
    await user.type(screen.getByLabelText("Search audit presets"), "OpenAPI");
    await waitFor(() => expect(screen.getAllByRole("article")).toHaveLength(1));
    await user.click(screen.getByRole("link", { name: "View checks →" }));
    await screen.findByText("The check list depends on your audit inputs.");
    await user.click(screen.getByRole("link", { name: "← Audit presets" }));
    expect(router.state.location.search).toBe("?keep=yes&q=OpenAPI");
    expect(await screen.findByLabelText("Search audit presets")).toHaveValue(
      "OpenAPI",
    );
  });

  it("shows the exact selected requirements, reads check details and searches their text", async () => {
    const user = userEvent.setup();
    setup("/catalog/audit-presets/source-review/1");
    await screen.findByText("Check authorization");
    expect(screen.getByText("2 checks")).toBeVisible();
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
    const check = screen.getByText("Check authorization").closest("details")!;
    expect(
      within(check).getByText("1–5 evidence items · artifact"),
    ).toBeVisible();
    await user.type(screen.getByLabelText("Search checks"), "trust boundary");
    await waitFor(() =>
      expect(screen.queryByText("Check authorization")).not.toBeInTheDocument(),
    );
    expect(screen.getByText("Check input validation")).toBeVisible();
    expect(screen.getByText("1 of 2 checks")).toBeVisible();
    expect(
      screen.queryByText("Check transport security"),
    ).not.toBeInTheDocument();
  });

  it("switches exact versions and preserves check searches through workflow visits", async () => {
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
      await screen.findByLabelText("Preset version"),
      "10",
    );
    await screen.findByText("Check transport security");
    expect(screen.getByText("3 checks")).toBeVisible();
    expect(router.state.location.pathname).toBe(
      "/catalog/audit-presets/source-review/10",
    );
    expect(router.state.location.search).toBe("");
    await act(async () => {
      await router.navigate(-1);
    });
    expect(await screen.findByLabelText("Preset version")).toHaveValue("1");
    expect(
      screen.queryByText("Check transport security"),
    ).not.toBeInTheDocument();
  });

  it.each(["metadata", "identifiers"] as const)(
    "renders %s disclosure without pretending the preset has no checks",
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
        new RegExp(
          `This standard exposes ${disclosure === "metadata" ? "metadata" : "requirement identifiers"} only`,
        ),
      );
      expect(screen.queryByText("0 checks")).not.toBeInTheDocument();
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

  it("explains dynamic check lists without fetching an unrelated standard", async () => {
    const { fetcher } = setup(
      "/catalog/audit-presets/openapi-operation-trace/1",
    );
    await screen.findByText("The check list depends on your audit inputs.");
    expect(screen.getByText(/Coverage tab/)).toBeVisible();
    expect(screen.queryByLabelText("Search checks")).not.toBeInTheDocument();
    expect(
      fetcher.mock.calls.some(([input]) =>
        String(input instanceof Request ? input.url : input).includes(
          "/audit-standards/",
        ),
      ),
    ).toBe(false);
  });

  it("keeps unavailable presets readable and surfaces standard loading errors", async () => {
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
    await screen.findByText("This preset cannot run on this server.");
    await screen.findByText("Standard unavailable");
    expect(
      screen.queryByText("No checks are defined for this preset."),
    ).not.toBeInTheDocument();
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

  it("rejects a truncated standard instead of showing an incomplete check list", async () => {
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
        screen.queryByText("No audit presets published."),
      ).not.toBeInTheDocument();
    },
  );

  it("shows a useful empty catalog state", async () => {
    setup("/catalog/audit-presets", { profiles: [] });
    await screen.findByText("No audit presets published.");
  });
});
