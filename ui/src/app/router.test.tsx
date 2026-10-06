import { act, render, screen, waitFor } from "@testing-library/react";
import { useEffect } from "react";
import {
  createMemoryRouter,
  matchRoutes,
  useParams,
  type RouteObject,
} from "react-router";
import { describe, expect, it, vi } from "vitest";

import { PublicAPI, type AuthSession } from "../api/client";
import type { SessionAPI } from "../auth/session";
import type { RuntimeConfig } from "../config/runtime-config";
import { ChecksRoute } from "../routes/checks";
import { StartCheckRoute } from "../routes/checks/start";
import { InboxRoute } from "../routes/inbox";
import { IssuesRoute } from "../routes/issues";
import { ReportsRoute } from "../routes/reports";
import { Application } from "./application";
import { applicationRoutes, lazyRoute } from "./router";

const session: AuthSession = {
  principal: {
    userId: "user_local",
    username: "owner",
    capabilities: ["user", "operations"],
  },
  csrfToken: "a".repeat(43),
  idleExpiresAt: "2026-10-06T20:00:00Z",
  absoluteExpiresAt: "2026-10-07T12:00:00Z",
};

const runtimeConfig: RuntimeConfig = {
  uiVersion: "0.1.0",
  supportedApiVersions: ["contractor.public.v1"],
  apiBaseUrl: "http://127.0.0.1:8080",
};

const sessionAPI: SessionAPI = {
  getSession: vi.fn(async () => session),
  login: vi.fn(async () => session),
  logout: vi.fn(async () => undefined),
};

function emptyPage(): Response {
  return new Response(JSON.stringify({ items: [], page: { hasMore: false } }), {
    headers: {
      "content-type": "application/json",
      "X-Contractor-API-Version": "contractor.public.v1",
    },
  });
}

function renderAt(path: string, routes = applicationRoutes()) {
  const publicAPI = new PublicAPI(
    runtimeConfig,
    vi.fn(async () => emptyPage()),
  );
  const router = createMemoryRouter(routes, { initialEntries: [path] });
  render(
    <Application api={sessionAPI} publicAPI={publicAPI} router={router} />,
  );
  return router;
}

function shellRoutes(routes: RouteObject[]): RouteObject[] {
  const children = routes[1]?.children?.[0]?.children;
  if (children === undefined) throw new Error("Shell routes are missing");
  return children;
}

function leafRoute(path: string, routes = applicationRoutes()) {
  return matchRoutes(routes, path)?.at(-1)?.route;
}

/** The component a URL renders inside the shell, loaded like the router does. */
async function routeComponent(path: string) {
  const lazy = leafRoute(path)?.lazy;
  if (typeof lazy !== "function") throw new Error(`${path} is not lazy`);
  return (await lazy()).Component;
}

const listMounts = vi.fn();

/** Stands in for a list + detail destination; counts its mounts. */
function ListProbe() {
  const params = useParams();
  useEffect(() => {
    listMounts();
  }, []);
  return <h1>{Object.values(params).join("/") || "Nothing selected"}</h1>;
}

describe("V3B destinations", () => {
  it.each([
    ["/", InboxRoute],
    ["/checks", ChecksRoute],
    [
      "/checks?state=waiting_review&project=project_example&check=audit_example",
      ChecksRoute,
    ],
    ["/checks/new", StartCheckRoute],
    [
      "/checks/new?project=project_example&objective=Find%20IDOR&type=owasp-top10",
      StartCheckRoute,
    ],
    ["/issues", IssuesRoute],
    ["/issues?state=proposed&project=project_example", IssuesRoute],
    ["/issues/audit_example/finding_example", IssuesRoute],
    ["/reports", ReportsRoute],
    ["/reports/audit_example", ReportsRoute],
  ])("routes %s to its page", async (path, component) => {
    expect(await routeComponent(path)).toBe(component);
  });

  it.each([
    "/inbox",
    "/checks/audit_example",
    "/checks/new/extra",
    "/issues/audit_example",
    "/issues/a/b/c",
    "/reports/audit_example/extra",
  ])("leaves %s to Page not found", (path) => {
    expect(leafRoute(path)?.path).toBe("*");
  });

  it("keeps /checks/new ahead of a check parameter route", () => {
    const routes = applicationRoutes();
    shellRoutes(routes).unshift({ path: "/checks/:auditId", element: null });
    expect(leafRoute("/checks/new", routes)?.path).toBe("/checks/new");
    expect(leafRoute("/checks/audit_example", routes)?.path).toBe(
      "/checks/:auditId",
    );
  });

  it.each([
    ["/", "Inbox"],
    ["/checks", "Checks"],
    ["/checks/new", "Start a check"],
    ["/issues/audit_example/finding_example", "Issues"],
    ["/reports/audit_example", "Reports"],
  ])("opens %s directly inside the shell", async (path, title) => {
    renderAt(path);
    expect(
      await screen.findByRole("heading", { level: 1, name: title }),
    ).toBeVisible();
    expect(screen.getByText("This page is being built.")).toBeVisible();
    expect(
      screen.getByRole("navigation", { name: "Primary navigation" }),
    ).toBeVisible();
    await waitFor(() => expect(document.title).toBe(`${title} · Contractor`));
  });

  it.each([
    ["/issues", "/issues/audit_example/finding_example"],
    ["/reports", "/reports/audit_example"],
  ])("keeps %s mounted while the selected item changes", async (list, item) => {
    listMounts.mockClear();
    const routes = applicationRoutes();
    for (const route of shellRoutes(routes)) {
      if (route.path === list || route.path?.startsWith(`${list}/`)) {
        route.lazy = lazyRoute(async () => ({ ListProbe }), "ListProbe");
      }
    }
    const router = renderAt(list, routes);
    expect(
      await screen.findByRole("heading", { name: "Nothing selected" }),
    ).toBeVisible();
    await act(async () => {
      await router.navigate(item);
    });
    const selected = item.slice(list.length + 1);
    expect(
      await screen.findByRole("heading", { name: selected }),
    ).toBeVisible();
    await act(async () => {
      await router.navigate(list);
    });
    expect(
      await screen.findByRole("heading", { name: "Nothing selected" }),
    ).toBeVisible();
    expect(listMounts).toHaveBeenCalledTimes(1);
  });
});
