import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { PublicAPI, type AuthSession } from "../api/client";
import { Application } from "../app/application";
import { applicationRoutes } from "../app/router";
import type { RuntimeConfig } from "../config/runtime-config";

const runtimeConfig: RuntimeConfig = {
  uiVersion: "0.1.0",
  supportedApiVersions: ["contractor.public.v1"],
  apiBaseUrl: "http://127.0.0.1:8080",
};

const session: AuthSession = {
  principal: {
    userId: "user_local",
    username: "owner",
    capabilities: ["user"],
  },
  csrfToken: "a".repeat(43),
  idleExpiresAt: "2099-01-01T00:00:00Z",
  absoluteExpiresAt: "2099-01-02T00:00:00Z",
};

function apiResponse(
  value: unknown,
  status = 200,
  headers: Record<string, string> = {},
): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: {
      "content-type": "application/json",
      "X-Contractor-API-Version": "contractor.public.v1",
      ...headers,
    },
  });
}

function skill(name: string, revision: string) {
  return {
    artifact: { namespace: "skills", name, revision },
    mediaType: "application/zip",
    size: 2048,
    current: true,
    frozen: true,
    createdAt: "2026-09-01T12:00:00Z",
  };
}

/** jsdom has no IntersectionObserver; every observed row is reported visible. */
class VisibleObserver {
  constructor(private readonly callback: IntersectionObserverCallback) {}
  observe(target: Element): void {
    this.callback(
      [{ isIntersecting: true, target } as IntersectionObserverEntry],
      this as unknown as IntersectionObserver,
    );
  }
  disconnect(): void {}
  unobserve(): void {}
}

beforeEach(() => {
  vi.stubGlobal("IntersectionObserver", VisibleObserver);
});
afterEach(() => {
  vi.unstubAllGlobals();
});

describe("Skills catalog", () => {
  it("shows an explicit error with retry when the archive request fails, and keeps the placeholder for packages without a description", async () => {
    let brokenArchiveReads = 0;
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const url = new URL(request.url);
        if (url.pathname === "/v1/auth/session") {
          return apiResponse(session);
        }
        if (url.pathname === "/v1/artifacts") {
          return apiResponse({
            items: [skill("broken", "r1"), skill("plain", "r2")],
            page: { hasMore: false },
          });
        }
        if (url.pathname === "/v1/artifacts/skills/broken/archive") {
          brokenArchiveReads += 1;
          if (brokenArchiveReads === 1) {
            return apiResponse(
              { code: "unavailable", message: "Archive store is unavailable" },
              503,
            );
          }
          return apiResponse(
            {
              artifact: { namespace: "skills", name: "broken", revision: "r1" },
              entries: [
                { kind: "file", path: "SKILL.md", size: 48, previewable: true },
              ],
            },
            200,
            { ETag: '"r1"' },
          );
        }
        if (url.pathname === "/v1/artifacts/skills/broken/archive/file") {
          const text = "---\ndescription: Reviews source evidence\n---\n# Hi\n";
          return apiResponse(
            {
              artifact: { namespace: "skills", name: "broken", revision: "r1" },
              path: "SKILL.md",
              text,
              size: new TextEncoder().encode(text).length,
            },
            200,
            { ETag: '"r1"' },
          );
        }
        if (url.pathname === "/v1/artifacts/skills/plain/archive") {
          return apiResponse(
            {
              artifact: { namespace: "skills", name: "plain", revision: "r2" },
              entries: [
                { kind: "file", path: "notes.txt", size: 4, previewable: true },
              ],
            },
            200,
            { ETag: '"r2"' },
          );
        }
        throw new Error(`unexpected ${request.method} ${url.pathname}`);
      }),
    );
    const router = createMemoryRouter(applicationRoutes(), {
      initialEntries: ["/catalog/skills"],
    });
    render(<Application api={api} publicAPI={api} router={router} />);

    const broken = (
      await screen.findByRole("link", { name: "broken" })
    ).closest("tr") as HTMLElement;
    const plain = screen
      .getByRole("link", { name: "plain" })
      .closest("tr") as HTMLElement;

    // The failed archive request is an explicit, retryable state.
    expect(
      await within(broken).findByText("Description unavailable"),
    ).toBeInTheDocument();
    expect(
      within(broken).queryByText(
        "Open the package to inspect its instructions.",
      ),
    ).toBeNull();
    const retry = within(broken).getByRole("button", { name: "Retry" });

    // A package without a SKILL.md description keeps the neutral placeholder.
    expect(
      await within(plain).findByText(
        "Open the package to inspect its instructions.",
      ),
    ).toBeInTheDocument();
    expect(within(plain).queryByRole("button", { name: "Retry" })).toBeNull();

    await userEvent.setup().click(retry);
    expect(
      await within(broken).findByText("Reviews source evidence"),
    ).toBeInTheDocument();
    expect(within(broken).queryByText("Description unavailable")).toBeNull();
    expect(brokenArchiveReads).toBe(2);
  });
});
