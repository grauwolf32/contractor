import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { PublicAPI, type AuthSession } from "../api/client";
import { queryKeys } from "../api/query-keys";
import { SessionProvider, useSession } from "./session";

const session: AuthSession = {
  principal: {
    userId: "user_local",
    username: "owner",
    capabilities: ["user", "operations"],
  },
  csrfToken: "a".repeat(43),
  idleExpiresAt: "2026-09-20T12:00:00Z",
  absoluteExpiresAt: "2026-09-21T12:00:00Z",
};

function response(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: {
      "Content-Type": "application/json",
      "X-Contractor-API-Version": "contractor.public.v1",
    },
  });
}

function SessionControls() {
  const current = useSession();
  return (
    <>
      <p>{current.session?.csrfToken ?? "Signed out"}</p>
      <button
        type="button"
        onClick={() =>
          void current.login({ username: "owner", password: "password" })
        }
      >
        Sign in again
      </button>
    </>
  );
}

describe("session query cancellation", () => {
  it.each([200, 401])(
    "keeps the new same-owner login when a cancelled session lookup returns %s",
    async (status) => {
      let lookups = 0;
      let resolveLookup!: (response: Response) => void;
      const pendingLookup = new Promise<Response>((resolve) => {
        resolveLookup = resolve;
      });
      const freshSession = { ...session, csrfToken: "b".repeat(43) };
      const api = new PublicAPI(
        {
          uiVersion: "0.1.0",
          supportedApiVersions: ["contractor.public.v1"],
          apiBaseUrl: "http://127.0.0.1:8080",
        },
        vi.fn(async (input) => {
          const url = new URL((input as Request).url);
          if (url.pathname === "/v1/auth/session") {
            lookups += 1;
            return lookups === 1 ? response(session) : pendingLookup;
          }
          if (url.pathname === "/v1/auth/login") return response(freshSession);
          return response({ code: "unauthorized" }, 401);
        }),
      );
      const queryClient = new QueryClient({
        defaultOptions: { queries: { retry: false } },
      });
      render(
        <QueryClientProvider client={queryClient}>
          <SessionProvider api={api} publicAPI={api}>
            <SessionControls />
          </SessionProvider>
        </QueryClientProvider>,
      );
      await screen.findByText(session.csrfToken);

      // A reconnect can revalidate the old cookie while another domain
      // request expires the UI session and cancels that query.
      void queryClient.invalidateQueries({ queryKey: queryKeys.session });
      await waitFor(() => expect(lookups).toBe(2));
      await act(async () => {
        await api.fetch("/v1/projects");
      });
      await screen.findByText("Signed out");
      await userEvent
        .setup()
        .click(screen.getByRole("button", { name: "Sign in again" }));
      await screen.findByText(freshSession.csrfToken);

      await act(async () => {
        resolveLookup(
          response(status === 200 ? session : { code: "unauthorized" }, status),
        );
        await pendingLookup;
      });
      expect(queryClient.getQueryData(queryKeys.session)).toEqual(freshSession);
      expect(api.mutationHeaders().get("X-CSRF-Token")).toBe(
        freshSession.csrfToken,
      );
    },
  );
});
