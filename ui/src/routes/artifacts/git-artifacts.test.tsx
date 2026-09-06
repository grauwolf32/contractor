import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router";
import { describe, expect, it, vi } from "vitest";
import { PublicAPI } from "../../api/client";
import { PublicAPIProvider } from "../../api/context";
import { gitKeyQueryKey } from "../../api/git-artifacts";
import { GitKeySettings } from "../settings/git-key";
import { GitImportDialog } from "./git-import-dialog";
import type { ReactNode } from "react";

function json(value: unknown, status = 200) {
  return new Response(status === 204 ? null : JSON.stringify(value), {
    status,
    headers: {
      "content-type": "application/json",
      "X-Contractor-API-Version": "contractor.public.v1",
    },
  });
}
function setup(
  child: ReactNode,
  fetch: (request: Request) => Promise<Response>,
) {
  const api = new PublicAPI(
    {
      uiVersion: "0.1.0",
      supportedApiVersions: ["contractor.public.v1"],
      apiBaseUrl: "http://127.0.0.1:8080",
    },
    vi.fn(async (input) =>
      fetch(input instanceof Request ? input : new Request(input)),
    ),
  );
  api.csrf.replace("a".repeat(43));
  const cache = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return {
    cache,
    ...render(
      <QueryClientProvider client={cache}>
        <PublicAPIProvider api={api}>
          <MemoryRouter>{child}</MemoryRouter>
        </PublicAPIProvider>
      </QueryClientProvider>,
    ),
  };
}
const imported = {
  artifact: { namespace: "sources", name: "source", revision: "rev-imported" },
  mediaType: "application/zip" as const,
  size: 123,
  gitSource: {
    repositoryUrl: "https://example.test:443/repo.git",
    requestedRef: "main",
    resolvedCommit: "a".repeat(40),
    importedAt: "2026-09-06T12:00:00Z",
  },
};

describe("Git artifacts", () => {
  it.each(["save", "replace", "remove"] as const)(
    "keeps the successful %s result when an older key GET arrives late",
    async (change) => {
      const previous = {
        configured: true,
        fingerprint: "SHA256:previous",
        keyType: "ssh-ed25519",
      };
      const saved = { ...previous, fingerprint: "SHA256:saved" };
      let resolveRead!: (response: Response) => void;
      const delayedRead = new Promise<Response>((resolve) => {
        resolveRead = resolve;
      });
      let reads = 0;
      let delayedRequest: Request | undefined;
      const { cache } = setup(<GitKeySettings />, async (request) => {
        if (request.method === "GET") {
          reads++;
          if (change !== "save" && reads === 1) return json(previous);
          delayedRequest = request;
          // Deliver the stale response even if transport cancellation races
          // with completion; query cancellation must also protect the cache.
          return delayedRead;
        }
        return request.method === "DELETE" ? json(undefined, 204) : json(saved);
      });
      if (change !== "save") {
        await screen.findByText(previous.fingerprint);
        void cache.invalidateQueries({ queryKey: gitKeyQueryKey });
      }
      await waitFor(() => expect(delayedRequest).toBeDefined());
      if (change !== "remove") {
        fireEvent.change(screen.getByLabelText("SSH private key"), {
          target: { value: "fixture-private-key-canary" },
        });
      }
      await userEvent.click(
        screen.getByRole("button", {
          name:
            change === "save"
              ? "Save Git key"
              : change === "replace"
                ? "Replace Git key"
                : "Remove Git key",
        }),
      );
      await screen.findByText(
        change === "remove" ? "Git SSH key removed." : "Git SSH key saved.",
      );
      await act(async () => {
        resolveRead(json(change === "save" ? { configured: false } : previous));
        await delayedRead;
      });
      await waitFor(() => expect(cache.isFetching()).toBe(0));
      expect(cache.getQueryData(gitKeyQueryKey)).toEqual(
        change === "remove" ? { configured: false } : saved,
      );
      expect(delayedRequest?.signal.aborted).toBe(true);
      const remove = screen.getByRole("button", { name: "Remove Git key" });
      if (change === "remove") {
        expect(remove).toBeDisabled();
        expect(
          screen.getByText("No Git SSH key configured."),
        ).toBeInTheDocument();
      } else {
        expect(remove).toBeEnabled();
        expect(screen.getByText(saved.fingerprint)).toBeInTheDocument();
      }
    },
  );
  it("reloads key metadata after cancelling the initial GET and failing to save", async () => {
    let reads = 0;
    let writes = 0;
    let initialRequest: Request | undefined;
    let resolveInitial!: (response: Response) => void;
    const initialRead = new Promise<Response>((resolve) => {
      resolveInitial = resolve;
    });
    const { cache } = setup(<GitKeySettings />, async (request) => {
      if (request.method === "GET") {
        reads++;
        if (reads === 1) {
          initialRequest = request;
          return initialRead;
        }
        return json({ configured: false });
      }
      writes++;
      return json(
        {
          code: "git_key_invalid",
          message: "Unsupported key",
          retryable: false,
        },
        400,
      );
    });
    await waitFor(() => expect(initialRequest).toBeDefined());
    fireEvent.change(screen.getByLabelText("SSH private key"), {
      target: { value: "invalid" },
    });
    await userEvent.click(screen.getByRole("button", { name: "Save Git key" }));
    await screen.findByRole("alert");
    await screen.findByText("No Git SSH key configured.");
    expect(initialRequest?.signal.aborted).toBe(true);
    expect(reads).toBe(2);
    expect(writes).toBe(1);
    expect(
      screen.queryByText("Loading Git key settings…"),
    ).not.toBeInTheDocument();
    await act(async () => {
      resolveInitial(json({ configured: true, fingerprint: "SHA256:stale" }));
      await initialRead;
    });
    expect(cache.getQueryData(gitKeyQueryKey)).toEqual({ configured: false });
  });
  it("saves and removes only key metadata, clears the secret and never caches it", async () => {
    const secret = "fixture-private-key-canary";
    const writes: string[] = [];
    const { cache } = setup(<GitKeySettings />, async (request) => {
      if (request.method === "GET") return json({ configured: false });
      if (request.method === "DELETE") return json(undefined, 204);
      writes.push(await request.text());
      return json({
        configured: true,
        fingerprint: "SHA256:fixture",
        keyType: "ssh-ed25519",
        updatedAt: imported.gitSource.importedAt,
      });
    });
    await screen.findByText("No Git SSH key configured.");
    fireEvent.change(screen.getByLabelText("SSH private key"), {
      target: { value: secret },
    });
    await userEvent.click(screen.getByRole("button", { name: "Save Git key" }));
    await screen.findByText("Git SSH key saved.");
    expect(screen.getByLabelText("SSH private key")).toHaveValue("");
    expect(writes).toEqual([JSON.stringify({ privateKey: secret })]);
    expect(
      JSON.stringify(
        cache
          .getQueryCache()
          .getAll()
          .map((query) => query.state.data),
      ),
    ).not.toContain(secret);
    expect(cache.getMutationCache().getAll()).toHaveLength(0);
    await userEvent.click(
      screen.getByRole("button", { name: "Remove Git key" }),
    );
    await screen.findByText("Git SSH key removed.");
    expect(screen.getByText("No Git SSH key configured.")).toBeInTheDocument();
  });
  it("keeps the existing key state on invalid replacement", async () => {
    setup(<GitKeySettings />, async (request) =>
      request.method === "GET"
        ? json({
            configured: true,
            fingerprint: "SHA256:previous",
            keyType: "ssh-ed25519",
          })
        : json(
            {
              code: "git_key_invalid",
              message: "Unsupported key",
              retryable: false,
            },
            400,
          ),
    );
    await screen.findByText("SHA256:previous");
    fireEvent.change(screen.getByLabelText("SSH private key"), {
      target: { value: "invalid" },
    });
    await userEvent.click(
      screen.getByRole("button", { name: "Replace Git key" }),
    );
    await screen.findByRole("alert");
    expect(screen.getByText("SHA256:previous")).toBeInTheDocument();
  });
  it("requires exact replacement consent and never bubbles submission into the Run form", async () => {
    const onImported = vi.fn(),
      outerSubmit = vi.fn(),
      requests: Request[] = [];
    setup(
      <form onSubmit={outerSubmit}>
        <GitImportDialog
          projectId="project-1"
          onClose={() => {}}
          onImported={onImported}
        />
      </form>,
      async (request) => {
        requests.push(request);
        if (request.method === "GET")
          return json({
            ...imported,
            artifact: { ...imported.artifact, revision: "rev-before" },
            current: true,
            frozen: false,
            createdAt: imported.gitSource.importedAt,
          });
        expect(request.headers.get("If-Match")).toBe('"rev-before"');
        expect(request.headers.get("If-None-Match")).toBeNull();
        return json(imported, 200);
      },
    );
    expect(screen.getByLabelText("Repository URL")).toHaveFocus();
    fireEvent.change(screen.getByLabelText("Repository URL"), {
      target: { value: "https://example.test/repo.git" },
    });
    await userEvent.click(
      screen.getByRole("button", { name: "Import snapshot" }),
    );
    await screen.findByText("rev-before");
    expect(
      requests.filter((request) => request.method === "POST"),
    ).toHaveLength(0);
    await userEvent.click(screen.getByLabelText("Replace this exact revision"));
    await userEvent.click(
      screen.getByRole("button", { name: "Import snapshot" }),
    );
    await waitFor(() => expect(onImported).toHaveBeenCalledWith(imported));
    expect(outerSubmit).not.toHaveBeenCalled();
    expect(requests.at(-1)?.url).toContain(
      "/v1/projects/project-1/artifacts/sources/source/git-import",
    );
  });
  it("rejects spaces before a request and aborts a pending import on close", async () => {
    let signal: AbortSignal | undefined;
    const onClose = vi.fn();
    const fetch = vi.fn(async (request: Request) => {
      if (request.method === "GET")
        return json(
          { code: "not_found", message: "not found", retryable: false },
          404,
        );
      signal = request.signal;
      return new Promise<Response>((_resolve, reject) =>
        request.signal.addEventListener("abort", () =>
          reject(new Error("aborted")),
        ),
      );
    });
    setup(<GitImportDialog onClose={onClose} onImported={vi.fn()} />, fetch);
    fireEvent.change(screen.getByLabelText("Repository URL"), {
      target: { value: "https://example.test/repo.git" },
    });
    fireEvent.change(screen.getByLabelText("Artifact name"), {
      target: { value: "has spaces" },
    });
    await userEvent.click(
      screen.getByRole("button", { name: "Import snapshot" }),
    );
    await screen.findByRole("alert");
    expect(fetch).not.toHaveBeenCalled();
    fireEvent.change(screen.getByLabelText("Artifact name"), {
      target: { value: "source" },
    });
    await userEvent.click(
      screen.getByRole("button", { name: "Import snapshot" }),
    );
    await waitFor(() => expect(signal).toBeDefined());
    await userEvent.click(
      screen.getByRole("button", { name: "Cancel import" }),
    );
    expect(signal?.aborted).toBe(true);
    expect(onClose).toHaveBeenCalledOnce();
  });
});
