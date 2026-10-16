import {
  act,
  fireEvent,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";

import { gitKeyQueryKey } from "../../api/git-artifacts";
import { conforming } from "../../test/builders";
import { fakeServer, type Routes } from "../../test/fake-server";
import { failure, json, noContent } from "../../test/responses";
import { renderWithAPI } from "../../test/render";
import { GitKeySettings } from "../settings/git-key";
import { GitImportDialog } from "./git-import-dialog";

function setup(child: ReactNode, routes: Routes) {
  const view = renderWithAPI(<>{child}</>, { server: fakeServer(routes) });
  return { ...view, cache: view.queryClient };
}

/** A configured key with `fingerprint`, checked against the contract. */
function gitKey(fingerprint: string) {
  const state = { configured: true, fingerprint, keyType: "ssh-ed25519" };
  conforming("GitKeyState", state);
  return state;
}

const imported = conforming("GitImportResult", {
  artifact: { namespace: "sources", name: "source", revision: "rev-imported" },
  mediaType: "application/zip",
  size: 123,
  gitSource: {
    repositoryUrl: "https://example.test:443/repo.git",
    requestedRef: "main",
    resolvedCommit: "a".repeat(40),
    importedAt: "2026-09-06T12:00:00Z",
  },
});

describe("Git artifacts", () => {
  it.each(["save", "replace", "remove"] as const)(
    "keeps the successful %s result when an older key GET arrives late",
    async (change) => {
      const previous = gitKey("SHA256:previous");
      const saved = { ...previous, fingerprint: "SHA256:saved" };
      let resolveRead!: (response: Response) => void;
      const delayedRead = new Promise<Response>((resolve) => {
        resolveRead = resolve;
      });
      let reads = 0;
      let delayedRequest: Request | undefined;
      const { cache } = setup(<GitKeySettings />, {
        "GET /v1/settings/git-key": ({ request }) => {
          reads++;
          if (change !== "save" && reads === 1) return json(previous);
          delayedRequest = request;
          // Deliver the stale response even if transport cancellation races
          // with completion; query cancellation must also protect the cache.
          return delayedRead;
        },
        "PUT /v1/settings/git-key": () => json(saved),
        "DELETE /v1/settings/git-key": () => noContent(),
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
      if (change === "remove") {
        const dialog = screen.getByRole("alertdialog", {
          name: "Remove Git key?",
        });
        expect(dialog).toHaveTextContent(previous.fingerprint);
        await userEvent.click(
          within(dialog).getByRole("button", { name: "Remove Git key" }),
        );
      }
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
    const { cache } = setup(<GitKeySettings />, {
      "GET /v1/settings/git-key": ({ request }) => {
        reads++;
        if (reads === 1) {
          initialRequest = request;
          return initialRead;
        }
        return json({ configured: false });
      },
      "PUT /v1/settings/git-key": () => {
        writes++;
        return failure(400, "git_key_invalid", "Unsupported key");
      },
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
    const { cache } = setup(<GitKeySettings />, {
      "GET /v1/settings/git-key": () => json({ configured: false }),
      "DELETE /v1/settings/git-key": () => noContent(),
      "PUT /v1/settings/git-key": async ({ request }) => {
        writes.push(await request.text());
        return json({
          ...gitKey("SHA256:fixture"),
          updatedAt: imported.gitSource.importedAt,
        });
      },
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
    await userEvent.click(
      within(
        screen.getByRole("alertdialog", { name: "Remove Git key?" }),
      ).getByRole("button", { name: "Remove Git key" }),
    );
    await screen.findByText("Git SSH key removed.");
    expect(screen.getByText("No Git SSH key configured.")).toBeInTheDocument();
  });
  it("requires confirmation for Git key removal and dismisses safely", async () => {
    const { user, sent } = setup(<GitKeySettings />, {
      "GET /v1/settings/git-key": () => json(gitKey("SHA256:keep-me")),
      "DELETE /v1/settings/git-key": () => noContent(),
    });
    const deletes = () => sent("DELETE", "/git-key");
    await screen.findByText("SHA256:keep-me");
    const trigger = await screen.findByRole("button", {
      name: "Remove Git key",
    });
    const open = async () => {
      await user.click(trigger);
      const dialog = screen.getByRole("alertdialog", {
        name: "Remove Git key?",
      });
      expect(dialog).toHaveTextContent("SHA256:keep-me");
      expect(
        within(dialog).getByRole("button", { name: "Cancel" }),
      ).toHaveFocus();
      expect(deletes()).toHaveLength(0);
      return dialog;
    };
    let dialog = await open();
    await user.click(within(dialog).getByRole("button", { name: "Cancel" }));
    await open();
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("alertdialog")).toBeNull();
    dialog = await open();
    await user.click(dialog.parentElement!);
    expect(screen.queryByRole("alertdialog")).toBeNull();
    expect(deletes()).toHaveLength(0);
    expect(screen.getByText("SHA256:keep-me")).toBeInTheDocument();
    dialog = await open();
    await user.click(
      within(dialog).getByRole("button", { name: "Remove Git key" }),
    );
    await waitFor(() => expect(deletes()).toHaveLength(1));
    await screen.findByText("Git SSH key removed.");
    expect(screen.queryByRole("alertdialog")).toBeNull();
  });
  it("keeps a pending Git key removal dialog open and shows a failed delete", async () => {
    let finishDelete!: (response: Response) => void;
    const delayedDelete = new Promise<Response>((resolve) => {
      finishDelete = resolve;
    });
    let deletes = 0;
    const { user } = setup(<GitKeySettings />, {
      "GET /v1/settings/git-key": () => json(gitKey("SHA256:still-here")),
      "DELETE /v1/settings/git-key": () => {
        deletes++;
        return delayedDelete;
      },
    });
    await screen.findByText("SHA256:still-here");
    await user.click(screen.getByRole("button", { name: "Remove Git key" }));
    const dialog = screen.getByRole("alertdialog", { name: "Remove Git key?" });
    await user.click(
      within(dialog).getByRole("button", { name: "Remove Git key" }),
    );
    await waitFor(() => expect(deletes).toBe(1));
    expect(
      within(dialog).getByRole("button", { name: "Cancel" }),
    ).toBeDisabled();
    await user.keyboard("{Escape}");
    await user.click(dialog.parentElement!);
    expect(dialog).toBeInTheDocument();
    await act(async () => {
      finishDelete(failure(500, "git_key_delete_failed", "Removal failed"));
      await delayedDelete;
    });
    expect(await within(dialog).findByRole("alert")).toHaveTextContent(
      "Removal failed",
    );
    await user.click(within(dialog).getByRole("button", { name: "Cancel" }));
    expect(screen.getByText("SHA256:still-here")).toBeInTheDocument();
    expect(deletes).toBe(1);
  });
  it("keeps the existing key state on invalid replacement", async () => {
    setup(<GitKeySettings />, {
      "GET /v1/settings/git-key": () => json(gitKey("SHA256:previous")),
      "PUT /v1/settings/git-key": () =>
        failure(400, "git_key_invalid", "Unsupported key"),
    });
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
      outerSubmit = vi.fn();
    const { requests } = setup(
      <form onSubmit={outerSubmit}>
        <GitImportDialog
          projectId="project-1"
          onClose={() => {}}
          onImported={onImported}
        />
      </form>,
      {
        "GET /v1/projects/{projectId}/artifacts/{namespace}/{name}/metadata":
          () =>
            json({
              ...imported,
              artifact: { ...imported.artifact, revision: "rev-before" },
              current: true,
              frozen: false,
              createdAt: imported.gitSource.importedAt,
            }),
        "POST /v1/projects/{projectId}/artifacts/{namespace}/{name}/git-import":
          ({ request }) => {
            expect(request.headers.get("If-Match")).toBe('"rev-before"');
            expect(request.headers.get("If-None-Match")).toBeNull();
            return json(imported, 200);
          },
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
    await userEvent.click(screen.getByLabelText("Replace this revision"));
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
    const { requests } = setup(
      <GitImportDialog onClose={onClose} onImported={vi.fn()} />,
      {
        "GET /v1/artifacts/{namespace}/{name}/metadata": () => failure(404),
        "POST /v1/artifacts/{namespace}/{name}/git-import": ({ request }) => {
          signal = request.signal;
          return new Promise<Response>((_resolve, reject) =>
            request.signal.addEventListener("abort", () =>
              reject(new Error("aborted")),
            ),
          );
        },
      },
    );
    fireEvent.change(screen.getByLabelText("Repository URL"), {
      target: { value: "https://example.test/repo.git" },
    });
    fireEvent.change(screen.getByLabelText("Name"), {
      target: { value: "has spaces" },
    });
    await userEvent.click(
      screen.getByRole("button", { name: "Import snapshot" }),
    );
    await screen.findByRole("alert");
    expect(requests).toHaveLength(0);
    fireEvent.change(screen.getByLabelText("Name"), {
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
