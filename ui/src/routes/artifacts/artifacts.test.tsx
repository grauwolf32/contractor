import {
  act,
  fireEvent,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import type { ArtifactMetadata } from "../../api/artifacts";
import { conforming } from "../../test/builders";
import { failure, json, page } from "../../test/fake-server";
import { CSRF_TOKEN } from "../../test/render";
import {
  artifactMetadata,
  artifactWrite,
  payload,
  renderApplication,
} from "./test-support";

const LIST = "GET /v1/artifacts";
const METADATA = "GET /v1/artifacts/{namespace}/{name}/metadata";
const VERSIONS = "GET /v1/artifacts/{namespace}/{name}/versions";
const LINEAGE = "GET /v1/artifacts/{namespace}/{name}/lineage";
const WRITE = "PUT /v1/artifacts/{namespace}/{name}";

/** `projects/source` at `revision`, a 4-byte text file. */
function source(
  revision: string,
  overrides: Partial<ArtifactMetadata> = {},
): ArtifactMetadata {
  return artifactMetadata("projects", "source", {
    artifact: { namespace: "projects", name: "source", revision },
    mediaType: "text/plain",
    size: 4,
    createdAt: "2026-09-01T10:00:00Z",
    ...overrides,
  });
}

/** A PUT that stays pending until the UI aborts it. */
function pendingUntilAborted(request: Request): Promise<Response> {
  return new Promise<Response>((_resolve, reject) =>
    request.signal.addEventListener("abort", () =>
      reject(new DOMException("aborted", "AbortError")),
    ),
  );
}

describe("Artifact routes", () => {
  it("shows Files inside the Library with its tabs, without Skill packages, paged in the URL", async () => {
    const file = artifactMetadata("inputs", "wordlist", {
      mediaType: "text/vnd.contractor.wordlist",
      size: 2048,
      createdAt: "2026-09-01T10:00:00Z",
    });
    const { router, sent } = renderApplication("/artifacts", {
      [LIST]: ({ url }) =>
        url.searchParams.get("cursor") === "page-2"
          ? json(
              page([
                { ...file, artifact: { ...file.artifact, name: "second" } },
              ]),
            )
          : json(page([file], "page-2")),
    });

    expect(
      await screen.findByRole("heading", { level: 1, name: "Library" }),
    ).toBeVisible();
    const tabs = screen.getByRole("navigation", { name: "Library sections" });
    expect(within(tabs).getByRole("link", { name: "Files" })).toHaveAttribute(
      "aria-current",
      "page",
    );
    expect(
      within(tabs)
        .getAllByRole("link")
        .map((link) => link.textContent),
    ).toEqual(["Check types", "Workflows", "Agents", "Skills", "Files"]);
    expect(
      screen.getByRole("heading", { level: 2, name: "Files" }),
    ).toBeVisible();
    await vi.waitFor(() =>
      expect(document.title).toBe("Files · Library · Contractor"),
    );
    const files = screen.getByRole("region", { name: "Files" });
    expect(within(files).getByRole("link", { name: "Skills" })).toHaveAttribute(
      "href",
      "/catalog/skills",
    );
    const row = await within(files).findByRole("link", {
      name: "inputs/wordlist",
    });
    expect(row).toHaveAttribute("href", "/artifacts/inputs/wordlist");
    expect(row.closest("li")).toHaveTextContent("Wordlist");
    expect(row.closest("li")).toHaveTextContent("2.0 KiB");
    expect(
      sent("GET", "/v1/artifacts").every(
        (request) =>
          new URL(request.url).searchParams.get("excludeNamespace") ===
          "skills",
      ),
    ).toBe(true);

    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Next" }));
    expect(
      await screen.findByRole("link", { name: "inputs/second" }),
    ).toBeVisible();
    expect(router.state.location.search).toBe("?cursor=page-2");
    await user.click(screen.getByRole("button", { name: "Previous" }));
    expect(
      await screen.findByRole("link", { name: "inputs/wordlist" }),
    ).toBeVisible();
    expect(router.state.location.search).toBe("");
  });

  it("keeps the revision lede neutral while metadata is loading or unavailable", async () => {
    let respond!: (response: Response) => void;
    const metadata = new Promise<Response>((resolve) => {
      respond = resolve;
    });
    renderApplication("/artifacts/projects/source", {
      [METADATA]: () => metadata,
    });
    expect(await screen.findByText("Loading this file…")).toBeVisible();
    expect(
      screen.getByRole("heading", { level: 1, name: "projects/source" }),
    ).toBeVisible();
    expect(screen.queryByText(/(Current|Historical) revision/)).toBeNull();
    await act(async () => {
      respond(failure(503, "unavailable", "Metadata unavailable"));
      await metadata;
    });
    expect(await screen.findByText("Could not load this file")).toBeVisible();
    expect(screen.queryByText(/(Current|Historical) revision/)).toBeNull();
    expect(
      screen.queryByRole("button", { name: "Download this revision" }),
    ).toBeNull();
  });

  it("recovers a failed filtered read without submitting a write or dropping filters", async () => {
    let reads = 0;
    const { requests } = renderApplication("/artifacts?namespace=projects", {
      [LIST]: ({ url }) => {
        expect(url.searchParams.get("namespace")).toBe("projects");
        reads += 1;
        return reads === 1
          ? failure(503, "unavailable", "Storage is temporarily unavailable")
          : json(page([]));
      },
    });
    expect(await screen.findByText("Could not load files")).toBeVisible();
    expect(
      screen.getByText("Storage is temporarily unavailable"),
    ).toBeVisible();
    await userEvent
      .setup()
      .click(screen.getByRole("button", { name: "Try again" }));
    expect(
      await screen.findByText("Nothing in the projects namespace"),
    ).toBeVisible();
    expect(
      within(
        screen.getByRole("button", { name: "Apply" }).closest("form")!,
      ).getByLabelText("Namespace"),
    ).toHaveValue("projects");
    expect(
      screen.queryByRole("navigation", { name: "File pages" }),
    ).not.toBeInTheDocument();
    expect(reads).toBe(2);
    expect(requests.every((request) => request.method === "GET")).toBe(true);
  });

  it("lists bindings and creates one exact Artifact without optimistic state", async () => {
    const current = artifactMetadata("projects", "existing", {
      artifact: {
        namespace: "projects",
        name: "existing",
        revision: "revision-1",
      },
      mediaType: "text/plain",
      size: 5,
      createdAt: "2026-08-31T12:00:00Z",
    });
    const stored = artifactMetadata("projects", "source", {
      artifact: {
        namespace: "projects",
        name: "source",
        revision: "revision-2",
      },
      size: 3,
    });
    const { sent } = renderApplication("/artifacts", {
      [LIST]: ({ url }) => {
        expect(url.searchParams.get("excludeNamespace")).toBe("skills");
        return url.searchParams.get("cursor") === "cursor-next"
          ? json(
              page([
                {
                  ...current,
                  artifact: {
                    ...current.artifact,
                    name: "next-page",
                    revision: "revision-next",
                  },
                },
              ]),
            )
          : json(page([current], "cursor-next"));
      },
      [WRITE]: () => json(artifactWrite(stored), 201, { ETag: '"revision-2"' }),
    });
    // Library rows open the current binding, not the listed revision.
    expect(
      await screen.findByRole("link", { name: "projects/existing" }),
    ).toHaveAttribute("href", "/artifacts/projects/existing");

    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Next" }));
    expect(
      await screen.findByRole("link", { name: "projects/next-page" }),
    ).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Previous" }));
    expect(
      await screen.findByRole("link", { name: "projects/existing" }),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Upload file" }));
    const file = new File(["zip"], "source.zip", {
      type: "application/zip",
    });
    const dropTarget = screen.getByText("Drop a file here").parentElement;
    expect(dropTarget).not.toBeNull();
    fireEvent.drop(dropTarget!, { dataTransfer: { files: [file] } });
    expect(screen.getByLabelText("Name")).toHaveValue("source");
    expect(screen.getByLabelText("Media type")).toHaveValue("application/zip");
    await user.click(screen.getByRole("button", { name: "Create binding" }));

    await vi.waitFor(() =>
      expect(sent("PUT", "/v1/artifacts/projects/source")).toHaveLength(1),
    );
    const notice = await screen.findByRole("status", {
      name: /File uploaded to your library/,
    });
    expect(notice).toHaveTextContent("projects/source");
    expect(
      within(notice).getByRole("link", { name: "Open file" }),
    ).toHaveAttribute("href", "/artifacts/projects/source?revision=revision-2");
    await vi.waitFor(() => expect(notice).toHaveFocus());
    const put = sent("PUT", "/v1/artifacts/projects/source")[0];
    expect(put?.headers.get("If-None-Match")).toBe("*");
    expect(put?.headers.get("If-Match")).toBeNull();
    expect(put?.headers.get("X-CSRF-Token")).toBe(CSRF_TOKEN);
    expect(put?.headers.get("Content-Type")).toBe("application/zip");
  });

  it("reconciles a CAS conflict and never retries the unsafe PUT", async () => {
    const { sent } = renderApplication("/artifacts", {
      [LIST]: () => json(page([])),
      [WRITE]: () => failure(409, "conflict", "resource state changed", true),
    });
    await screen.findByText("No files yet");
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Upload file" }));
    await user.type(screen.getByLabelText("Name"), "source");
    await user.upload(
      screen.getByLabelText("Drop a file here"),
      new File(["next"], "source.txt", { type: "text/plain" }),
    );
    await user.click(screen.getByRole("button", { name: "Create binding" }));

    expect(await screen.findByText(/binding changed/i)).toBeInTheDocument();
    // The list is read again after the refused write.
    await vi.waitFor(() =>
      expect(sent("GET", "/v1/artifacts").length).toBeGreaterThanOrEqual(2),
    );
    expect(sent("PUT", "/v1/artifacts/projects/source")).toHaveLength(1);
  });

  it("keeps Skill packages on the dedicated Skills surface", async () => {
    const { sent } = renderApplication("/artifacts", {
      [LIST]: ({ url }) => {
        expect(url.searchParams.get("excludeNamespace")).toBe("skills");
        return json(page([]));
      },
    });
    await screen.findByText("No files yet");

    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Upload file" }));
    const uploadForm = screen.getByRole("dialog", { name: "Upload file" });
    const namespace = within(uploadForm).getByLabelText("Namespace");
    await user.clear(namespace);
    await user.type(namespace, "skills");
    await user.upload(
      within(uploadForm).getByLabelText("Drop a file here"),
      new File(["zip"], "reviewed-skill.zip", {
        type: "application/vnd.contractor.agent-skill+zip",
      }),
    );
    await user.click(
      within(uploadForm).getByRole("button", { name: "Create binding" }),
    );

    const alert = await screen.findByRole("alert");
    expect(within(alert).getByRole("link", { name: "Skills" })).toHaveAttribute(
      "href",
      "/catalog/skills",
    );
    expect(sent("PUT", "")).toHaveLength(0);
  });

  it("renders exact history/lineage and escapes bounded preview text", async () => {
    const preview = "<script>alert('not executable')</script>";
    const selected = artifactMetadata("projects", "architecture", {
      artifact: {
        namespace: "projects",
        name: "architecture",
        revision: "revision-2",
      },
      mediaType: "text/plain",
      size: preview.length,
      current: false,
      createdAt: "2026-08-31T12:00:00Z",
    });
    const current = {
      ...selected,
      artifact: { ...selected.artifact, revision: "revision-3" },
      current: true,
    };
    renderApplication("/artifacts/projects/architecture?revision=revision-2", {
      [METADATA]: () => json(selected),
      [VERSIONS]: () => json(page([current, selected])),
      [LINEAGE]: () =>
        json(
          page([
            conforming("ArtifactLineageEdge", {
              kind: "input_fork",
              sourceScope: "user",
              source: selected.artifact,
              targetScope: "run",
              target: {
                namespace: "inputs",
                name: "architecture",
                revision: "revision-run-1",
              },
              runId: "run-example",
              stageExecutionId: "stage-example",
              createdAt: "2026-08-31T12:01:00Z",
            }),
            conforming("ArtifactLineageEdge", {
              kind: "project_output_publish",
              sourceScope: "run",
              source: selected.artifact,
              targetScope: "project",
              target: selected.artifact,
              createdAt: "2026-08-31T12:02:00Z",
            }),
          ]),
        ),
      "GET /v1/artifacts/{namespace}/{name}": () =>
        payload(preview, "text/plain"),
    });
    expect(
      (await screen.findAllByText("revision-2", { selector: "code" })).length,
    ).toBeGreaterThan(0);
    expect(
      screen.getByText("Historical revision", { selector: ".lede" }),
    ).toBeVisible();
    expect(screen.queryByText("input fork")).not.toBeInTheDocument();
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Versions" }));
    expect(await screen.findByText("input fork")).toBeInTheDocument();
    expect(screen.getByText("project output publish")).toBeInTheDocument();
    expect(screen.getByText("Run run-example")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Load preview" }));
    expect(await screen.findByText(preview)).toBeInTheDocument();
    expect(document.querySelector("script")).toBeNull();
    expect(
      screen.getByText("Historical revisions are read-only."),
    ).toBeInTheDocument();
  });

  it("keeps contextual return navigation and current status after a user Artifact version upload", async () => {
    const original = source("revision-1");
    const latest = source("revision-2");
    const returnState = {
      returnTo: "/artifacts?namespace=projects&cursor=page-two",
      returnLabel: "Filtered Artifacts",
      returnState: { returnTo: "/catalog/skills", returnLabel: "Skills" },
    };
    let written = false;
    const { router } = renderApplication(
      {
        pathname: "/artifacts/projects/source",
        search: "?revision=revision-1",
        state: returnState,
      },
      {
        [WRITE]: ({ request }) => {
          expect(request.headers.get("If-Match")).toBe('"revision-1"');
          written = true;
          return json(artifactWrite(latest), 201, { ETag: '"revision-2"' });
        },
        [METADATA]: ({ url }) =>
          json(
            url.searchParams.get("revision") === "revision-2"
              ? latest
              : { ...original, current: !written },
          ),
        [VERSIONS]: () => json(page([latest, { ...original, current: false }])),
        [LINEAGE]: () => json(page([])),
        [LIST]: () => json(page([])),
      },
    );
    expect(
      await screen.findByText("Current revision", { selector: ".lede" }),
    ).toBeVisible();
    const user = userEvent.setup();
    await user.click(
      screen.getByText("Upload a new version", { selector: "summary" }),
    );
    await user.upload(
      screen.getByLabelText("Drop a file here"),
      new File(["new!"], "source.txt", { type: "text/plain" }),
    );
    await user.click(
      screen.getByRole("button", { name: "Upload new version" }),
    );
    await vi.waitFor(() =>
      expect(router.state.location.search).toBe("?revision=revision-2"),
    );
    expect(router.state.location.state).toEqual(returnState);
    expect(
      await screen.findByText("Current revision", { selector: ".lede" }),
    ).toBeVisible();
    const back = screen.getByRole("link", { name: "← Filtered Artifacts" });
    expect(back).toHaveAttribute("href", returnState.returnTo);
    await user.click(screen.getByRole("button", { name: "Versions" }));
    await user.click(await screen.findByRole("link", { name: /revision-1/ }));
    await vi.waitFor(() =>
      expect(router.state.location.search).toBe("?revision=revision-1"),
    );
    expect(router.state.location.state).toEqual(returnState);
    expect(
      await screen.findByText("Historical revision", { selector: ".lede" }),
    ).toBeVisible();
    await user.click(
      screen.getByRole("link", { name: "← Filtered Artifacts" }),
    );
    await vi.waitFor(() =>
      expect(router.state.location.pathname).toBe("/artifacts"),
    );
    expect(router.state.location.state).toEqual(returnState.returnState);
  });

  it.each([
    { pinned: false, path: { pathname: "/artifacts/projects/source" } },
    {
      pinned: true,
      path: {
        pathname: "/artifacts/projects/source",
        search: "?revision=revision-1",
      },
    },
  ])(
    "keeps a concurrent-writer upload conflict visible with the new current revision (pinned: $pinned)",
    async ({ path }) => {
      const original = source("revision-1");
      const concurrent = source("revision-2", {
        size: 6,
        createdAt: "2026-09-01T10:05:00Z",
      });
      const mine = source("revision-3");
      const returnState = { returnTo: "/artifacts", returnLabel: "Artifacts" };
      // Another writer binds revision-2 while this page shows revision-1.
      let concurrentWriter = false;
      const { router, sent } = renderApplication(
        { search: "", ...path, state: returnState },
        {
          [WRITE]: ({ request }) => {
            if (request.headers.get("If-Match") === '"revision-2"') {
              return json(artifactWrite(mine), 200, { ETag: '"revision-3"' });
            }
            concurrentWriter = true;
            return failure(
              409,
              "conflict",
              "resource state changed; retry with the current revision",
              true,
            );
          },
          [METADATA]: ({ url }) => {
            switch (url.searchParams.get("revision")) {
              case "revision-1":
                return json({ ...original, current: !concurrentWriter });
              case "revision-3":
                return json(mine);
            }
            return json(concurrentWriter ? concurrent : original);
          },
          [LIST]: () => json(page([])),
        },
      );
      const writes = () => sent("PUT", "/v1/artifacts/projects/source");
      expect(
        await screen.findByText("Current revision", { selector: ".lede" }),
      ).toBeVisible();
      const user = userEvent.setup();
      await user.click(
        screen.getByText("Upload a new version", { selector: "summary" }),
      );
      await user.upload(
        screen.getByLabelText("Drop a file here"),
        new File(["mine"], "source.txt", { type: "text/plain" }),
      );
      await user.click(
        screen.getByRole("button", { name: "Upload new version" }),
      );

      // The page and the new-version form moved to the new current revision.
      expect(
        await screen.findByText("revision-2", { selector: "code" }),
      ).toBeInTheDocument();
      expect(
        screen.getByText("Current revision", { selector: ".lede" }),
      ).toBeVisible();
      const notice = screen.getByRole("alert");
      expect(notice).toHaveTextContent(/based on revision-1 was rejected/);
      expect(notice).toHaveTextContent(/binding changed/);
      expect(router.state.location.search).toBe("");
      expect(router.state.location.state).toEqual(returnState);
      expect(writes()).toHaveLength(1);
      expect(writes()[0]?.headers.get("If-Match")).toBe('"revision-1"');

      // An explicit new upload against the reviewed revision clears it.
      await user.upload(
        screen.getByLabelText("Drop a file here"),
        new File(["mine"], "source.txt", { type: "text/plain" }),
      );
      await user.click(
        screen.getByRole("button", { name: "Upload new version" }),
      );
      await vi.waitFor(() =>
        expect(router.state.location.search).toBe("?revision=revision-3"),
      );
      expect(
        await screen.findByText("revision-3", { selector: "code" }),
      ).toBeInTheDocument();
      expect(screen.queryByRole("alert")).not.toBeInTheDocument();
      expect(writes().map((write) => write.headers.get("If-Match"))).toEqual([
        '"revision-1"',
        '"revision-2"',
      ]);
    },
  );

  it("stays where the user went when a version upload answers late", async () => {
    let answer!: (response: Response) => void;
    const original = source("revision-1");
    const { router } = renderApplication("/artifacts/projects/source", {
      [METADATA]: () => json(original),
      [WRITE]: () =>
        new Promise<Response>((resolve) => {
          answer = resolve;
        }),
      [LIST]: () => json(page([])),
    });
    const user = userEvent.setup();
    await user.click(
      await screen.findByText("Upload a new version", { selector: "summary" }),
    );
    await user.upload(
      screen.getByLabelText("Drop a file here"),
      new File(["new!"], "source.txt", { type: "text/plain" }),
    );
    await user.click(
      screen.getByRole("button", { name: "Upload new version" }),
    );
    await waitFor(() => expect(answer).toBeDefined());
    await act(async () => {
      await router.navigate("/artifacts");
    });
    await screen.findByText("No files yet");
    await act(async () => {
      answer(
        json(artifactWrite(source("revision-2")), 201, {
          ETag: '"revision-2"',
        }),
      );
    });
    await waitFor(() =>
      expect(router.state.location.pathname).toBe("/artifacts"),
    );
    expect(router.state.location.search).toBe("");
  });

  it("does not offer inline preview for binary content", async () => {
    const binary = source("revision-1", {
      mediaType: "application/octet-stream",
      size: 1024,
    });
    const { sent } = renderApplication(
      "/artifacts/projects/source?revision=revision-1",
      {
        [METADATA]: () => json(binary),
        [VERSIONS]: () => json(page([binary])),
        [LINEAGE]: () => json(page([])),
      },
    );
    const previewButton = await screen.findByRole("button", {
      name: "Load preview",
    });
    expect(
      screen.getByText("Current revision", { selector: ".lede" }),
    ).toBeVisible();
    expect(previewButton).toBeDisabled();
    expect(
      screen.getByText(/Inline preview is unavailable/),
    ).toBeInTheDocument();
    // A binary preview never reads the Artifact bytes.
    expect(sent("GET", "/v1/artifacts/projects/source")).toHaveLength(0);
  });

  it("restores the namespace and cursor after inspecting an Artifact", async () => {
    const binding = artifactMetadata("sources", "example", {
      mediaType: "text/plain",
      size: 4,
      createdAt: "2026-09-01T10:00:00Z",
    });
    const { router } = renderApplication(
      "/artifacts?namespace=sources&cursor=page-two",
      {
        [METADATA]: () => json(binding),
        [LIST]: ({ url }) => {
          expect(url.searchParams.get("namespace")).toBe("sources");
          expect(url.searchParams.get("cursor")).toBe("page-two");
          return json(page([binding]));
        },
      },
    );
    const user = userEvent.setup();
    await user.click(
      await screen.findByRole("link", { name: "sources/example" }),
    );
    await user.click(await screen.findByRole("link", { name: "← Files" }));
    expect(router.state.location.search).toBe(
      "?namespace=sources&cursor=page-two",
    );
    expect(screen.getByPlaceholderText("all namespaces")).toHaveValue(
      "sources",
    );
    expect(
      await screen.findByRole("link", { name: "sources/example" }),
    ).toBeVisible();
  });
});

describe("Files upload dialog", () => {
  /** Starts a file upload whose PUT stays pending until it is aborted. */
  async function startPendingUpload() {
    let signal: AbortSignal | undefined;
    const { sent } = renderApplication("/artifacts", {
      [LIST]: () => json(page([])),
      [WRITE]: ({ request, params }) => {
        expect(params).toEqual({ namespace: "inputs", name: "wordlist" });
        signal = request.signal;
        return pendingUntilAborted(request);
      },
    });
    const listReads = () => sent("GET", "/v1/artifacts").length;
    await screen.findByText("No files yet");
    const user = userEvent.setup();
    const uploadButton = screen.getByRole("button", { name: "Upload file" });
    await user.click(uploadButton);
    const dialog = screen.getByRole("dialog", { name: "Upload file" });
    // Closing cancels the upload, which may still have been stored.
    expect(dialog).toHaveAccessibleDescription(
      /^Closing this dialog cancels a running upload\. .* it may already be stored\.$/,
    );
    const namespace = within(dialog).getByLabelText("Namespace");
    await user.clear(namespace);
    await user.type(namespace, "inputs");
    await user.upload(
      within(dialog).getByLabelText("Drop a file here"),
      new File(["admin\n"], "wordlist.txt", { type: "text/plain" }),
    );
    await user.click(
      within(dialog).getByRole("button", { name: "Create binding" }),
    );
    await waitFor(() => expect(signal).toBeDefined());
    expect(
      within(dialog).getByRole("button", { name: "Uploading…" }),
    ).toBeDisabled();
    return {
      user,
      dialog,
      uploadButton,
      listReadsBefore: listReads(),
      listReads,
      aborted: () => signal?.aborted,
    };
  }

  type PendingUpload = Awaited<ReturnType<typeof startPendingUpload>>;

  it.each([
    [
      "Cancel upload",
      (pending: PendingUpload) =>
        pending.user.click(
          within(pending.dialog).getByRole("button", {
            name: "Cancel upload",
          }),
        ),
    ],
    [
      "the close button",
      (pending: PendingUpload) =>
        pending.user.click(
          within(pending.dialog).getByRole("button", {
            name: "Close file upload",
          }),
        ),
    ],
    ["Escape", (pending: PendingUpload) => pending.user.keyboard("{Escape}")],
  ])(
    "aborts a running upload closed with %s, then reads the list again",
    async (_how, close) => {
      const pending = await startPendingUpload();
      await act(async () => {
        await close(pending);
      });

      expect(pending.aborted()).toBe(true);
      expect(pending.dialog).not.toBeInTheDocument();
      // The Server may have stored the file before the abort.
      await waitFor(() =>
        expect(pending.listReads()).toBeGreaterThan(pending.listReadsBefore),
      );
      await waitFor(() => expect(pending.uploadButton).toHaveFocus());
      expect(
        screen.queryByRole("status", { name: /File uploaded/ }),
      ).toBeNull();
    },
  );
});
