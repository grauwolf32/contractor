import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { renderApplication } from "../test/application";
import { conforming } from "../test/builders";
import { failure, json, page, type FakeServer } from "../test/fake-server";

function skill(name: string, revision: string) {
  return conforming("ArtifactMetadata", {
    artifact: { namespace: "skills", name, revision },
    mediaType: "application/zip",
    size: 2048,
    current: true,
    frozen: true,
    createdAt: "2026-09-01T12:00:00Z",
  });
}

/** A Skill package archive listing `paths`, as one SKILL.md-sized file each. */
function archive(name: string, revision: string, paths: string[]) {
  return json(
    conforming("ArtifactArchive", {
      artifact: { namespace: "skills", name, revision },
      entries: paths.map((path) => ({
        kind: "file",
        path,
        size: path === "SKILL.md" ? 48 : 4,
        previewable: true,
      })),
    }),
    200,
    { ETag: `"${revision}"` },
  );
}

/** Archive reads of the Skill package `name`. */
function archiveReads(server: FakeServer, name: string): number {
  return server.sent("GET", `/v1/artifacts/skills/${name}/archive`).length;
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
    const { server } = renderApplication("/catalog/skills", {
      server: {
        "GET /v1/artifacts": () =>
          json(page([skill("broken", "r1"), skill("plain", "r2")])),
        "GET /v1/artifacts/{namespace}/{name}/archive": ({ params }) => {
          if (params.name === "plain")
            return archive("plain", "r2", ["notes.txt"]);
          return archiveReads(server, "broken") === 1
            ? failure(503, "unavailable", "Archive store is unavailable")
            : archive("broken", "r1", ["SKILL.md"]);
        },
        "GET /v1/artifacts/{namespace}/{name}/archive/file": () => {
          const text = "---\ndescription: Reviews source evidence\n---\n# Hi\n";
          return json(
            conforming("ArtifactArchiveFile", {
              artifact: { namespace: "skills", name: "broken", revision: "r1" },
              path: "SKILL.md",
              text,
              size: new TextEncoder().encode(text).length,
            }),
            200,
            { ETag: '"r1"' },
          );
        },
      },
      capabilities: ["user"],
    });

    const broken = (
      await screen.findByRole("link", { name: "broken" })
    ).closest("li") as HTMLElement;
    const plain = screen
      .getByRole("link", { name: "plain" })
      .closest("li") as HTMLElement;

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
    expect(archiveReads(server, "broken")).toBe(2);
  });

  it("explains how to create the first package in Library words", async () => {
    renderApplication("/catalog/skills", {
      server: { "GET /v1/artifacts": () => json(page([])) },
      capabilities: ["user"],
    });

    const empty = (await screen.findByText("No global Skill packages found."))
      .parentElement!;
    expect(empty).toHaveTextContent(
      "Upload a Skill ZIP to create the first package.",
    );
    expect(empty).toHaveTextContent(/bundled with the server configuration/);
    // Files refuses the skills namespace, so the hint does not send users there.
    expect(empty).not.toHaveTextContent(/artifact|Files/i);
    expect(screen.getByRole("button", { name: "Upload Skills" })).toBeVisible();
  });

  it("lists packages with size, files and revision, and uploads a Skill ZIP", async () => {
    let stored = false;
    const { server } = renderApplication("/catalog/skills", {
      server: {
        "GET /v1/artifacts": () =>
          json(
            page(
              stored
                ? [skill("review", "r1"), skill("triage", "r9")]
                : [skill("review", "r1")],
            ),
          ),
        "GET /v1/artifacts/{namespace}/{name}/archive": ({ params }) =>
          params.name === "review"
            ? archive("review", "r1", ["notes.md"])
            : failure(503, "unavailable", "Later"),
        "PUT /v1/artifacts/{namespace}/{name}": () => {
          stored = true;
          return json(
            conforming("ArtifactWriteResponse", {
              artifact: { namespace: "skills", name: "triage", revision: "r9" },
              mediaType: "application/vnd.contractor.agent-skill+zip",
              size: 3,
            }),
            201,
            { ETag: '"r9"' },
          );
        },
      },
      capabilities: ["user"],
    });
    const puts = () => server.sent("PUT", "/v1/artifacts/skills/triage");

    const link = await screen.findByRole("link", { name: "review" });
    expect(link).toHaveAttribute("href", "/artifacts/skills/review");
    const row = link.closest("li") as HTMLElement;
    expect(within(row).getByText("2.0 KiB · Skill package")).toBeVisible();
    expect(await within(row).findByText("1 file")).toBeVisible();
    expect(within(row).getByText("Current revision")).toBeVisible();
    expect(within(row).getByText("r1")).toBeInTheDocument();

    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Upload Skills" }));
    const dialog = screen.getByRole("dialog", { name: "Upload Skills" });
    expect(dialog).toHaveTextContent("up to 16 MiB");
    expect(dialog).toHaveTextContent(
      "Stored as a Skill package: application/vnd.contractor.agent-skill+zip",
    );
    await user.type(within(dialog).getByLabelText("Name"), "triage");
    const oversized = new File(["zip"], "too-big.zip", {
      type: "application/zip",
    });
    Object.defineProperty(oversized, "size", { value: 16 * 1024 * 1024 + 1 });
    await user.upload(
      within(dialog).getByLabelText("Drop a file here"),
      oversized,
    );
    await user.click(within(dialog).getByRole("button", { name: "Upload" }));
    expect(within(dialog).getByRole("alert")).toHaveTextContent(
      "Skill package exceeds the 16 MiB upload limit.",
    );
    expect(puts()).toHaveLength(0);

    // The drop zone is now named by the chosen file.
    await user.upload(
      within(dialog).getByLabelText("too-big.zip"),
      new File(["zip"], "triage.zip", { type: "application/zip" }),
    );
    await user.click(within(dialog).getByRole("button", { name: "Upload" }));
    expect(
      await screen.findByText("Global Skill revision stored."),
    ).toBeVisible();
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(puts()).toHaveLength(1);
    expect(puts()[0]!.headers.get("Content-Type")).toBe(
      "application/vnd.contractor.agent-skill+zip",
    );
    expect(await screen.findByRole("link", { name: "triage" })).toHaveAttribute(
      "href",
      "/artifacts/skills/triage",
    );
  });
});
