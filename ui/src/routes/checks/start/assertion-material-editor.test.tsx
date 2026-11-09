import { screen, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { artifactMetadata, artifactWrite } from "../../../test/builders";
import { failure, json } from "../../../test/responses";
import {
  assertionProfileFixture,
  sourceZip,
  materialFixture,
} from "./test-fixtures";
import { renderStart } from "./test-support";

describe("current checklist authoring in New check", () => {
  it("starts with one item and saves stable keys after removal without starting an Audit", async () => {
    let saved = artifactMetadata("materials", "assertions", {
      mediaType: "application/json",
    });
    const writes: Array<{
      items: Array<{ key: string; statement: string; grounds: string }>;
      schema: string;
    }> = [];
    const view = renderStart(
      "/checks/new?project=project_shop&type=source-checklist",
      {
        profiles: [assertionProfileFixture()],
        materials: [sourceZip],
        upload: async (request, url) => {
          const text = await request.text();
          writes.push(JSON.parse(text));
          saved = artifactMetadata(
            "materials",
            decodeURIComponent(url.pathname.split("/").at(-1)!),
            {
              mediaType: "application/json",
              size: new TextEncoder().encode(text).byteLength,
            },
          );
          return json(artifactWrite(saved), 201, {
            ETag: `"${saved.artifact.revision}"`,
          });
        },
        metadata: () => json(saved),
      },
    );
    await view.user.click(
      await screen.findByRole("button", { name: "Write a checklist" }),
    );
    expect(screen.getByLabelText(/^Check 1/)).toHaveValue("");
    expect(screen.getByRole("button", { name: "Start check" })).toBeDisabled();
    expect(screen.queryByLabelText("Check 2")).not.toBeInTheDocument();
    expect(
      screen.queryByRole("region", { name: "Scope" }),
    ).not.toBeInTheDocument();
    expect(view.requests.filter((r) => r.method !== "GET")).toHaveLength(0);
    await view.user.type(
      screen.getByLabelText(/^Check 1/),
      "Private exports require their owning project",
    );
    await view.user.click(screen.getByRole("button", { name: "Add check" }));
    await view.user.type(screen.getByLabelText(/^Check 2/), "Temporary claim");
    await view.user.click(screen.getByRole("button", { name: "Add check" }));
    await view.user.type(
      screen.getByLabelText(/^Check 3/),
      "Exports cannot cross project boundaries",
    );
    await view.user.click(
      screen.getByRole("button", { name: "Remove check 2" }),
    );
    await view.user.type(
      screen.getByLabelText(/^Why check these/),
      "Exports contain private project data",
    );
    await view.user.click(
      screen.getByRole("button", { name: "Save and attach checklist" }),
    );
    await waitFor(() => expect(writes).toHaveLength(1));
    expect(writes[0]?.schema).toBe("contractor.audit.assertions.v1");
    expect(writes[0]?.items.map((i) => i.key)).toEqual([
      "assertion-1",
      "assertion-3",
    ]);
    expect(writes[0]?.items[1]).toMatchObject({
      statement: "Exports cannot cross project boundaries",
      grounds: "Exports contain private project data",
    });
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Start check" })).toBeEnabled(),
    );
    expect(view.requests.filter((r) => r.method === "POST")).toHaveLength(0);
  });
  it("requires saving or closing the editor before using an already attached checklist", async () => {
    const view = renderStart(
      "/checks/new?project=project_shop&type=source-checklist",
      {
        profiles: [assertionProfileFixture()],
        materials: [
          sourceZip,
          materialFixture("existing-checklist", "application/json"),
        ],
      },
    );
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Start check" })).toBeEnabled(),
    );
    await view.user.click(
      screen.getByRole("button", { name: "Write a checklist" }),
    );
    expect(screen.getByRole("button", { name: "Start check" })).toBeDisabled();
    expect(
      screen.getByRole("button", { name: "Save as draft" }),
    ).toBeDisabled();
    await view.user.click(
      screen.getByRole("button", { name: "Close checklist editor" }),
    );
    expect(screen.getByRole("button", { name: "Start check" })).toBeEnabled();
    expect(
      view.requests.filter((request) => request.method !== "GET"),
    ).toHaveLength(0);
  });
  it("keeps entries after a failed retention and retries the same exact material", async () => {
    const paths: string[] = [];
    let fail = true;
    let saved = artifactMetadata("materials", "assertions", {
      mediaType: "application/json",
    });
    const view = renderStart(
      "/checks/new?project=project_shop&type=source-checklist",
      {
        profiles: [assertionProfileFixture()],
        materials: [sourceZip],
        upload: async (request, url) => {
          paths.push(url.pathname);
          if (fail) return failure(503, "storage_transaction_conflict");
          saved = artifactMetadata(
            "materials",
            url.pathname.split("/").at(-1)!,
            {
              mediaType: "application/json",
              size: (await request.arrayBuffer()).byteLength,
            },
          );
          return json(artifactWrite(saved), 201, {
            ETag: `"${saved.artifact.revision}"`,
          });
        },
        metadata: () => (fail ? failure(404, "not_found") : json(saved)),
      },
    );
    await view.user.click(
      await screen.findByRole("button", { name: "Write a checklist" }),
    );
    await view.user.type(
      screen.getByLabelText(/^Check 1/),
      "A private export stays within its project",
    );
    await view.user.type(
      screen.getByLabelText(/^Why check these/),
      "Private data boundary",
    );
    await view.user.click(
      screen.getByRole("button", { name: "Save and attach checklist" }),
    );
    await screen.findByRole("alert");
    expect(screen.getByLabelText(/^Check 1/)).toHaveValue(
      "A private export stays within its project",
    );
    fail = false;
    await view.user.click(
      screen.getByRole("button", { name: "Save and attach checklist" }),
    );
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Start check" })).toBeEnabled(),
    );
    expect(paths).toHaveLength(2);
    expect(paths[0]).toBe(paths[1]);
  });
});
