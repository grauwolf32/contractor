import { act, screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { artifactMetadata, artifactWrite } from "../../../test/builders";
import { deferred } from "../../../test/deferred";
import { bytes, failure, json } from "../../../test/responses";
import {
  materialFixture,
  profileFixture,
  sourceZip,
  top10,
} from "./test-fixtures";
import { renderStart } from "./test-support";

const path = "/checks/new?project=project_shop";
function picker() {
  return screen.getByRole("dialog", { name: "Choose source code" });
}
const first = {
  ...materialFixture("source-a", "text/plain"),
  size: new TextEncoder().encode("Saved source-a-r1").length,
};
const second = {
  ...materialFixture("source-b", "text/plain"),
  size: new TextEncoder().encode("Saved source-b-r1").length,
};
const profile = profileFixture("text-review", {
  inputs: { source: { required: true, media_types: ["text/plain"] } },
});

describe("New check material chooser", () => {
  it("previews an exact candidate, cancels without choosing, and confirms only on Attach", async () => {
    const view = renderStart(path, {
      profiles: [profile],
      materials: [first, second],
      content: (_request, url) =>
        bytes(`Saved ${url.searchParams.get("revision")}`, "text/plain"),
    });
    const choose = await screen.findByRole("button", {
      name: "Choose source code material",
    });
    await view.user.type(
      screen.getByLabelText("Your objective, in your own words"),
      "Retain this objective",
    );
    await view.user.selectOptions(
      screen.getByLabelText("Time limit"),
      "604800",
    );
    await view.user.click(choose);
    await view.user.click(
      within(picker()).getByRole("button", {
        name: "sources/source-a@source-a-r1",
      }),
    );
    expect(
      await within(picker()).findByText("Saved source-a-r1"),
    ).toBeVisible();
    await view.user.click(
      within(picker()).getByRole("button", { name: "Cancel" }),
    );
    expect(choose).toBeVisible();
    await view.user.click(choose);
    await view.user.click(
      within(picker()).getByRole("button", {
        name: "sources/source-b@source-b-r1",
      }),
    );
    await view.user.click(
      within(picker()).getByRole("button", { name: "Attach material" }),
    );
    expect(
      screen.getByLabelText("Your objective, in your own words"),
    ).toHaveValue("Retain this objective");
    expect(screen.getByLabelText("Time limit")).toHaveValue("604800");
    await view.user.click(
      screen.getByRole("button", { name: "Save as draft" }),
    );
    await waitFor(() => expect(view.sent("POST", "/audits")).toHaveLength(1));
    expect(await view.sent("POST", "/audits")[0]!.json()).toMatchObject({
      inputs: { source: second.artifact },
    });
  });

  it("does not attach a selected candidate hidden by the search", async () => {
    const view = renderStart(path, {
      profiles: [profile],
      materials: [first, second],
    });
    await view.user.click(
      await screen.findByRole("button", {
        name: "Choose source code material",
      }),
    );
    await view.user.click(
      within(picker()).getByRole("button", {
        name: "sources/source-a@source-a-r1",
      }),
    );
    await view.user.type(within(picker()).getByRole("searchbox"), "missing");
    expect(
      within(picker()).getByRole("button", { name: "Attach material" }),
    ).toBeDisabled();
    expect(within(picker()).getByText("No matching materials")).toBeVisible();
    await view.user.click(
      within(picker()).getByRole("button", { name: "Cancel" }),
    );
    expect(
      screen.getByRole("button", { name: "Choose source code material" }),
    ).toBeVisible();
  });

  it("retries the exact metadata read after an upload without repeating the write", async () => {
    let failRead = true;
    const added = artifactMetadata("sources", "new-source", {
      mediaType: "application/zip",
      size: 7,
      artifact: {
        namespace: "sources",
        name: "new-source",
        revision: "saved-r7",
      },
    });
    const view = renderStart(path, {
      profiles: [top10],
      materials: [sourceZip],
      upload: () => json(artifactWrite(added), 201, { ETag: '"saved-r7"' }),
      metadata: () =>
        failRead
          ? failure(503, "server_unavailable")
          : json(added, 200, { ETag: '"saved-r7"' }),
    });
    await view.user.click(
      await screen.findByRole("button", {
        name: "Change source code material",
      }),
    );
    await view.user.click(
      within(picker()).getByRole("button", { name: "Add material" }),
    );
    const upload = screen.getByRole("dialog", { name: "Add source code" });
    await view.user.upload(
      within(upload).getByLabelText("Drop a file here"),
      new File(["fixture"], "new-source.zip", { type: "application/zip" }),
    );
    await view.user.click(
      within(upload).getByRole("button", { name: "Add and attach" }),
    );
    await waitFor(() => expect(upload).not.toBeInTheDocument());
    expect(
      await within(picker()).findByText(
        "The material was added, but its exact version could not be read.",
      ),
    ).toBeVisible();
    expect(view.sent("PUT", "/sources/new-source")).toHaveLength(1);
    failRead = false;
    await view.user.click(
      within(picker()).getByRole("button", { name: "Try again" }),
    );
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(screen.getByText("saved-r7")).toBeVisible();
    expect(view.sent("PUT", "/sources/new-source")).toHaveLength(1);
  });

  it("ignores an uploaded material's late read after closing the chooser", async () => {
    const answer = deferred<Response>();
    const added = artifactMetadata("sources", "new-source", {
      mediaType: "application/zip",
      size: 7,
    });
    const view = renderStart(path, {
      profiles: [top10],
      materials: [sourceZip],
      upload: () =>
        json(artifactWrite(added), 201, {
          ETag: `"${added.artifact.revision}"`,
        }),
      metadata: () => answer.promise,
    });
    await view.user.click(
      await screen.findByRole("button", {
        name: "Change source code material",
      }),
    );
    await view.user.click(
      within(picker()).getByRole("button", { name: "Add material" }),
    );
    const upload = screen.getByRole("dialog", { name: "Add source code" });
    await view.user.upload(
      within(upload).getByLabelText("Drop a file here"),
      new File(["fixture"], "new-source.zip", { type: "application/zip" }),
    );
    await view.user.click(
      within(upload).getByRole("button", { name: "Add and attach" }),
    );
    await waitFor(() => expect(upload).not.toBeInTheDocument());
    await within(picker()).findByText(
      "Material added. Reading its saved version…",
    );
    await view.user.click(
      within(picker()).getByRole("button", { name: "Close material chooser" }),
    );
    await act(async () =>
      answer.resolve(
        json(added, 200, { ETag: `"${added.artifact.revision}"` }),
      ),
    );
    expect(screen.queryByText("new-source")).toBeNull();
    expect(screen.getByText(sourceZip.artifact.revision)).toBeVisible();
  });
});
