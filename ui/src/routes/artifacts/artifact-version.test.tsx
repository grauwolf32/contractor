import { screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { fakeServer, json } from "../../test/fake-server";
import { renderWithAPI } from "../../test/render";
import {
  MAXIMUM_SKILL_ARCHIVE_BYTES,
  SKILL_ARCHIVE_MEDIA_TYPE,
} from "./artifact-file";
import { ArtifactWriteForm } from "./common";
import { ArtifactDetailRoute } from "./detail";
import { artifactMetadata, artifactWrite } from "./test-support";

describe("Artifact version media types", () => {
  it.each(["user", "project"])(
    "keeps the %s binding's semantic type when a file has a generic browser MIME type",
    async (scope) => {
      const mediaType = "application/vnd.contractor.workspace-overlay+json";
      const onWritten = vi.fn();
      const write = () =>
        json(
          artifactWrite(
            artifactMetadata("overlays", "baseline", {
              artifact: {
                namespace: "overlays",
                name: "baseline",
                revision: "revision-2",
              },
              mediaType,
              size: 2,
            }),
          ),
          201,
          { ETag: '"revision-2"' },
        );
      const form = {
        fixedIdentity: { namespace: "overlays", name: "baseline" },
        initialMediaType: mediaType,
        expectedRevision: "revision-1",
        onWritten,
      };
      const { user, requests } = renderWithAPI(
        scope === "user" ? (
          <ArtifactWriteForm {...form} />
        ) : (
          <ArtifactWriteForm
            scope={{ kind: "project", id: "project_example" }}
            {...form}
          />
        ),
        {
          server: fakeServer({
            "PUT /v1/artifacts/{namespace}/{name}": write,
            "PUT /v1/projects/{projectId}/artifacts/{namespace}/{name}": write,
          }),
        },
      );
      expect(screen.getByLabelText("Media type")).toHaveValue(mediaType);
      await user.upload(
        screen.getByLabelText("Drop a file here"),
        new File(["{}"], "baseline.json", { type: "application/json" }),
      );
      expect(screen.getByLabelText("Media type")).toHaveValue(mediaType);
      await user.click(
        screen.getByRole("button", { name: "Upload new version" }),
      );
      await waitFor(() => expect(onWritten).toHaveBeenCalledOnce());
      expect(requests).toHaveLength(1);
      expect(requests[0]?.headers.get("Content-Type")).toBe(mediaType);
      expect(requests[0]?.headers.get("If-Match")).toBe('"revision-1"');
    },
  );

  it.each(["user", "project"])(
    "preserves a newly selected %s format when a file is chosen later",
    async (scope) => {
      const { user, requests } = renderWithAPI(
        scope === "user" ? (
          <ArtifactWriteForm onWritten={vi.fn()} />
        ) : (
          <ArtifactWriteForm
            scope={{ kind: "project", id: "project_example" }}
            onWritten={vi.fn()}
          />
        ),
      );
      await user.selectOptions(
        screen.getByLabelText("File format"),
        "application/json",
      );
      await user.upload(
        screen.getByLabelText("Drop a file here"),
        new File(["notes"], "notes.txt", { type: "text/plain" }),
      );
      expect(screen.getByLabelText("Media type")).toHaveValue(
        "application/json",
      );
      expect(requests).toHaveLength(0);
    },
  );

  it("updates a Skill from its detail page with the canonical type and 16 MiB limit", async () => {
    const metadata = artifactMetadata("skills", "review", {
      artifact: { namespace: "skills", name: "review", revision: "revision-1" },
      mediaType: SKILL_ARCHIVE_MEDIA_TYPE,
      size: 3,
    });
    const { user, sent } = renderWithAPI(null, {
      path: "/artifacts/skills/review",
      routes: [
        {
          path: "/artifacts/:namespace/:name",
          element: <ArtifactDetailRoute />,
        },
      ],
      server: fakeServer({
        "GET /v1/artifacts/{namespace}/{name}/metadata": () => json(metadata),
        "PUT /v1/artifacts/{namespace}/{name}": () =>
          json(
            artifactWrite({
              ...metadata,
              artifact: { ...metadata.artifact, revision: "revision-2" },
            }),
            201,
            { ETag: '"revision-2"' },
          ),
      }),
    });
    const writes = () => sent("PUT", "/v1/artifacts/skills/review");
    await user.click(
      await screen.findByText("Upload a new version", { selector: "summary" }),
    );
    expect(screen.getByLabelText("Media type")).toHaveValue(
      SKILL_ARCHIVE_MEDIA_TYPE,
    );
    expect(screen.getByLabelText("Media type")).toBeDisabled();
    const oversized = new File(["zip"], "review.zip", {
      type: "application/zip",
    });
    Object.defineProperty(oversized, "size", {
      value: MAXIMUM_SKILL_ARCHIVE_BYTES + 1,
    });
    await user.upload(screen.getByLabelText("Drop a file here"), oversized);
    await user.click(
      screen.getByRole("button", { name: "Upload new version" }),
    );
    expect(screen.getByRole("alert")).toHaveTextContent("16 MiB upload limit");
    expect(writes()).toHaveLength(0);

    await user.upload(
      screen.getByLabelText("review.zip"),
      new File(["zip"], "review.zip", { type: "application/zip" }),
    );
    await user.click(
      screen.getByRole("button", { name: "Upload new version" }),
    );
    await waitFor(() => expect(writes()).toHaveLength(1));
    expect(writes()[0]?.headers.get("Content-Type")).toBe(
      SKILL_ARCHIVE_MEDIA_TYPE,
    );
    expect(writes()[0]?.headers.get("If-Match")).toBe('"revision-1"');
  });
});
