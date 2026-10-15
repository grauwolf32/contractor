import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { fakeServer, json } from "../../test/fake-server";
import { renderWithAPI } from "../../test/render";
import { inferredArtifactMediaType } from "./artifact-file";
import { ArtifactWriteForm } from "./common";
import { LoadedArtifactPreview } from "./loaded-preview";
import { artifactMetadata, artifactWrite } from "./test-support";

const wordlist = "text/vnd.contractor.wordlist";

describe("Wordlist Artifact journey", () => {
  it.each(["user", "project"])(
    "uploads a selected wordlist semantic type in %s scope without MIME inference replacing it",
    async (scope) => {
      const onWritten = vi.fn();
      const written = artifactWrite(
        artifactMetadata("inputs", "paths", {
          artifact: {
            namespace: "inputs",
            name: "paths",
            revision: "revision-1",
          },
          mediaType: wordlist,
          size: 6,
        }),
      );
      const write = () => json(written, 201, { ETag: '"revision-1"' });
      const { user, requests } = renderWithAPI(
        scope === "user" ? (
          <ArtifactWriteForm fixedNamespace="inputs" onWritten={onWritten} />
        ) : (
          <ArtifactWriteForm
            scope={{ kind: "project", id: "project_example" }}
            fixedNamespace="inputs"
            onWritten={onWritten}
          />
        ),
        {
          server: fakeServer({
            "PUT /v1/artifacts/{namespace}/{name}": write,
            "PUT /v1/projects/{projectId}/artifacts/{namespace}/{name}": write,
          }),
        },
      );
      await user.selectOptions(screen.getByLabelText("File format"), wordlist);
      await user.upload(
        screen.getByLabelText("Drop a file here"),
        new File(["a\nb\nc\n"], "paths.txt", { type: "text/plain" }),
      );
      expect(screen.getByLabelText("Media type")).toHaveValue(wordlist);
      expect(screen.getByLabelText("Name")).toHaveValue("paths");
      await user.click(screen.getByRole("button", { name: "Create binding" }));
      await waitFor(() => expect(onWritten).toHaveBeenCalledOnce());
      expect(onWritten).toHaveBeenCalledWith(written);
      expect(requests).toHaveLength(1);
      const request = requests[0]!;
      expect(new URL(request.url).pathname).toBe(
        scope === "user"
          ? "/v1/artifacts/inputs/paths"
          : "/v1/projects/project_example/artifacts/inputs/paths",
      );
      expect(request.headers.get("Content-Type")).toBe(wordlist);
      expect(request.headers.get("If-None-Match")).toBe("*");
      expect(await request.text()).toBe("a\nb\nc\n");
    },
  );

  it("infers ordinary .txt files as text and previews wordlist payloads literally", async () => {
    expect(
      inferredArtifactMediaType(
        new File(["a"], "paths.TXT"),
        "application/octet-stream",
      ),
    ).toBe("text/plain");
    const source = "<script>example</script>\n# literal payload\n";
    const { container } = render(
      <LoadedArtifactPreview mediaType={wordlist} source={source} />,
    );
    await waitFor(() =>
      expect(container.querySelector("pre")).toHaveTextContent(
        "<script>example</script>",
      ),
    );
    expect(container.querySelector("pre")?.textContent).toBe(source);
    expect(container.querySelector("script")).toBeNull();
  });
});
