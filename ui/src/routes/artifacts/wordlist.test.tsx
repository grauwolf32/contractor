import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router";
import { describe, expect, it, vi } from "vitest";

import { PublicAPI } from "../../api/client";
import { PublicAPIProvider } from "../../api/context";
import { ProjectArtifactWriteForm } from "../projects/common";
import { ArtifactWriteForm } from "./common";
import { inferredArtifactMediaType } from "./artifact-file";
import { LoadedArtifactPreview } from "./loaded-preview";

const wordlist = "text/vnd.contractor.wordlist";

describe("Wordlist Artifact journey", () => {
  it.each(["user", "project"])(
    "uploads a selected wordlist semantic type in %s scope without MIME inference replacing it",
    async (scope) => {
      const writes: Request[] = [];
      const onWritten = vi.fn();
      const artifact = {
        namespace: "inputs",
        name: "paths",
        revision: "revision-1",
      };
      const api = new PublicAPI(
        {
          uiVersion: "0.1.0",
          supportedApiVersions: ["contractor.public.v1"],
          apiBaseUrl: "http://127.0.0.1:8080",
        },
        vi.fn(async (input) => {
          const request = input instanceof Request ? input : new Request(input);
          writes.push(request);
          return new Response(
            JSON.stringify({ artifact, mediaType: wordlist, size: 6 }),
            {
              status: 201,
              headers: {
                "Content-Type": "application/json",
                "X-Contractor-API-Version": "contractor.public.v1",
                ETag: '"revision-1"',
              },
            },
          );
        }),
      );
      api.csrf.replace("a".repeat(43));
      render(
        <QueryClientProvider client={new QueryClient()}>
          <PublicAPIProvider api={api}>
            <MemoryRouter>
              {scope === "user" ? (
                <ArtifactWriteForm
                  fixedNamespace="inputs"
                  onWritten={onWritten}
                />
              ) : (
                <ProjectArtifactWriteForm
                  projectId="project_example"
                  fixedNamespace="inputs"
                  onWritten={onWritten}
                />
              )}
            </MemoryRouter>
          </PublicAPIProvider>
        </QueryClientProvider>,
      );
      const user = userEvent.setup();
      await user.selectOptions(screen.getByLabelText("File format"), wordlist);
      await user.upload(
        screen.getByLabelText("Drop a file here"),
        new File(["a\nb\nc\n"], "paths.txt", { type: "text/plain" }),
      );
      expect(screen.getByLabelText("Media type")).toHaveValue(wordlist);
      expect(screen.getByLabelText("Name")).toHaveValue("paths");
      await user.click(screen.getByRole("button", { name: "Create binding" }));
      await waitFor(() => expect(onWritten).toHaveBeenCalledOnce());
      expect(onWritten).toHaveBeenCalledWith({
        artifact,
        mediaType: wordlist,
        size: 6,
      });
      expect(writes).toHaveLength(1);
      const request = writes[0]!;
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
