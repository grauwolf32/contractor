import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import proof from "../../../../../api/testdata/finding-trace-evidence.json";
import { createArtifactPreviewPlan } from "../plan";
import { LoadedArtifactPreview } from "../view";
import { parseCodeEvidence } from "./code-evidence-document";
import CodeEvidencePreview from "./code-evidence";

describe("retained code evidence", () => {
  it("recognizes the Runtime-produced proof while generic JSON stays source", async () => {
    expect(parseCodeEvidence(JSON.stringify(proof))).toEqual(proof);
    await expect(
      createArtifactPreviewPlan("application/json", JSON.stringify(proof)),
    ).resolves.toMatchObject({ kind: "code-evidence", document: proof });
    await expect(
      createArtifactPreviewPlan("application/json", '{"text":"code"}'),
    ).resolves.toEqual({ kind: "source" });
    await expect(
      createArtifactPreviewPlan("text/plain", JSON.stringify(proof)),
    ).resolves.toEqual({ kind: "source" });
  });
  it("renders actual coordinates, saved selection and syntax without changing text", async () => {
    const document = parseCodeEvidence(JSON.stringify(proof))!;
    const { container } = render(<CodeEvidencePreview document={document} />);
    expect(
      screen.getByText("Line 24 selected · saved workspace"),
    ).toBeVisible();
    expect(screen.getByText("Line 22")).toBeVisible();
    expect(container.querySelector("pre")?.textContent).toBe(
      proof.workspace.text,
    );
    expect(container.querySelector('[data-line="16"]')).toBeVisible();
    expect(container.querySelector('[data-selected="true"]')).toHaveAttribute(
      "data-line",
      "24",
    );
    expect(container.querySelectorAll('[data-selected="true"]')).toHaveLength(
      1,
    );
    expect(container.querySelector(".hljs-keyword")).toBeVisible();
    await userEvent.click(screen.getByRole("button", { name: "Wrap lines" }));
    expect(container.querySelector("pre")).toHaveAttribute(
      "data-wrap",
      "false",
    );
    expect(container.querySelector("pre")?.textContent).toBe(
      proof.workspace.text,
    );
  });
  it.each(["\n", "\r\n", "\r"])(
    "preserves %j line endings and escapes hostile/unknown-language text",
    (ending) => {
      const text = `<!-- comment -->${ending}<img src="https://example.invalid/tracker" onerror="alert(1)">${ending}<script>alert(1)</script>${ending}`;
      const document = {
        ...proof,
        workspace: {
          ...proof.workspace,
          file: "example.unknown",
          startLine: 2,
          endLine: 3,
          snippetStartLine: 1,
          snippetEndLine: 3,
          text,
        },
      } as const;
      expect(parseCodeEvidence(JSON.stringify(document))).toEqual(document);
      const { container } = render(
        <CodeEvidencePreview
          document={parseCodeEvidence(JSON.stringify(document))!}
        />,
      );
      expect(container.querySelector("pre")?.textContent).toBe(text);
      expect(container.querySelectorAll("[data-line]")).toHaveLength(3);
      expect(
        container.querySelectorAll("pre img,pre script,pre iframe,pre svg"),
      ).toHaveLength(0);
      expect(container.querySelectorAll("[data-selected]")).toHaveLength(2);
    },
  );
  it("keeps the original JSON in Source and explains invalid declared proofs", async () => {
    const source = JSON.stringify({
      ...proof,
      schema: "contractor.trace-finding-evidence.v2",
    });
    render(
      <LoadedArtifactPreview source={source} mediaType="application/json" />,
    );
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "code evidence is invalid or unsupported",
    );
    await userEvent.click(screen.getByRole("tab", { name: "Source" }));
    expect(
      screen.getByRole("tabpanel", { name: "Source" }).querySelector("pre")
        ?.textContent,
    ).toBe(source);
  });
  it.each([
    { workspace: { ...proof.workspace, startLine: 15 } },
    { workspace: { ...proof.workspace, snippetEndLine: 25 } },
    { workspace: { ...proof.workspace, snippetStartLine: 0 } },
    { workspace: { ...proof.workspace, endLine: Number.MAX_SAFE_INTEGER + 1 } },
    { workspace: { ...proof.workspace, text: "" } },
    {
      source: {
        ...proof.source,
        artifact: { ...proof.source.artifact, revision: "" },
      },
    },
    { source: { ...proof.source, digest: "invalid" } },
    { requestDigest: "invalid" },
    { extra: true },
  ])("rejects incoherent or malformed display data %j", async (change) => {
    const source = JSON.stringify({ ...proof, ...change });
    expect(parseCodeEvidence(source)).toBeUndefined();
    await expect(
      createArtifactPreviewPlan("application/json", source),
    ).resolves.toEqual({ kind: "invalid-code-evidence" });
  });
  it("rejects duplicate keys and nested unknown fields", () => {
    const source = JSON.stringify(proof);
    expect(
      parseCodeEvidence(
        source.replace('"schema":', '"requestDigest":"invalid","schema":'),
      ),
    ).toBeUndefined();
    expect(
      parseCodeEvidence(
        JSON.stringify({
          ...proof,
          workspace: { ...proof.workspace, unexpected: true },
        }),
      ),
    ).toBeUndefined();
  });
  it("admits a maximum-size snippet and bounds syntax DOM expansion without losing text", async () => {
    const lines = Array.from(
      { length: 400 },
      () => 'return "x"; '.repeat(26) + "\n",
    );
    let text = lines.join("");
    text = " ".repeat(128 * 1024 - text.length) + text;
    const value = {
      ...proof,
      workspace: {
        ...proof.workspace,
        snippetStartLine: 100000,
        snippetEndLine: 100399,
        startLine: 100000,
        endLine: 100399,
        text,
      },
    };
    const document = parseCodeEvidence(JSON.stringify(value));
    expect(document).toBeDefined();
    const { container } = render(<CodeEvidencePreview document={document!} />);
    expect(container.querySelector("pre")?.textContent).toBe(text);
    expect(container.querySelectorAll("[data-line]")).toHaveLength(400);
    expect(container.querySelectorAll(".hljs-keyword")).toHaveLength(0);
    expect(container.querySelectorAll("pre span").length).toBeLessThanOrEqual(
      16400,
    );
    expect(
      parseCodeEvidence(
        JSON.stringify({
          ...value,
          workspace: { ...value.workspace, text: "x" + text },
        }),
      ),
    ).toBeUndefined();
    expect(
      parseCodeEvidence(
        JSON.stringify({
          ...value,
          workspace: {
            ...value.workspace,
            snippetEndLine: 100400,
            text: text + "\n",
          },
        }),
      ),
    ).toBeUndefined();
    await waitFor(() =>
      expect(container.querySelector('[data-line="100399"]')).toBeVisible(),
    );
  });
});
