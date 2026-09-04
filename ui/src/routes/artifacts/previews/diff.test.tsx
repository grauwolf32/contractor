import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import DiffArtifactPreview from "./diff";
import { parseUnifiedDiff } from "./diff-parser";

const source = [
  "--- a/src/config.ts",
  "+++ b/src/config.ts",
  "@@ -1,3 +1,4 @@",
  " export const mode = {",
  "-  preview: false,",
  "+  preview: true,",
  "+  color: true,",
  " };",
  "\\ No newline at end of file",
].join("\n");

describe("Diff Artifact preview", () => {
  it("classifies unified diff lines and tracks both line numbers", () => {
    const parsed = parseUnifiedDiff(source);

    expect(parsed).toMatchObject({ additions: 2, deletions: 1, files: 1 });
    expect(
      parsed.lines.map(({ kind, oldLine, newLine }) => ({
        kind,
        oldLine,
        newLine,
      })),
    ).toEqual([
      { kind: "file", oldLine: undefined, newLine: undefined },
      { kind: "file", oldLine: undefined, newLine: undefined },
      { kind: "hunk", oldLine: undefined, newLine: undefined },
      { kind: "context", oldLine: 1, newLine: 1 },
      { kind: "deletion", oldLine: 2, newLine: undefined },
      { kind: "addition", oldLine: undefined, newLine: 2 },
      { kind: "addition", oldLine: undefined, newLine: 3 },
      { kind: "context", oldLine: 3, newLine: 4 },
      { kind: "meta", oldLine: undefined, newLine: undefined },
    ]);
  });

  it("renders escaped, color-addressable rows and a change summary", () => {
    const view = render(<DiffArtifactPreview source={`${source}\n`} />);

    expect(screen.getByLabelText("Unified diff preview")).toBeInTheDocument();
    expect(screen.getByText("1 file changed")).toBeInTheDocument();
    expect(screen.getByText("+2 additions")).toBeInTheDocument();
    expect(screen.getByText("−1 deletion")).toBeInTheDocument();
    expect(view.container.querySelectorAll(".diff-line-addition")).toHaveLength(
      2,
    );
    expect(view.container.querySelectorAll(".diff-line-deletion")).toHaveLength(
      1,
    );
  });

  it("does not mistake changed content for the next file header", () => {
    const parsed = parseUnifiedDiff(
      [
        "--- a/markers.txt",
        "+++ b/markers.txt",
        "@@ -1,2 +1,2 @@",
        "--- old marker",
        "+++ new marker",
        " unchanged",
      ].join("\n"),
    );

    expect(parsed).toMatchObject({ additions: 1, deletions: 1, files: 1 });
    expect(parsed.lines.map((line) => line.kind)).toEqual([
      "file",
      "file",
      "hunk",
      "deletion",
      "addition",
      "context",
    ]);
  });

  it("shows an explicit empty state", () => {
    render(<DiffArtifactPreview source="" />);

    expect(screen.getByText("No changes in this diff.")).toBeInTheDocument();
  });
});
