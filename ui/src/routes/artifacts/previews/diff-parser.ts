type DiffLineKind =
  "addition" | "context" | "deletion" | "file" | "hunk" | "meta";

interface DiffLine {
  content: string;
  kind: DiffLineKind;
  newLine?: number | undefined;
  oldLine?: number | undefined;
}

interface ParsedDiff {
  additions: number;
  deletions: number;
  files: number;
  lines: DiffLine[];
}

const HUNK_HEADER = /^@@ -([0-9]+)(?:,([0-9]+))? \+([0-9]+)(?:,([0-9]+))? @@/;
const FILE_METADATA =
  /^(?:diff |index |--- |\+\+\+ |new file mode |deleted file mode |old mode |new mode |similarity index |dissimilarity index |rename from |rename to |copy from |copy to |Binary files |Binary path changed:)/;

export function parseUnifiedDiff(source: string): ParsedDiff {
  const sourceLines = source === "" ? [] : source.split("\n");
  if (sourceLines.at(-1) === "") {
    sourceLines.pop();
  }

  let additions = 0;
  let deletions = 0;
  let files = 0;
  let gitFiles = 0;
  let oldLine: number | undefined;
  let newLine: number | undefined;
  let oldLinesRemaining = 0;
  let newLinesRemaining = 0;
  let inHunk = false;

  const finishHunkIfComplete = () => {
    if (oldLinesRemaining === 0 && newLinesRemaining === 0) {
      inHunk = false;
      oldLine = undefined;
      newLine = undefined;
    }
  };

  const lines = sourceLines.map((content): DiffLine => {
    const normalized = content.endsWith("\r") ? content.slice(0, -1) : content;
    const hunk = HUNK_HEADER.exec(normalized);
    if (hunk !== null) {
      oldLine = Number(hunk[1]);
      newLine = Number(hunk[3]);
      oldLinesRemaining = hunk[2] === undefined ? 1 : Number(hunk[2]);
      newLinesRemaining = hunk[4] === undefined ? 1 : Number(hunk[4]);
      inHunk = true;
      finishHunkIfComplete();
      return { content: normalized, kind: "hunk" };
    }
    if (!inHunk && FILE_METADATA.test(normalized)) {
      if (normalized.startsWith("diff ")) {
        gitFiles += 1;
      } else if (normalized.startsWith("--- ")) {
        files += 1;
      }
      return { content: normalized, kind: "file" };
    }
    if (normalized.startsWith("+")) {
      const line = { content: normalized, kind: "addition", newLine } as const;
      additions += 1;
      if (inHunk && newLine !== undefined) {
        newLine += 1;
        newLinesRemaining = Math.max(0, newLinesRemaining - 1);
        finishHunkIfComplete();
      }
      return line;
    }
    if (normalized.startsWith("-")) {
      const line = { content: normalized, kind: "deletion", oldLine } as const;
      deletions += 1;
      if (inHunk && oldLine !== undefined) {
        oldLine += 1;
        oldLinesRemaining = Math.max(0, oldLinesRemaining - 1);
        finishHunkIfComplete();
      }
      return line;
    }
    if (normalized.startsWith("\\")) {
      return { content: normalized, kind: "meta" };
    }

    const line = {
      content: normalized,
      kind: "context",
      oldLine,
      newLine,
    } as const;
    if (inHunk && oldLine !== undefined && newLine !== undefined) {
      oldLine += 1;
      newLine += 1;
      oldLinesRemaining = Math.max(0, oldLinesRemaining - 1);
      newLinesRemaining = Math.max(0, newLinesRemaining - 1);
      finishHunkIfComplete();
    }
    return line;
  });

  return {
    additions,
    deletions,
    files: Math.max(files, gitFiles),
    lines,
  };
}
