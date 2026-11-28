import { common, createLowlight } from "lowlight";
import { type ReactNode } from "react";
import { sourceLines } from "../source-lines";

const highlighter = createLowlight(common);
const LANGUAGES: Record<string, string> = {
  js: "javascript",
  jsx: "javascript",
  mjs: "javascript",
  cjs: "javascript",
  ts: "typescript",
  tsx: "typescript",
  py: "python",
  go: "go",
  java: "java",
  c: "c",
  h: "c",
  cc: "cpp",
  cpp: "cpp",
  hpp: "cpp",
  cs: "csharp",
  rb: "ruby",
  php: "php",
  rs: "rust",
  kt: "kotlin",
  kts: "kotlin",
  swift: "swift",
  sh: "bash",
  bash: "bash",
  sql: "sql",
  html: "xml",
  xml: "xml",
  css: "css",
  json: "json",
  yaml: "yaml",
  yml: "yaml",
};

/** Explicit file extension only: never auto-detect syntax or insert HTML from source. */
export function highlightCode(
  source: string,
  file: string,
): ReactNode[] | undefined {
  const language = LANGUAGES[file.split(".").at(-1)?.toLowerCase() ?? ""];
  if (!language || new TextEncoder().encode(source).length > 128 * 1024)
    return undefined;
  const lines = sourceLines(source, 400);
  if (!lines) return undefined;
  try {
    const tree = highlighter.highlight(language, source);
    type Node = (typeof tree.children)[number];
    const tokens: { text: string; className: string }[] = [];
    // LIM-UI-25: bound DOM expansion independently from the source byte bound.
    let budget = 16000;
    function collect(node: Node, classes: string, depth: number) {
      if (--budget < 0 || depth > 64)
        throw new Error("Syntax display budget exceeded");
      if (node.type === "text")
        tokens.push({ text: node.value, className: classes });
      else if (node.type === "element") {
        const names = node.properties.className;
        const next =
          `${classes} ${Array.isArray(names) ? names.join(" ") : ""}`.trim();
        node.children.forEach((child) => collect(child, next, depth + 1));
      }
    }
    tree.children.forEach((node) => collect(node, "", 0));
    if (tokens.map((token) => token.text).join("") !== source) return undefined;
    let tokenIndex = 0,
      offset = 0;
    function take(length: number, render: boolean): ReactNode[] {
      const result: ReactNode[] = [];
      while (length > 0) {
        const token = tokens[tokenIndex]!;
        const count = Math.min(length, token.text.length - offset);
        if (render && count) {
          if (--budget < 0) throw new Error("Syntax display budget exceeded");
          const text = token.text.slice(offset, offset + count);
          result.push(
            token.className ? (
              <span key={result.length} className={token.className}>
                {text}
              </span>
            ) : (
              text
            ),
          );
        }
        offset += count;
        length -= count;
        if (offset === token.text.length) {
          tokenIndex++;
          offset = 0;
        }
      }
      return result;
    }
    return lines.map((line) => {
      const content = take(line.text.length, true);
      take(line.ending.length, false);
      return content;
    });
  } catch {
    return undefined;
  }
}
