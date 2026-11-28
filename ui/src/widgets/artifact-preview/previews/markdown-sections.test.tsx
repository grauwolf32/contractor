import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import MarkdownArtifactPreview from "./markdown";
import {
  MAXIMUM_RENDERED_SECTION_CHARACTERS,
  splitMarkdownSections,
} from "./markdown-sections";

const WORDS = "word ".repeat(3400);
// One paragraph longer than a section, so every safe start around it becomes
// a section boundary and each case shows exactly where the splitter cuts.
const F = `Filler ${WORDS}\n\n`;
// A trailing definition keeps the body in one section and renders nothing,
// so it gives the whole-document render to compare with.
const WHOLE = "\n\n[whole-document-render]: /unused\n";

async function renderedHTML(source: string): Promise<string> {
  const view = render(<MarkdownArtifactPreview source={source} />);
  await waitFor(() => expect(screen.queryByRole("status")).toBeNull());
  const html = view.container.querySelector("article")?.innerHTML ?? "";
  view.unmount();
  return html;
}

async function expectSameAsWhole(body: string): Promise<void> {
  expect(splitMarkdownSections(body + WHOLE)).toHaveLength(1);
  expect(await renderedHTML(body)).toBe(await renderedHTML(body + WHOLE));
}

/** The first line of each section, cut to ten characters. */
function sectionStarts(body: string): string[] {
  return splitMarkdownSections(body).map((section) =>
    (section.split(/\r?\n/, 1)[0] ?? "").slice(0, 10),
  );
}

const FILLER = "Filler wor";

const cases: { name: string; body: string; starts: string[] }[] = [
  {
    name: "a heading",
    body: `${F}## Findings\n\n${F}`,
    starts: [FILLER, "## Finding", FILLER],
  },
  {
    name: "list items after blank lines",
    body: `${F}- one\n\n- two ${WORDS}\n\n* three ${WORDS}\n\n1. four ${WORDS}\n\n2) five ${WORDS}\n\n-\n\n${F}`,
    starts: [FILLER, FILLER],
  },
  {
    name: "lines that only look like list markers",
    body: `${F}**Bold** ${WORDS}\n\n---\n\n${F}***\n\n${F}10x ${WORDS}\n\n${F}`,
    starts: [
      FILLER,
      "**Bold** w",
      "---",
      FILLER,
      "***",
      FILLER,
      "10x word w",
      FILLER,
    ],
  },
  {
    name: "block quotes",
    body: `${F}> quote\n>\n> more\n\n${F}> second\n\n${F}`,
    starts: [FILLER, "> quote", FILLER, "> second", FILLER],
  },
  {
    name: "indented code across a blank line",
    body: `${F}    code\n\n    more code\n\n${F}`,
    starts: [FILLER, FILLER],
  },
  {
    name: "a heading inside a fence",
    body: `${F}\`\`\`text\n${F}# Not a heading\n\`\`\`\n\n${F}`,
    starts: [FILLER, "```text", FILLER],
  },
  {
    name: "a shorter closing fence",
    body: `${F}~~~~\n${F}# Code\n~~~\n\n# Code\n~~~~\n\n${F}`,
    starts: [FILLER, "~~~~", FILLER],
  },
  {
    name: "a backtick in a backtick info string",
    body: `${F}\`\`\`a\`b\n\n${F}\`\`\`\n${F}# Code\n\`\`\`\n\n${F}`,
    starts: [FILLER, "```a`b", FILLER, "```", FILLER],
  },
  {
    name: "a fence in an ordered list item",
    body: `${F}1. Run\n   \`\`\`sh\n   make\n\n   make lint\n   \`\`\`\n\n# After\n\n${F}`,
    starts: [FILLER, "# After", FILLER],
  },
  {
    name: "a fence after a table",
    body: `${F}| a | b |\n| - | - |\n| 1 | 2 |\n\`\`\`\n${F}# Code\n\`\`\`\n\n${F}`,
    starts: [FILLER, "| a | b |", FILLER],
  },
  {
    name: "an HTML comment",
    body: `${F}<!--\n${F}# Hidden\n\n-->\n\n${F}`,
    starts: [FILLER, "<!--", FILLER],
  },
  {
    name: "a script block",
    body: `${F}<SCRIPT>\n${F}# Hidden\n\n</script>\n\n${F}`,
    starts: [FILLER, "<SCRIPT>", FILLER],
  },
  {
    name: "processing instructions, declarations and CDATA",
    body: `${F}<?php\n\n# Hidden\n?>\n\n${F}<!DOCTYPE\n\n# Hidden\n>\n\n${F}<![CDATA[\n\n# Hidden\n]]>\n\n${F}<!-- one line -->\n\n${F}`,
    starts: [
      FILLER,
      "<?php",
      FILLER,
      "<!DOCTYPE",
      FILLER,
      "<![CDATA[",
      FILLER,
      "<!-- one l",
      FILLER,
    ],
  },
  {
    name: "an HTML block that ends at a blank line",
    body: `${F}<div>\n<b>raw</b>\n</div>\n\n# After\n\n${F}`,
    starts: [FILLER, "<div>", FILLER],
  },
  {
    name: "Windows line endings",
    body: `${F}# Title\r\n\r\nText\r\n\`\`\`\r\n${F.replaceAll("\n", "\r\n")}# Code\r\n\`\`\`\r\n\r\n${F}`,
    starts: [FILLER, "# Title", "Text", FILLER],
  },
];

const unsplittable: { name: string; construct: string }[] = [
  {
    name: "a fence-like line in a possible HTML block",
    construct: "<div>\n```\n</div>\n\n# Heading\n\n```\n\n# Code\n```\n\n",
  },
  {
    name: "a fence-like line after an inline tag in a paragraph",
    construct: "Text\n<b>bold</b>\n```\n\n# Code\n```\n\n",
  },
  {
    name: "an indented fence that a less indented line may close",
    construct:
      "- item\n  ```\n  code\n\n# Heading\n\n```\ncode\n\n# Code\n```\n\n",
  },
  {
    name: "an indented HTML block that a less indented line may close",
    construct: "- item\n  <!--\n\n# Heading\n\n<!--\n\n# Hidden\n-->\n\n",
  },
  {
    name: "a link reference definition",
    construct:
      "See [the guide][guide].\n\n[guide]: https://example.com/guide\n\n",
  },
  {
    name: "a footnote",
    construct: "A claim.[^source]\n\n[^source]: The evidence.\n\n",
  },
];

describe("Markdown sections", () => {
  for (const { name, body, starts } of cases) {
    it(`splits around ${name} as the whole document renders`, async () => {
      expect(splitMarkdownSections(body).join("")).toBe(body);
      expect(sectionStarts(body)).toEqual(starts);
      await expectSameAsWhole(body);
    });
  }

  for (const { name, construct } of unsplittable) {
    it(`keeps a body with ${name} whole`, async () => {
      const body = F + construct + F;
      expect(splitMarkdownSections(body)).toEqual([body]);
      await expectSameAsWhole(body);
    });
  }

  it("bounds every section of a 1 MiB report", () => {
    let body = "";
    for (let index = 0; body.length < 1024 * 1024; index++) {
      body += `## Section ${index}\n\nA paragraph with \`code\` and **bold** text.\n\n- one\n- two\n\n`;
    }
    const parts = splitMarkdownSections(body);
    expect(parts.join("")).toBe(body);
    expect(parts.length).toBeGreaterThan(60);
    for (const part of parts) {
      expect(part.length).toBeLessThanOrEqual(16 * 1024);
    }
  });

  it("resolves a reference to a definition far below it", async () => {
    render(
      <MarkdownArtifactPreview
        source={`${F}See [the guide][guide].\n\n${F}[guide]: https://example.com/guide\n`}
      />,
    );
    expect(
      await screen.findByRole("link", { name: "the guide" }),
    ).toHaveAttribute("href", "https://example.com/guide");
  });

  it("shows the first section at once and the rest in later steps", async () => {
    const body = Array.from(
      { length: 8 },
      (_, index) => `## Part ${index}\n\n${F}`,
    ).join("");
    render(<MarkdownArtifactPreview source={body} />);
    expect(screen.getByRole("heading", { name: "Part 0" })).toBeVisible();
    expect(screen.queryByRole("heading", { name: "Part 7" })).toBeNull();
    expect(screen.getByRole("status")).toHaveTextContent(
      "Rendering the rest of the document…",
    );
    expect(
      await screen.findByRole("heading", { name: "Part 7" }),
    ).toBeVisible();
    await waitFor(() => expect(screen.queryByRole("status")).toBeNull());
  });

  it("starts again from the first section when the document changes", async () => {
    const document = (title: string) =>
      Array.from(
        { length: 8 },
        (_, index) => `## ${title} ${index}\n\n${F}`,
      ).join("");
    const view = render(
      <MarkdownArtifactPreview source={document("Before")} />,
    );
    expect(
      await screen.findByRole("heading", { name: "Before 7" }),
    ).toBeVisible();
    view.rerender(<MarkdownArtifactPreview source={document("After")} />);
    expect(screen.getByRole("heading", { name: "After 0" })).toBeVisible();
    expect(screen.queryByRole("heading", { name: "After 7" })).toBeNull();
    expect(screen.queryByRole("heading", { name: /Before/ })).toBeNull();
    expect(
      await screen.findByRole("heading", { name: "After 7" }),
    ).toBeVisible();
  });

  it("shows one block above the bound as text without parsing it", async () => {
    let table = "| Endpoint | Method |\n| --- | --- |\n";
    while (table.length <= MAXIMUM_RENDERED_SECTION_CHARACTERS) {
      table += "| `/v1/items/{id}` | **GET** |\n";
    }
    const view = render(
      <MarkdownArtifactPreview
        source={`# Inventory\n\n${table}\n## After\n\nDone.\n`}
      />,
    );
    expect(await screen.findByRole("heading", { name: "After" })).toBeVisible();
    expect(screen.getByRole("heading", { name: "Inventory" })).toBeVisible();
    expect(
      screen.getByText(/too long to render, so it is shown as text/),
    ).toHaveTextContent("more than 262,144 characters");
    expect(view.container.querySelector("table")).toBeNull();
    expect(view.container.querySelector("pre")).toHaveTextContent(
      "| Endpoint | Method |",
    );
  });
});
