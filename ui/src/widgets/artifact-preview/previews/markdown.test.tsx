import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import MarkdownArtifactPreview from "./markdown";
import { splitFrontmatter } from "./frontmatter";

describe("Markdown Artifact preview", () => {
  it("presents Skill frontmatter as bounded text and preserves the document source", () => {
    const source =
      '---\nname: review\ndescription: "<img src=x onerror=alert(1)>"\nlicense: MIT\n---\n# Instructions\nRead the files.';
    const parsed = splitFrontmatter(source);
    expect(parsed.fields.name).toBe("review");
    expect(parsed.body).toBe("# Instructions\nRead the files.");
    expect(source.startsWith("---\nname:")).toBe(true);
    const view = render(<MarkdownArtifactPreview source={source} />);
    expect(screen.getByRole("heading", { name: "Instructions" })).toBeVisible();
    expect(screen.getByText("<img src=x onerror=alert(1)>")).toBeVisible();
    expect(view.container.querySelector("img")).toBeNull();
    expect(view.container.querySelectorAll("h2")).toHaveLength(0);
  });

  it("rejects aliases and duplicate metadata, leaves ordinary Markdown and unbounded headers intact", () => {
    for (const header of [
      "name: &name review\ndescription: *name",
      "name: one\nname: two",
    ]) {
      expect(
        splitFrontmatter(`---\n${header}\n---\n# Instructions`),
      ).toMatchObject({ fields: {}, invalid: true, body: "# Instructions" });
    }
    for (const source of [
      "---\nNormal text\n---\nMore text",
      `---\nname: ${"x".repeat(17000)}\n---\nBody`,
    ]) {
      expect(splitFrontmatter(source)).toEqual({ body: source, fields: {} });
    }
  });
  it("renders Markdown without executing HTML or loading images", () => {
    const view = render(
      <MarkdownArtifactPreview
        source={[
          "# Report",
          "",
          "**Ready**",
          "",
          "| Check | Result |",
          "| --- | --- |",
          "| Render | Passed |",
          "",
          "![tracking](https://example.invalid/pixel.png)",
          "",
          "<script>window.compromised = true</script>",
        ].join("\n")}
      />,
    );

    expect(screen.getByRole("heading", { name: "Report" })).toBeInTheDocument();
    expect(
      screen.getByText("Ready", { selector: "strong" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("table")).toBeInTheDocument();
    expect(screen.getByText("Image omitted: tracking")).toBeInTheDocument();
    expect(view.container.querySelector("img")).toBeNull();
    expect(view.container.querySelector("script")).toBeNull();
  });
});
