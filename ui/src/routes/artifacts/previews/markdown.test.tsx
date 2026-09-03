import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import MarkdownArtifactPreview from "./markdown";

describe("Markdown Artifact preview", () => {
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
