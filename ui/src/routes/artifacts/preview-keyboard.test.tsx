import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { renderWithAPI } from "../../test/render";
import { ArtifactPreviewPanel } from "./preview";
import { artifactMetadata } from "../../test/builders";

describe("Artifact preview keyboard navigation", () => {
  it("moves selection and focus with arrows, Home and End, then Tabs into the panel", async () => {
    const { user } = renderWithAPI(
      <ArtifactPreviewPanel
        metadata={artifactMetadata("reports", "result", {
          mediaType: "text/markdown",
          size: 10,
          frozen: true,
        })}
        loadPreview={async () => "# Result\nEvidence"}
        loadOnMountKey={["preview", "keyboard"]}
        unavailableCopy="Download this file"
      />,
    );
    const rendered = await screen.findByRole("tab", { name: "Rendered" });
    const source = screen.getByRole("tab", { name: "Source" });
    rendered.focus();
    await user.keyboard("{ArrowRight}");
    expect(source).toHaveFocus();
    expect(source).toHaveAttribute("aria-selected", "true");
    expect(rendered).toHaveAttribute("tabindex", "-1");
    await user.tab();
    expect(screen.getByRole("tabpanel", { name: "Source" })).toHaveFocus();
    source.focus();
    await user.keyboard("{Home}");
    expect(rendered).toHaveFocus();
    await user.keyboard("{ArrowLeft}");
    expect(source).toHaveFocus();
    await user.keyboard("{Home}{End}");
    expect(source).toHaveFocus();
  });
});
