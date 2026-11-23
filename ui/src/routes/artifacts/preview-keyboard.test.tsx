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

it("keeps a compact saved snippet readable and exposes Retry after a failed read", async () => {
  let reads = 0;
  const view = renderWithAPI(
    <ArtifactPreviewPanel
      compact
      metadata={artifactMetadata("evidence", "source", {
        mediaType: "text/plain",
        size: 5,
      })}
      loadPreview={async () => {
        reads++;
        if (reads === 1) throw new Error("Unavailable source");
        return "saved";
      }}
      loadOnMountKey={["preview", "compact-retry"]}
      unavailableCopy="Download this file"
    />,
  );
  await view.user.click(
    await screen.findByRole("button", { name: "Retry preview" }),
  );
  expect(await screen.findByText("saved", { exact: true })).toBeVisible();
  expect(screen.queryByRole("button", { name: "Reload preview" })).toBeNull();
  expect(screen.queryByText(/Text files up to/)).toBeNull();
});
