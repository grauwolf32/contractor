import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { ArtifactPreviewPanel } from "./preview";

describe("Artifact preview keyboard navigation", () => {
  it("moves selection and focus with arrows, Home and End, then Tabs into the panel", async () => {
    const client = new QueryClient();
    render(
      <QueryClientProvider client={client}>
        <ArtifactPreviewPanel
          metadata={{
            artifact: { namespace: "reports", name: "result", revision: "r1" },
            mediaType: "text/markdown",
            size: 10,
            current: true,
            frozen: true,
            createdAt: "2026-09-01T10:00:00Z",
          }}
          loadPreview={async () => "# Result\nEvidence"}
          loadOnMountKey={["preview", "keyboard"]}
          unavailableCopy="Download this file"
        />
      </QueryClientProvider>,
    );
    const user = userEvent.setup();
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
