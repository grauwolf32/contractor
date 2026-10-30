import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { renderWithAPI } from "../../test/render";
import { artifactMetadata } from "../../test/builders";
import { ArtifactPreviewPanel } from "./preview";
import { AUTOMATIC_OPENAPI_LIMIT_LABEL } from "./preview-plan";

// LIM-UI-17 in docs/spec/platform/limits.md.
const MAXIMUM_AUTOMATIC_OPENAPI_BYTES = 128 * 1024;

function renderLarge(source: string) {
  return renderWithAPI(
    <ArtifactPreviewPanel
      metadata={artifactMetadata("reports", "api", {
        mediaType: "application/yaml",
        size: new TextEncoder().encode(source).length,
        frozen: true,
      })}
      loadPreview={async () => source}
      loadOnMountKey={["preview", "openapi-on-request"]}
      unavailableCopy="Download this file"
    />,
  );
}

describe("OpenAPI preview above LIM-UI-17", () => {
  it("opens on Source and parses the document only on request", async () => {
    // Declares OpenAPI 3.1 at the top level but is not a readable document.
    const source = `openapi: 3.1.0\npaths: [\n# ${"x".repeat(MAXIMUM_AUTOMATIC_OPENAPI_BYTES)}`;
    const { user } = renderLarge(source);

    const sourceTab = await screen.findByRole("tab", { name: "Source" });
    expect(sourceTab).toHaveAttribute("aria-selected", "true");
    const rendered = screen.getByRole("tab", { name: "Rendered" });
    expect(rendered).toHaveAttribute("aria-selected", "false");
    await user.click(rendered);
    expect(
      screen.getByText(
        new RegExp(
          `larger than ${AUTOMATIC_OPENAPI_LIMIT_LABEL}, so it is not rendered automatically`,
        ),
      ),
    ).toBeVisible();
    expect(AUTOMATIC_OPENAPI_LIMIT_LABEL).toBe("128 KiB");

    await user.click(screen.getByRole("button", { name: "Render document" }));
    expect(
      await screen.findByText(/This document is not a valid OpenAPI document/),
    ).toBeVisible();
    expect(
      screen.queryByRole("button", { name: "Render document" }),
    ).toBeNull();
  });
});
