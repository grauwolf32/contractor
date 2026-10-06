import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { STATUS_TONES } from "../app/status-tone";
import { ProgressSegments, StatusChip, StatusGlyph } from "./status";

describe("StatusGlyph", () => {
  it("is decorative without a label", () => {
    const { container, rerender } = render(<StatusGlyph tone="partial" />);
    const svg = container.querySelector("svg");
    expect(svg).toHaveAttribute("aria-hidden", "true");
    expect(svg).toHaveAttribute("width", "16");
    expect(screen.queryByRole("img")).toBeNull();
    rerender(<StatusGlyph tone="partial" label="" />);
    expect(container.querySelector("svg")).toHaveAttribute(
      "aria-hidden",
      "true",
    );
  });

  it("is an image named by its label", () => {
    render(<StatusGlyph tone="blocked" label="Blocked" size={20} />);
    const glyph = screen.getByRole("img", { name: "Blocked" });
    expect(glyph).not.toHaveAttribute("aria-hidden");
    expect(glyph).toHaveAttribute("data-tone", "blocked");
    expect(glyph).toHaveAttribute("height", "20");
  });

  it("draws a shape for every tone", () => {
    for (const tone of STATUS_TONES) {
      const { container, unmount } = render(<StatusGlyph tone={tone} />);
      expect(container.querySelector("svg")?.childElementCount).toBeGreaterThan(
        0,
      );
      unmount();
    }
  });
});

describe("StatusChip", () => {
  it("shows the word with a decorative glyph", () => {
    const { container } = render(
      <StatusChip tone="review">Needs review</StatusChip>,
    );
    const chip = screen.getByText("Needs review");
    expect(chip).toHaveAttribute("data-tone", "review");
    expect(chip).toHaveAttribute("data-size", "md");
    expect(container.querySelector("svg")).toHaveAttribute(
      "aria-hidden",
      "true",
    );
  });

  it("can drop the glyph", () => {
    const { container } = render(
      <StatusChip tone="done" glyph={false} size="sm">
        Met
      </StatusChip>,
    );
    expect(container.querySelector("svg")).toBeNull();
    expect(screen.getByText("Met")).toHaveAttribute("data-size", "sm");
  });
});

describe("ProgressSegments", () => {
  it("is one labelled image with a bar per item in order", () => {
    render(
      <ProgressSegments
        label="0 of 3 done: 1 partially traced, 1 blocked, 1 not checked yet"
        segments={[
          { tone: "partial", label: "Partially traced" },
          { tone: "blocked", label: "Blocked" },
          { tone: "idle", label: "Not checked yet" },
        ]}
      />,
    );
    const line = screen.getByRole("img", {
      name: "0 of 3 done: 1 partially traced, 1 blocked, 1 not checked yet",
    });
    const bars = Array.from(line.children);
    expect(bars.map((bar) => bar.getAttribute("data-tone"))).toEqual([
      "partial",
      "blocked",
      "idle",
    ]);
    expect(bars[1]).toHaveAttribute("title", "Blocked");
  });

  it("tightens the gaps for long lists", () => {
    const segments = Array.from({ length: 90 }, () => ({
      tone: "idle" as const,
      label: "Not checked yet",
    }));
    render(<ProgressSegments label="90 items" segments={segments} size="sm" />);
    const line = screen.getByRole("img", { name: "90 items" });
    expect(line).toHaveAttribute("data-density", "dense");
    expect(line).toHaveAttribute("data-size", "sm");
  });
});
