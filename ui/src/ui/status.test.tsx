import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { STATUS_TONES } from "../app/status-tone";
import {
  ProgressSegments,
  StatusChip,
  StatusGlyph,
  type ProgressSegment,
} from "./status";

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

  it("tightens the gaps above 40 items", () => {
    const segments = Array.from({ length: 60 }, () => ({
      tone: "idle" as const,
      label: "Not checked yet",
    }));
    render(<ProgressSegments label="60 items" segments={segments} size="sm" />);
    const line = screen.getByRole("img", { name: "60 items" });
    expect(line).toHaveAttribute("data-density", "compact");
    expect(line).toHaveAttribute("data-size", "sm");
    expect(line.children).toHaveLength(60);
  });

  it("merges runs of one tone above 60 items, keeping order and shares", () => {
    const run = (
      count: number,
      tone: ProgressSegment["tone"],
      label: string,
    ): ProgressSegment[] =>
      Array.from({ length: count }, () => ({ tone, label }));
    const segments = [
      ...run(110, "done", "Met"),
      ...run(2, "done", "Fully traced"),
      ...run(30, "blocked", "Issue found"),
      ...run(1, "partial", "Inconclusive"),
      ...run(137, "idle", "Not checked yet"),
    ];
    const label = "112 of 280 requirements met: 30 issues found, …";
    render(<ProgressSegments label={label} segments={segments} />);
    const line = screen.getByRole("img", { name: label });
    expect(line).toHaveAttribute("data-density", "runs");
    const bars = Array.from(line.children) as HTMLElement[];
    expect(
      bars.map((bar) => [
        bar.getAttribute("data-tone"),
        bar.style.flexGrow,
        bar.getAttribute("title"),
      ]),
    ).toEqual([
      ["done", "112", "Met (110), Fully traced (2)"],
      ["blocked", "30", "Issue found (30)"],
      ["partial", "1", "Inconclusive"],
      ["idle", "137", "Not checked yet (137)"],
    ]);
  });

  it("names at most three labels in a merged bar's title", () => {
    const segments: ProgressSegment[] = Array.from({ length: 61 }, (_, n) => ({
      tone: "done",
      label: `GET /api/${n % 5}`,
    }));
    render(<ProgressSegments label="61 endpoints met" segments={segments} />);
    const [bar] = Array.from(
      screen.getByRole("img", { name: "61 endpoints met" }).children,
    );
    expect(bar).toHaveAttribute(
      "title",
      "GET /api/0 (13), GET /api/1 (12), GET /api/2 (12) and 2 more",
    );
  });
});
