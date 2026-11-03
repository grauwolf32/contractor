import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { MetricBars } from "./chart";

describe("compact observed metric bars", () => {
  it("retains measured zero and restart identities without filling missing observations", () => {
    render(
      <MetricBars
        title="CPU usage"
        from="2026-11-03T12:00:00Z"
        to="2026-11-03T12:01:00Z"
        expectedStepSeconds={15}
        unit="cores"
        data={[
          {
            observedAt: "2026-11-03T12:00:00Z",
            generation: "before",
            value: 0,
          },
          { observedAt: "2026-11-03T12:00:15Z", generation: "before" },
          { observedAt: "2026-11-03T12:00:45Z", generation: "after", value: 2 },
        ]}
      />,
    );
    const chart = screen.getByRole("img", { name: "CPU usage" });
    expect(chart.children).toHaveLength(2);
    expect(chart.children[0]).toHaveAttribute("data-generation", "before");
    expect(chart.children[0]).toHaveStyle({ height: "0%" });
    expect(chart.children[1]).toHaveAttribute("data-generation", "after");
    expect(
      screen.getByLabelText("CPU usage numeric summary"),
    ).toHaveTextContent(
      "Minimum0 coresLatest2 coresMaximum2 coresObserved points2",
    );
  });

  it("shows missing observations explicitly without creating a zero chart", () => {
    render(
      <MetricBars
        title="CPU usage"
        from="2026-11-03T12:00:00Z"
        to="2026-11-03T12:01:00Z"
        expectedStepSeconds={15}
        unit="cores"
        data={[]}
      />,
    );
    expect(screen.getByText("No observations in this range.")).toBeVisible();
    expect(screen.queryByRole("img")).toBeNull();
  });
});
