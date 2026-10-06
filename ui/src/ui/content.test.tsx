import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { ActivityLog, EmptyState, TechnicalDetails } from "./content";
import { clockTime } from "./format";

describe("ActivityLog", () => {
  it("shows local HH:MM, glyphs for timed entries and dots for steps", () => {
    render(
      <ActivityLog
        aria-label="Activity on this endpoint"
        entries={[
          {
            id: "stopped",
            time: new Date(2026, 9, 4, 23, 56),
            tone: "partial",
            title: "Stopped with a partial trace.",
            text: "The identity service is outside this code.",
          },
          { id: "step", text: "Compared sibling views." },
          { id: "live", time: "now", tone: "progress", text: "Tracing." },
          { id: "start", time: new Date(2026, 9, 4, 7, 5), text: "Started." },
        ]}
      />,
    );
    const log = screen.getByRole("list", { name: "Activity on this endpoint" });
    const [stopped, step, live, start] = within(log).getAllByRole("listitem");
    expect(stopped).toHaveTextContent(
      "23:56Stopped with a partial trace. The identity service is outside this code.",
    );
    expect(within(stopped!).getByText("23:56").tagName).toBe("TIME");
    expect(stopped!.querySelector("svg")).toHaveAttribute(
      "data-tone",
      "partial",
    );
    expect(step!.querySelector("svg")).toBeNull();
    expect(step!.querySelector(".ui-activity-dot")).not.toBeNull();
    expect(live).toHaveTextContent("nowTracing.");
    expect(start).toHaveTextContent("07:05Started.");
    expect(start!.querySelector("svg")).toHaveAttribute("data-tone", "neutral");
  });

  it("formats times and keeps free text", () => {
    expect(clockTime(undefined)).toBeUndefined();
    expect(clockTime(new Date(Number.NaN))).toBeUndefined();
    expect(clockTime("Today")).toEqual({ label: "Today" });
    const time = clockTime("2026-10-04T23:56:00Z");
    expect(time?.dateTime).toBe("2026-10-04T23:56:00.000Z");
    expect(time?.label).toMatch(/^\d\d:\d\d$/);
  });
});

describe("TechnicalDetails", () => {
  it("is a closed disclosure with a quiet summary by default", async () => {
    const { container } = render(
      <TechnicalDetails description="Steps, limits and AI usage, for admins and debugging.">
        <p>Round 3</p>
      </TechnicalDetails>,
    );
    const details = container.querySelector("details");
    expect(details).not.toHaveAttribute("open");
    expect(screen.getByText("Technical details")).toBeInTheDocument();
    expect(
      screen.getByText("Steps, limits and AI usage, for admins and debugging."),
    ).toBeInTheDocument();
    await userEvent.setup().click(screen.getByText("Technical details"));
    expect(details).toHaveAttribute("open");
  });

  it("can start open with another summary", () => {
    const { container } = render(
      <TechnicalDetails summary="Run details" defaultOpen>
        <p>Digest</p>
      </TechnicalDetails>,
    );
    expect(container.querySelector("details")).toHaveAttribute("open");
    expect(screen.getByText("Run details")).toBeInTheDocument();
  });
});

describe("EmptyState", () => {
  it("renders title, explanation and action", () => {
    render(
      <EmptyState
        title="Nothing needs you"
        action={<button type="button">Start a check</button>}
      >
        Possible issues arrive here as checks find them.
      </EmptyState>,
    );
    expect(screen.getByText("Nothing needs you")).toBeInTheDocument();
    expect(
      screen.getByText("Possible issues arrive here as checks find them."),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Start a check" }),
    ).toBeInTheDocument();
  });
});
