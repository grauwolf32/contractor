import { render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { RecordedTime } from "./recorded-time";
import { formatTimestamp } from "../routes/artifacts/common";

describe("RecordedTime", () => {
  beforeEach(() => {
    vi.useFakeTimers({ toFake: ["Date", "setInterval", "clearInterval"] });
    vi.setSystemTime(new Date("2026-09-22T12:00:00Z"));
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it("renders a relative time element that carries the absolute value", () => {
    const value = "2026-09-22T09:30:00Z";
    const view = render(<RecordedTime value={value} className="stamp" />);
    const element = view.container.querySelector("time");
    expect(element).not.toBeNull();
    expect(element).toHaveAttribute("datetime", value);
    expect(element).toHaveClass("stamp");
    expect(element).toHaveTextContent("2 hours ago");
    expect(element?.getAttribute("title")).toContain(formatTimestamp(value));
    expect(screen.getByTitle(new RegExp(formatTimestamp(value)))).toBe(element);
  });

  it("picks the unit from the elapsed time and reports future moments", () => {
    const { rerender, container } = render(
      <RecordedTime value="2026-09-22T11:59:30Z" />,
    );
    expect(container.querySelector("time")).toHaveTextContent("this minute");
    rerender(<RecordedTime value="2026-09-15T12:00:00Z" />);
    expect(container.querySelector("time")).toHaveTextContent("7 days ago");
    rerender(<RecordedTime value="2026-09-22T15:00:00Z" />);
    expect(container.querySelector("time")).toHaveTextContent("in 3 hours");
  });

  it("falls back to a label for values that are not dates", () => {
    render(<RecordedTime value="not-a-date" />);
    expect(screen.getByText("Unknown date")).toHaveAttribute(
      "datetime",
      "not-a-date",
    );
  });
});
