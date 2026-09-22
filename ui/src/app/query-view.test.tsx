import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { QueryView } from "./query-view";

function view(data: string | undefined, error: Error | null) {
  const onRetry = vi.fn();
  render(
    <QueryView
      query={{ data, error, isFetching: false }}
      loading={<p>Loading…</p>}
      errorContext="Could not load"
      onRetry={onRetry}
    >
      {(value) => <p>Loaded {value}</p>}
    </QueryView>,
  );
  return onRetry;
}

describe("QueryView", () => {
  it("shows loading until data or an error arrives", () => {
    view(undefined, null);
    expect(screen.getByText("Loading…")).toBeInTheDocument();
  });

  it("replaces content with the error panel only when nothing is loaded", () => {
    view(undefined, new Error("Server unavailable"));
    expect(screen.getByRole("alert")).toHaveTextContent("Could not load");
    expect(screen.queryByText(/Loaded/)).not.toBeInTheDocument();
  });

  it("keeps loaded data visible with a stale warning after a failed refetch", async () => {
    const onRetry = view("value", new Error("Server unavailable"));
    expect(screen.getByText("Loaded value")).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent(
      "showing the last loaded data",
    );
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    await userEvent
      .setup()
      .click(screen.getByRole("button", { name: "Try again" }));
    expect(onRetry).toHaveBeenCalledOnce();
  });
});
