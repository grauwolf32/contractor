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

  it("renders the empty state for loaded data selected as empty", () => {
    render(
      <QueryView
        query={{ data: [] as string[], error: null, isFetching: false }}
        loading={<p>Loading…</p>}
        onRetry={vi.fn()}
        isEmpty={(items) => items.length === 0}
        empty={<p>Nothing yet</p>}
      >
        {(items) => <p>Loaded {items.length}</p>}
      </QueryView>,
    );
    expect(screen.getByText("Nothing yet")).toBeInTheDocument();
    expect(screen.queryByText(/Loaded/)).not.toBeInTheDocument();
  });

  it("reads a query result directly: default loading text and its own refetch", async () => {
    const refetch = vi.fn();
    const { rerender } = render(
      <QueryView
        query={{ data: undefined, error: null, isFetching: true, refetch }}
      >
        {(value: string) => <p>Loaded {value}</p>}
      </QueryView>,
    );
    expect(screen.getByRole("status")).toHaveTextContent("Loading…");
    rerender(
      <QueryView
        query={{
          data: undefined,
          error: new Error("Server unavailable"),
          isFetching: false,
          refetch,
        }}
      >
        {(value: string) => <p>Loaded {value}</p>}
      </QueryView>,
    );
    await userEvent
      .setup()
      .click(screen.getByRole("button", { name: "Try again" }));
    expect(refetch).toHaveBeenCalledOnce();
  });

  it("shows nothing while loading when loading is null", () => {
    const { container } = render(
      <QueryView
        query={{ data: undefined, error: null, isFetching: true }}
        loading={null}
      >
        {(value: string) => <p>Loaded {value}</p>}
      </QueryView>,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("adds a hint under the error panel and names the retry", async () => {
    const refetch = vi.fn();
    render(
      <QueryView
        query={{
          data: undefined,
          error: new Error("Gone"),
          isFetching: false,
          refetch,
        }}
        errorContext="Could not load the report"
        errorHint={(error) => `Open the check again (${error.message}).`}
        retryLabel="Reload report"
      >
        {(value: string) => <p>Loaded {value}</p>}
      </QueryView>,
    );
    expect(screen.getByRole("alert")).toHaveTextContent(
      "Could not load the report",
    );
    expect(screen.getByText("Open the check again (Gone).")).toBeVisible();
    await userEvent
      .setup()
      .click(screen.getByRole("button", { name: "Reload report" }));
    expect(refetch).toHaveBeenCalledOnce();
  });

  it("renders an area's own error, also over loaded data", () => {
    const renderError = vi.fn(
      (error: Error, _retry: () => void, stale: boolean) => (
        <p role="alert">
          {stale ? "Stale" : "Failed"}: {error.message}
        </p>
      ),
    );
    const { rerender } = render(
      <QueryView
        query={{ data: undefined, error: new Error("x"), isFetching: false }}
        renderError={renderError}
      >
        {(value: string) => <p>Loaded {value}</p>}
      </QueryView>,
    );
    expect(screen.getByRole("alert")).toHaveTextContent("Failed: x");
    rerender(
      <QueryView
        query={{ data: "a", error: new Error("y"), isFetching: false }}
        renderError={renderError}
      >
        {(value: string) => <p>Loaded {value}</p>}
      </QueryView>,
    );
    expect(screen.getByRole("alert")).toHaveTextContent("Stale: y");
    expect(screen.getByText("Loaded a")).toBeVisible();
  });

  it("keeps loaded pages and retries only the failed next page", async () => {
    const refetch = vi.fn();
    const fetchNextPage = vi.fn();
    render(
      <QueryView
        query={{
          data: "page one",
          error: new Error("Page two failed"),
          isFetching: false,
          isFetchNextPageError: true,
          refetch,
          fetchNextPage,
        }}
        continuationContext="More versions could not be loaded."
      >
        {(value: string) => <p>Loaded {value}</p>}
      </QueryView>,
    );
    expect(screen.getByText("Loaded page one")).toBeVisible();
    expect(screen.queryByText(/showing the last loaded data/)).toBeNull();
    expect(screen.getByRole("alert")).toHaveTextContent(
      "More versions could not be loaded.Page two failed",
    );
    await userEvent
      .setup()
      .click(screen.getByRole("button", { name: "Try again" }));
    expect(fetchNextPage).toHaveBeenCalledOnce();
    expect(refetch).not.toHaveBeenCalled();
  });
});
