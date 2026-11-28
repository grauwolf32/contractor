import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useState } from "react";
import { QueryClient } from "@tanstack/react-query";
import { costFixture } from "../test/llm-cost-fixtures";
import { renderWithAPI } from "../test/render";
import { fakeServer } from "../test/fake-server";
import { failure, json, unavailableCostForURL } from "../test/responses";
import { Button } from "../ui";
import { LLMCostPanel } from "./llm-cost";
import { LLMCostCard } from "../widgets/llm-cost/card";
import { formatUSD } from "./format";

describe("LLM cost presentation", () => {
  afterEach(() => vi.useRealTimers());
  it.each([
    [null, "—"],
    ["0", "$0.00"],
    ["0.00000625", "<$0.01"],
    ["12.845", "$12.85"],
    [
      "99999999999999999999999999999999999999.9",
      "$99,999,999,999,999,999,999,999,999,999,999,999,999.90",
    ],
  ])("formats %s without binary money arithmetic", (amount, display) => {
    expect(formatUSD(amount)).toBe(display);
  });
  it("shows exact tiny cost and switches breakdown in place", () => {
    render(<LLMCostCard value={costFixture("run", "run_1")} />);
    expect(screen.getByText("<$0.01")).toHaveAttribute(
      "title",
      "USD 0.00000625",
    );
    fireEvent.click(screen.getByRole("button", { name: "Details" }));
    expect(screen.getAllByText("USD 0.00000625").length).toBeGreaterThan(0);
    fireEvent.click(screen.getByRole("button", { name: "By model" }));
    expect(screen.getByText("openrouter/demo-worker")).toBeVisible();
    expect(screen.queryByText("Worker analysis")).not.toBeInTheDocument();
  });
  it("keeps source and coverage separate and retains the Project partition", () => {
    render(
      <LLMCostCard
        value={costFixture("project", "project_shop", {
          coverage: "partial",
          basis: "estimated",
          historicalGap: true,
        })}
      />,
    );
    expect(screen.getByText("Estimate")).toBeVisible();
    expect(screen.getByText("Partial")).toBeVisible();
    expect(screen.getByText("$48.76")).toBeVisible();
    expect(screen.getByText("$37.70")).toBeVisible();
    expect(screen.getByText("$11.06")).toBeVisible();
    expect(screen.getByText(/Known subtotal/)).toBeVisible();
  });
  it("refetches once on terminal transition and then stops polling", async () => {
    vi.useFakeTimers();
    const server = fakeServer({
      "GET /v1/runs/{runId}/llm-cost": () => json(costFixture("run", "run_1")),
    });
    function View() {
      const [active, setActive] = useState(true);
      return (
        <>
          <Button onClick={() => setActive(false)}>Finish</Button>
          <LLMCostPanel
            scope="run"
            id="run_1"
            active={active}
            refreshKey={active ? "running" : "succeeded"}
          />
        </>
      );
    }
    renderWithAPI(<View />, { server });
    await act(() => vi.advanceTimersByTimeAsync(100));
    expect(server.sent("GET", "/llm-cost")).toHaveLength(1);
    fireEvent.click(screen.getByRole("button", { name: "Finish" }));
    await act(() => vi.advanceTimersByTimeAsync(100));
    expect(server.sent("GET", "/llm-cost")).toHaveLength(2);
    await act(() => vi.advanceTimersByTimeAsync(30000));
    expect(server.sent("GET", "/llm-cost")).toHaveLength(2);
    expect(
      within(screen.getByRole("region", { name: "Run LLM cost" })).getByText(
        "<$0.01",
      ),
    ).toBeVisible();
  });
  it("distinguishes unavailable cost from a known zero", () => {
    const unknown = unavailableCostForURL(
      "GET",
      new URL("https://example.test/v1/runs/run_1/llm-cost"),
    )!;
    const view = render(<LLMCostCard value={unknown} />);
    expect(screen.getByText("No recorded cost")).toBeVisible();
    expect(screen.getByText("No cost data")).toBeVisible();
    expect(
      screen.getByText("Earlier work has no retained monetary data."),
    ).toBeVisible();
    expect(screen.queryByText("$0.00")).not.toBeInTheDocument();
    view.rerender(
      <LLMCostCard
        value={costFixture("run", "run_1", {
          amount: "0",
          checksAmount: "0",
          standaloneAmount: "0",
          calls: "0",
          pricedCalls: "0",
          basis: "none",
          activity: [],
          models: [],
        })}
      />,
    );
    expect(screen.getByText("$0.00")).toBeVisible();
    expect(screen.queryByText("Unavailable")).not.toBeInTheDocument();
  });
  it("keeps the last known amount when a refresh fails", async () => {
    let failed = false;
    const server = fakeServer({
      "GET /v1/runs/{runId}/llm-cost": () =>
        failed
          ? failure(503, "storage_transaction_conflict")
          : json(costFixture("run", "run_1")),
    });
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    const view = renderWithAPI(<LLMCostPanel scope="run" id="run_1" />, {
      server,
      queryClient: client,
    });
    expect(await screen.findByText("<$0.01")).toBeVisible();
    failed = true;
    await act(() =>
      view.queryClient.invalidateQueries({
        queryKey: ["llm-cost", "run", "run_1"],
      }),
    );
    expect(
      await screen.findByText(
        "Could not refresh; showing the last loaded data.",
      ),
    ).toBeVisible();
    expect(screen.getByText("<$0.01")).toBeVisible();
    expect(screen.queryByText("$0.00")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Try again" })).toBeVisible();
  });
});
