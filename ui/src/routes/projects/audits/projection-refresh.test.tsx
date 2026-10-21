import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";

import type { Audit } from "../../../api/audits";
import { useAuditProjectionRefresh } from "./projection-refresh";

const audit = (revision: number, state: Audit["state"]) =>
  ({ auditId: "audit_example", revision, state }) as Audit;

describe("useAuditProjectionRefresh", () => {
  it("refreshes once when polling stops, even with a new key array per render", async () => {
    const queryClient = new QueryClient();
    const invalidate = vi.spyOn(queryClient, "invalidateQueries");
    const wrapper = ({ children }: { children: ReactNode }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );
    const { rerender } = renderHook(
      ({ current }: { current: Audit }) =>
        useAuditProjectionRefresh(current, ["audits", current.auditId]),
      { wrapper, initialProps: { current: audit(1, "active") } },
    );
    const finished = audit(2, "completed");
    rerender({ current: finished });
    await waitFor(() => expect(invalidate).toHaveBeenCalledOnce());
    rerender({ current: finished });
    rerender({ current: finished });
    await Promise.resolve();
    expect(invalidate).toHaveBeenCalledOnce();
    expect(invalidate).toHaveBeenCalledWith({
      queryKey: ["audits", "audit_example"],
      exact: true,
    });
  });
});

it("refreshes dependent projections immediately when preparation accepts its first Round", async () => {
  const queryClient = new QueryClient();
  const invalidate = vi.spyOn(queryClient, "invalidateQueries");
  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
  const before = { ...audit(1, "active"), phase: "preparing" as const };
  const after = {
    ...audit(2, "active"),
    phase: "rounds" as const,
    currentRoundId: "round_1",
  };
  const { rerender } = renderHook(
    ({ current }: { current: Audit }) =>
      useAuditProjectionRefresh(current, ["audits", current.auditId, "items"]),
    { wrapper, initialProps: { current: before as Audit } },
  );
  rerender({ current: after });
  await waitFor(() => expect(invalidate).toHaveBeenCalledOnce());
  rerender({ current: after });
  await Promise.resolve();
  expect(invalidate).toHaveBeenCalledOnce();
});
