import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter, RouterProvider } from "react-router";
import { expect, it, vi } from "vitest";
import captures from "../../../../api/testdata/http-capture-cases.json";
import {
  HTTP_EXCHANGE_MEDIA_TYPE,
  type ArtifactMetadata,
} from "../../api/artifacts";
import { PublicAPI } from "../../api/client";
import { PublicAPIProvider } from "../../api/context";
import { RunArtifactDetailRoute, RunArtifactLibrary } from "./artifacts";

it("browses 130 HTTP materials without reading other payloads and keeps selection exact", async () => {
  const source = JSON.stringify({
    schema: "contractor.http.exchange.v1",
    capture: captures.cases[0]!.value,
    finding: { proposal_id: "proposal-100", receipt_id: "receipt-100" },
  });
  const materials: ArtifactMetadata[] = Array.from(
    { length: 130 },
    (_, index) => ({
      artifact: {
        namespace: "finding-proposals",
        name: "http.exchange.proposal-" + String(index).padStart(3, "0"),
        revision: "exact-" + index,
      },
      mediaType: HTTP_EXCHANGE_MEDIA_TYPE,
      size: new TextEncoder().encode(source).length,
      frozen: true,
      current: true,
      createdAt: "2026-10-13T00:00:00Z",
    }),
  );
  const observed: URL[] = [];
  function json(value: unknown) {
    return new Response(JSON.stringify(value), {
      headers: {
        "Content-Type": "application/json",
        "X-Contractor-API-Version": "contractor.public.v1",
      },
    });
  }
  const api = new PublicAPI(
    {
      uiVersion: "0.6.4",
      supportedApiVersions: ["contractor.public.v1"],
      apiBaseUrl: "http://127.0.0.1:8080",
    },
    vi.fn(async (input) => {
      const url = new URL(input instanceof Request ? input.url : String(input));
      observed.push(url);
      if (url.pathname === "/v1/runs/run-test/artifacts") {
        expect(url.searchParams.get("limit")).toBe("50");
        const start = Number(url.searchParams.get("cursor") ?? "0");
        const end = start + 50;
        return json({
          items: materials.slice(start, end),
          page: {
            hasMore: end < materials.length,
            ...(end < materials.length ? { nextCursor: String(end) } : {}),
          },
        });
      }
      const selected = materials[100]!;
      if (url.pathname.endsWith("/metadata")) return json(selected);
      if (
        url.pathname.endsWith("/versions") ||
        url.pathname.endsWith("/lineage")
      )
        return json({ items: [], page: { hasMore: false } });
      expect(url.pathname).toBe(
        "/v1/runs/run-test/artifacts/finding-proposals/http.exchange.proposal-100",
      );
      expect(url.searchParams.get("revision")).toBe("exact-100");
      return new Response(source, {
        headers: {
          "Content-Type": HTTP_EXCHANGE_MEDIA_TYPE,
          "Content-Length": String(selected.size),
          "X-Contractor-API-Version": "contractor.public.v1",
        },
      });
    }),
  );
  const router = createMemoryRouter(
    [
      {
        path: "/",
        element: (
          <RunArtifactLibrary runId="run-test" open onToggle={() => {}} />
        ),
      },
      {
        path: "/runs/:runId/artifacts/:namespace/:name",
        element: <RunArtifactDetailRoute />,
      },
    ],
    { initialEntries: ["/"] },
  );
  render(
    <QueryClientProvider
      client={
        new QueryClient({ defaultOptions: { queries: { retry: false } } })
      }
    >
      <PublicAPIProvider api={api}>
        <RouterProvider router={router} />
      </PublicAPIProvider>
    </QueryClientProvider>,
  );
  const user = userEvent.setup();
  await screen.findByRole("link", { name: /http.exchange.proposal-000/ });
  await user.click(
    screen.getByRole("button", { name: "Possible issues and evidence" }),
  );
  await waitFor(() =>
    expect(observed.at(-1)?.searchParams.get("namespace")).toBe(
      "finding-proposals",
    ),
  );
  await user.click(screen.getByRole("button", { name: "Next" }));
  await screen.findByRole("link", { name: /http.exchange.proposal-050/ });
  await user.click(screen.getByRole("button", { name: "Next" }));
  const selected = await screen.findByRole("link", {
    name: /http.exchange.proposal-100/,
  });
  expect(
    observed.every((url) => url.pathname === "/v1/runs/run-test/artifacts"),
  ).toBe(true);
  await user.click(screen.getByRole("button", { name: "Previous" }));
  await screen.findByRole("link", { name: /http.exchange.proposal-050/ });
  await user.click(screen.getByRole("button", { name: "Next" }));
  await user.click(
    await screen.findByRole("link", { name: /http.exchange.proposal-100/ }),
  );
  expect(selected).toHaveAttribute(
    "href",
    "/runs/run-test/artifacts/finding-proposals/http.exchange.proposal-100?revision=exact-100",
  );
  await screen.findByRole("button", { name: "Load preview" });
  expect(
    observed.filter((url) =>
      url.pathname.endsWith("http.exchange.proposal-100"),
    ),
  ).toHaveLength(0);
  await user.click(screen.getByRole("button", { name: "Load preview" }));
  expect(await screen.findByRole("tab", { name: "Request" })).toBeVisible();
  expect(
    observed.filter((url) =>
      url.pathname.endsWith("http.exchange.proposal-100"),
    ),
  ).toHaveLength(1);
});
