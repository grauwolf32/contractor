import { screen, waitFor } from "@testing-library/react";
import { expect, it } from "vitest";

import captures from "../../../../api/testdata/http-capture-cases.json";
import { HTTP_EXCHANGE_MEDIA_TYPE } from "../../api/artifacts";
import { conforming } from "../../test/builders";
import { bytes, fakeServer, json, page } from "../../test/fake-server";
import { renderWithAPI } from "../../test/render";
import { RunArtifactDetailRoute, RunArtifactLibrary } from "./artifacts";

const SOURCE = JSON.stringify({
  schema: "contractor.http.exchange.v1",
  capture: captures.cases[0]!.value,
  finding: { proposal_id: "proposal-100", receipt_id: "receipt-100" },
});

/** 130 HTTP exchanges saved with possible issues: three pages of 50. */
const MATERIALS = Array.from({ length: 130 }, (_, index) =>
  conforming("ArtifactMetadata", {
    artifact: {
      namespace: "finding-proposals",
      name: "http.exchange.proposal-" + String(index).padStart(3, "0"),
      revision: "exact-" + index,
    },
    mediaType: HTTP_EXCHANGE_MEDIA_TYPE,
    size: new TextEncoder().encode(SOURCE).length,
    frozen: true,
    current: true,
    createdAt: "2026-10-13T00:00:00Z",
  }),
);

// Three pages of a realistic list render in about 6 s on a loaded machine,
// past the default 5 s; the test asserts reads, never time.
it(
  "browses 130 HTTP materials without reading other payloads and keeps selection exact",
  {
    timeout: 20_000,
  },
  async () => {
    const selected = MATERIALS[100]!;
    const server = fakeServer({
      "GET /v1/runs/{runId}/artifacts": ({ url }) => {
        expect(url.searchParams.get("limit")).toBe("50");
        const start = Number(url.searchParams.get("cursor") ?? "0");
        const end = start + 50;
        return json(
          page(
            MATERIALS.slice(start, end),
            end < MATERIALS.length ? String(end) : undefined,
          ),
        );
      },
      "GET /v1/runs/{runId}/artifacts/{namespace}/{name}/metadata": () =>
        json(selected),
      "GET /v1/runs/{runId}/artifacts/{namespace}/{name}/versions": () =>
        json(page([])),
      "GET /v1/runs/{runId}/artifacts/{namespace}/{name}/lineage": () =>
        json(page([])),
      "GET /v1/runs/{runId}/artifacts/{namespace}/{name}": ({ url }) => {
        expect(url.pathname).toBe(
          "/v1/runs/run-test/artifacts/finding-proposals/http.exchange.proposal-100",
        );
        expect(url.searchParams.get("revision")).toBe("exact-100");
        return bytes(SOURCE, HTTP_EXCHANGE_MEDIA_TYPE, {
          "Content-Length": String(selected.size),
        });
      },
    });
    const observed = () =>
      server.requests.map((request) => new URL(request.url));
    const { user } = renderWithAPI(null, {
      server,
      routes: [
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
    });
    await screen.findByRole("link", { name: /http.exchange.proposal-000/ });
    await user.click(
      screen.getByRole("button", { name: "Possible issues and evidence" }),
    );
    await waitFor(() =>
      expect(observed().at(-1)?.searchParams.get("namespace")).toBe(
        "finding-proposals",
      ),
    );
    await user.click(screen.getByRole("button", { name: "Next" }));
    await screen.findByRole("link", { name: /http.exchange.proposal-050/ });
    await user.click(screen.getByRole("button", { name: "Next" }));
    const link = await screen.findByRole("link", {
      name: /http.exchange.proposal-100/,
    });
    expect(
      observed().every((url) => url.pathname === "/v1/runs/run-test/artifacts"),
    ).toBe(true);
    await user.click(screen.getByRole("button", { name: "Previous" }));
    await screen.findByRole("link", { name: /http.exchange.proposal-050/ });
    await user.click(screen.getByRole("button", { name: "Next" }));
    await user.click(
      await screen.findByRole("link", { name: /http.exchange.proposal-100/ }),
    );
    expect(link).toHaveAttribute(
      "href",
      "/runs/run-test/artifacts/finding-proposals/http.exchange.proposal-100?revision=exact-100",
    );
    await screen.findByRole("button", { name: "Load preview" });
    const payloadReads = () =>
      observed().filter((url) =>
        url.pathname.endsWith("http.exchange.proposal-100"),
      );
    expect(payloadReads()).toHaveLength(0);
    await user.click(screen.getByRole("button", { name: "Load preview" }));
    expect(await screen.findByRole("tab", { name: "Request" })).toBeVisible();
    expect(payloadReads()).toHaveLength(1);
  },
);
