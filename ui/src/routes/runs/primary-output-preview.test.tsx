import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router";
import { describe, expect, it, vi } from "vitest";

import { PublicAPI } from "../../api/client";
import { PublicAPIProvider } from "../../api/context";
import type { RunStatus } from "../../api/runs";
import type { WorkflowResource } from "../../api/workflows";
import type { RuntimeConfig } from "../../config/runtime-config";
import { RunOutputGallery } from "./artifacts";

const runtimeConfig: RuntimeConfig = {
  uiVersion: "0.1.0",
  supportedApiVersions: ["contractor.public.v1"],
  apiBaseUrl: "http://127.0.0.1:8080",
};

const workflow = {
  ref: { name: "source-review", version: "1" },
  entryStage: "review",
  parameters: {},
  inputs: {},
  outputs: {
    report: {
      required: true,
      mediaTypes: ["text/plain"],
      primary: true,
    },
    missing: {
      required: true,
      mediaTypes: ["text/plain"],
      primary: true,
    },
    trace: { required: false, mediaTypes: ["application/json"] },
  },
  stages: {},
} satisfies WorkflowResource;

type GalleryRun = Pick<RunStatus, "runId" | "workflow" | "state" | "outputs">;

function runFixture(outputs: RunStatus["outputs"]): GalleryRun {
  return {
    runId: "run-primary",
    workflow: "source-review@1",
    state: "succeeded",
    outputs,
  };
}

function apiResponse(
  body: BodyInit | null,
  options: ResponseInit = {},
): Response {
  const headers = new Headers(options.headers);
  headers.set("X-Contractor-API-Version", "contractor.public.v1");
  return new Response(body, { ...options, headers });
}

function jsonResponse(value: unknown, options: ResponseInit = {}): Response {
  return apiResponse(JSON.stringify(value), {
    ...options,
    headers: { "content-type": "application/json", ...options.headers },
  });
}

function renderGallery(api: PublicAPI, run: GalleryRun) {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <PublicAPIProvider api={api}>
        <MemoryRouter>
          <RunOutputGallery run={run} />
        </MemoryRouter>
      </PublicAPIProvider>
    </QueryClientProvider>,
  );
}

function resultCard(name: string): HTMLElement {
  const card = screen.getByRole("heading", { name }).closest("article");
  if (card === null) {
    throw new Error(`Result card ${name} is missing`);
  }
  return card;
}

describe("Run primary output preview", () => {
  it("loads only one selected exact result after one action", async () => {
    const requests: Request[] = [];
    const reportSource = "safe primary report";
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        requests.push(request);
        const url = new URL(request.url);
        if (url.pathname === "/v1/workflows/source-review/versions/1") {
          return jsonResponse(workflow);
        }
        if (
          url.pathname ===
          "/v1/runs/run-primary/artifacts/outputs/report/metadata"
        ) {
          expect(url.searchParams.get("revision")).toBe("report-r1");
          return jsonResponse({
            artifact: {
              namespace: "outputs",
              name: "report",
              revision: "report-r1",
            },
            mediaType: "text/plain",
            size: reportSource.length,
            current: true,
            frozen: true,
            createdAt: "2026-09-07T08:00:00Z",
          });
        }
        if (url.pathname === "/v1/runs/run-primary/artifacts/outputs/report") {
          return apiResponse(reportSource, {
            headers: {
              "content-length": String(reportSource.length),
              "content-type": "text/plain",
            },
          });
        }
        throw new Error(`Unexpected request ${request.method} ${url.pathname}`);
      }),
    );
    const view = renderGallery(
      api,
      runFixture({
        report: {
          namespace: "outputs",
          name: "report",
          revision: "report-r1",
        },
        trace: {
          namespace: "outputs",
          name: "trace",
          revision: "trace-r1",
        },
        legacy: {
          namespace: "outputs",
          name: "legacy",
          revision: "legacy-r1",
        },
      }),
    );

    expect(await screen.findAllByText("Declared primary result")).toHaveLength(
      2,
    );
    const headings = [
      ...view.container.querySelectorAll(".run-result-card h4"),
    ].map((heading) => heading.textContent);
    expect(headings).toEqual(["report", "missing", "trace", "legacy"]);
    expect(resultCard("missing")).toHaveTextContent(
      "Required output is missing",
    );
    expect(
      requests.filter((request) => request.url.includes("/artifacts/")),
    ).toHaveLength(0);

    await userEvent.click(
      within(resultCard("report")).getByRole("button", {
        name: "Preview result",
      }),
    );

    expect(
      await screen.findByText(reportSource, { selector: "pre" }),
    ).toBeInTheDocument();
    expect(
      requests.filter((request) =>
        request.url.includes("/artifacts/outputs/report"),
      ),
    ).toHaveLength(2);
    expect(
      requests.some((request) =>
        request.url.includes("/artifacts/outputs/trace"),
      ),
    ).toBe(false);
    expect(screen.queryByRole("button", { name: "Load preview" })).toBeNull();
  });

  it("keeps present outputs unclassified when the exact Workflow is unavailable", async () => {
    const requests: Request[] = [];
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        requests.push(request);
        return jsonResponse(
          {
            code: "not_found",
            message: "exact Workflow version is unavailable",
            retryable: false,
            requestId: "request-workflow-missing",
          },
          { status: 404 },
        );
      }),
    );
    renderGallery(
      api,
      runFixture({
        report: {
          namespace: "outputs",
          name: "report",
          revision: "report-r1",
        },
      }),
    );

    expect(
      await screen.findByText(/Output roles are unavailable/),
    ).toBeInTheDocument();
    expect(resultCard("report")).toHaveTextContent("Unclassified Run output");
    expect(screen.queryByText("Declared primary result")).toBeNull();
    expect(requests).toHaveLength(1);
  });

  it("does not request bytes for an oversized selected output", async () => {
    const requests: Request[] = [];
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        requests.push(request);
        const url = new URL(request.url);
        if (url.pathname === "/v1/workflows/source-review/versions/1") {
          return jsonResponse(workflow);
        }
        if (url.pathname.endsWith("/metadata")) {
          return jsonResponse({
            artifact: {
              namespace: "outputs",
              name: "report",
              revision: "report-r1",
            },
            mediaType: "text/plain",
            size: 256 * 1024 + 1,
            current: true,
            frozen: true,
            createdAt: "2026-09-07T08:00:00Z",
          });
        }
        throw new Error(`Unexpected byte request ${url.pathname}`);
      }),
    );
    renderGallery(
      api,
      runFixture({
        report: {
          namespace: "outputs",
          name: "report",
          revision: "report-r1",
        },
      }),
    );
    await screen.findAllByText("Declared primary result");
    await userEvent.click(
      within(resultCard("report")).getByRole("button", {
        name: "Preview result",
      }),
    );

    expect(
      await screen.findByText(/Inline preview is unavailable/),
    ).toBeInTheDocument();
    expect(requests).toHaveLength(2);
    expect(
      within(resultCard("report")).getByRole("link", {
        name: /Open outputs\/report@report-r1 exact details/,
      }),
    ).toHaveAttribute(
      "href",
      "/runs/run-primary/artifacts/outputs/report?revision=report-r1",
    );
  });

  it("rejects metadata for a different revision without loading its bytes", async () => {
    const requests: Request[] = [];
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        requests.push(request);
        const url = new URL(request.url);
        if (url.pathname === "/v1/workflows/source-review/versions/1") {
          return jsonResponse(workflow);
        }
        if (url.pathname.endsWith("/metadata")) {
          return jsonResponse({
            artifact: {
              namespace: "outputs",
              name: "report",
              revision: "stale-r0",
            },
            mediaType: "text/plain",
            size: 5,
            current: false,
            frozen: true,
            createdAt: "2026-09-07T07:59:00Z",
          });
        }
        throw new Error(`Unexpected byte request ${url.pathname}`);
      }),
    );
    renderGallery(
      api,
      runFixture({
        report: {
          namespace: "outputs",
          name: "report",
          revision: "report-r1",
        },
      }),
    );
    await screen.findAllByText("Declared primary result");
    await userEvent.click(
      within(resultCard("report")).getByRole("button", {
        name: "Preview result",
      }),
    );

    expect(
      await screen.findByText(/did not match the selected exact Run output/),
    ).toBeInTheDocument();
    expect(requests).toHaveLength(2);
    expect(screen.queryByText("stale-r0")).toBeNull();
  });
});
