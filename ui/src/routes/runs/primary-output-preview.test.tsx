import { screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { MAXIMUM_PREVIEW_BYTES } from "../../api/artifacts";
import type { RunStatus } from "../../api/runs";
import { conforming } from "../../test/builders";
import { workflowResource } from "../../test/builders-execution";
import { fakeServer, type Routes } from "../../test/fake-server";
import { bytes, failure, json } from "../../test/responses";
import { renderWithAPI } from "../../test/render";
import { RunOutputGallery } from "./outputs";

const saved = vi.hoisted(() => [] as Array<{ blob: Blob; filename: string }>);
vi.mock("../../app/download", () => ({
  saveBlob: (blob: Blob, filename: string) => saved.push({ blob, filename }),
}));

const workflow = workflowResource("source-review", "1", {
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
});

const REPORT = { namespace: "outputs", name: "report", revision: "report-r1" };
const TRACE = { namespace: "outputs", name: "trace", revision: "trace-r1" };

/** Metadata of `artifact`: a frozen file of `size` bytes. */
function metadata(
  artifact: typeof REPORT,
  mediaType: string,
  size: number,
  overrides: { current?: boolean; createdAt?: string } = {},
) {
  return conforming("ArtifactMetadata", {
    artifact,
    mediaType,
    size,
    current: overrides.current ?? true,
    frozen: true,
    createdAt: overrides.createdAt ?? "2026-09-07T08:00:00Z",
  });
}

/** The gallery of a succeeded Run with `outputs`; the Workflow is published. */
function renderGallery(outputs: RunStatus["outputs"], routes: Routes = {}) {
  const server = fakeServer({
    "GET /v1/workflows/{name}/versions/{version}": () => json(workflow),
    ...routes,
  });
  const view = renderWithAPI(
    <RunOutputGallery
      run={{
        runId: "run-primary",
        workflow: "source-review@1",
        state: "succeeded",
        outputs,
      }}
    />,
    { server },
  );
  return {
    ...view,
    /** Paths and query strings of the reads so far. */
    reads: () => server.requests.map((request) => new URL(request.url)),
  };
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
    const reportSource = "safe primary report";
    const { container, reads, user } = renderGallery(
      {
        report: REPORT,
        trace: TRACE,
        legacy: { namespace: "outputs", name: "legacy", revision: "legacy-r1" },
      },
      {
        "GET /v1/runs/{runId}/artifacts/{namespace}/{name}/metadata": ({
          url,
        }) => {
          expect(url.pathname).toBe(
            "/v1/runs/run-primary/artifacts/outputs/report/metadata",
          );
          expect(url.searchParams.get("revision")).toBe("report-r1");
          return json(metadata(REPORT, "text/plain", reportSource.length));
        },
        "GET /v1/runs/{runId}/artifacts/{namespace}/{name}": ({ url }) => {
          expect(url.pathname).toBe(
            "/v1/runs/run-primary/artifacts/outputs/report",
          );
          return bytes(reportSource, "text/plain");
        },
      },
    );

    expect(await screen.findAllByText("Declared primary result")).toHaveLength(
      2,
    );
    const headings = [...container.querySelectorAll(".run-result-card h3")].map(
      (heading) => heading.textContent,
    );
    expect(headings).toEqual(["report", "missing", "trace", "legacy"]);
    expect(resultCard("missing")).toHaveTextContent(
      "Required output is missing",
    );
    expect(
      reads().filter((url) => url.pathname.includes("/artifacts/")),
    ).toHaveLength(0);

    await user.click(
      within(resultCard("report")).getByRole("button", {
        name: "Preview result",
      }),
    );

    expect(
      await screen.findByText(reportSource, { selector: "pre" }),
    ).toBeInTheDocument();
    expect(
      reads().filter((url) =>
        url.pathname.includes("/artifacts/outputs/report"),
      ),
    ).toHaveLength(2);
    expect(
      reads().some((url) => url.pathname.includes("/artifacts/outputs/trace")),
    ).toBe(false);
    expect(screen.queryByRole("button", { name: "Load preview" })).toBeNull();
  });

  it("keeps present outputs unclassified when the exact Workflow is unavailable", async () => {
    const { reads } = renderGallery(
      { report: REPORT },
      {
        "GET /v1/workflows/{name}/versions/{version}": () =>
          failure(404, "not_found", "exact Workflow version is unavailable"),
      },
    );

    expect(
      await screen.findByText(/Output roles are unavailable/),
    ).toBeInTheDocument();
    expect(resultCard("report")).toHaveTextContent("Unclassified Run output");
    expect(screen.queryByText("Declared primary result")).toBeNull();
    expect(reads()).toHaveLength(1);
  });

  it("does not request bytes for an oversized selected output", async () => {
    const { reads, user } = renderGallery(
      { report: REPORT },
      {
        "GET /v1/runs/{runId}/artifacts/{namespace}/{name}/metadata": () =>
          json(metadata(REPORT, "text/plain", MAXIMUM_PREVIEW_BYTES + 1)),
      },
    );
    await screen.findAllByText("Declared primary result");
    await user.click(
      within(resultCard("report")).getByRole("button", {
        name: "Preview result",
      }),
    );

    expect(
      await screen.findByText(/Inline preview is unavailable/),
    ).toBeInTheDocument();
    expect(reads()).toHaveLength(2);
    expect(
      within(resultCard("report")).getByRole("link", {
        name: /Open outputs\/report@report-r1 details/,
      }),
    ).toHaveAttribute(
      "href",
      "/runs/run-primary/artifacts/outputs/report?revision=report-r1",
    );
  });

  it("rejects metadata for a different revision without loading its bytes", async () => {
    const { reads, user } = renderGallery(
      { report: REPORT },
      {
        "GET /v1/runs/{runId}/artifacts/{namespace}/{name}/metadata": () =>
          json(
            metadata({ ...REPORT, revision: "stale-r0" }, "text/plain", 5, {
              current: false,
              createdAt: "2026-09-07T07:59:00Z",
            }),
          ),
      },
    );
    await screen.findAllByText("Declared primary result");
    await user.click(
      within(resultCard("report")).getByRole("button", {
        name: "Preview result",
      }),
    );

    expect(
      await screen.findByText(/did not match the requested Artifact revision/),
    ).toBeInTheDocument();
    expect(reads()).toHaveLength(2);
    expect(screen.queryByText("stale-r0")).toBeNull();
  });

  it("downloads the exact original revision with one action", async () => {
    saved.length = 0;
    const source = '{"report":true}';
    const { reads, user } = renderGallery(
      { trace: TRACE },
      {
        "GET /v1/runs/{runId}/artifacts/{namespace}/{name}/metadata": ({
          url,
        }) => {
          expect(url.pathname).toBe(
            "/v1/runs/run-primary/artifacts/outputs/trace/metadata",
          );
          return json(metadata(TRACE, "application/json", source.length));
        },
        "GET /v1/runs/{runId}/artifacts/{namespace}/{name}": ({ url }) => {
          expect(url.pathname).toBe(
            "/v1/runs/run-primary/artifacts/outputs/trace",
          );
          // The payload is an artifact's own JSON, not an API body.
          return bytes(source, "application/json");
        },
      },
    );
    await screen.findAllByText("Declared supporting result");

    await user.click(
      within(resultCard("trace")).getByRole("button", {
        name: "Download trace",
      }),
    );

    await vi.waitFor(() => expect(saved).toHaveLength(1));
    expect(await saved[0]!.blob.text()).toBe(source);
    expect(
      reads()
        .filter((url) => url.pathname.includes("/artifacts/"))
        .map((url) => `${url.pathname}?${url.searchParams.toString()}`),
    ).toEqual([
      "/v1/runs/run-primary/artifacts/outputs/trace/metadata?revision=trace-r1",
      "/v1/runs/run-primary/artifacts/outputs/trace?revision=trace-r1",
    ]);
    // Downloading does not open the inline preview.
    expect(
      within(resultCard("trace")).getByRole("button", {
        name: "Preview result",
      }),
    ).toBeInTheDocument();
  });
});
