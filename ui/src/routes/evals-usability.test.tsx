import { webcrypto } from "node:crypto";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter } from "react-router";
import { beforeEach, expect, it, vi } from "vitest";
import { PublicAPI } from "../api/client";
import { Application } from "../app/application";
import { applicationRoutes } from "../app/router";
import { MemoryStorage } from "../test/storage";
import {
  createEvalFixture,
  EVAL_API_VERSION,
  EVAL_FIXTURE_ORIGIN,
} from "../test/evals-fixture";
import pairs from "../../../api/testdata/evals/valid/pair-page.json";
import { MAX_EVAL_DOCUMENT_BYTES } from "./evals/setup-model";

beforeEach(() => {
  vi.stubGlobal("localStorage", new MemoryStorage());
  vi.stubGlobal("crypto", webcrypto);
});

function start(fixture: ReturnType<typeof createEvalFixture>, path: string) {
  const api = new PublicAPI(
    {
      uiVersion: "0.4.0",
      apiBaseUrl: EVAL_FIXTURE_ORIGIN,
      supportedApiVersions: [EVAL_API_VERSION],
    },
    async (input) =>
      fixture.fetch(input instanceof Request ? input : new Request(input)),
  );
  const router = createMemoryRouter(applicationRoutes(), {
    initialEntries: [path],
  });
  return {
    ...render(<Application api={api} publicAPI={api} router={router} />),
    router,
  };
}

function datasetFile(raw: string): File {
  const file = new File([raw], "dataset.json", { type: "application/json" });
  Object.defineProperty(file, "text", { value: async () => raw });
  return file;
}

it("retries the current pair page after a transient network failure", async () => {
  const fixture = createEvalFixture({ prepared: true });
  const original = fixture.fetch;
  let fail = true,
    requests = 0;
  fixture.fetch = async (request) => {
    if (new URL(request.url).pathname.endsWith("/pairs")) {
      requests++;
      if (fail) throw new TypeError("offline");
    }
    return original(request);
  };
  const user = userEvent.setup();
  start(
    fixture,
    "/evals/experiments/experiment-1/comparison?filter=all&viewSnapshot=view-7",
  );
  await screen.findByText("Public API is unavailable");
  fail = false;
  await user.click(screen.getByRole("button", { name: "Try again" }));
  await waitFor(() => expect(requests).toBeGreaterThan(1));
  expect(
    await screen.findByRole("link", { name: "unsafe-query / sample 1" }),
  ).toBeVisible();
  expect(
    screen.queryByText("Public API is unavailable"),
  ).not.toBeInTheDocument();
});

it("supports typing comma-separated runtime labels", async () => {
  const fixture = createEvalFixture();
  const user = userEvent.setup();
  start(fixture, "/evals/experiments/experiment-1/setup");
  await screen.findByLabelText("A runtime labels");
  await user.click(
    screen.getAllByText("Execution settings", { exact: true })[0]!,
  );
  const input = screen.getByLabelText("A runtime labels");
  await user.clear(input);
  await user.type(input, "linux, gpu");
  expect(input).toHaveValue("linux, gpu");
  await user.click(screen.getByRole("button", { name: "Save draft" }));
  await waitFor(() =>
    expect(fixture.state.experiment.draft!.variants[0]!.runtimeLabels).toEqual([
      "linux",
      "gpu",
    ]),
  );
});

it("labels the configured baseline as A regardless of variant array order", async () => {
  const fixture = createEvalFixture();
  const draft = fixture.state.experiment.draft!;
  draft.variants.reverse();
  const user = userEvent.setup();
  start(fixture, "/evals/experiments/experiment-1/setup");
  const baseline = draft.variants.find(
    (v) => v.id === draft.comparison.baseline,
  )!;
  expect(await screen.findByLabelText("A version")).toHaveValue(
    baseline.selector,
  );
  await within(screen.getByLabelText("A version")).findByRole("option", {
    name: /trace-a@2/,
  });
  await user.selectOptions(screen.getByLabelText("A version"), "trace-a@2");
  await user.click(screen.getByRole("button", { name: "Save draft" }));
  await waitFor(() =>
    expect(
      fixture.state.experiment.draft!.variants.find(
        (v) => v.id === draft.comparison.baseline,
      )?.selector,
    ).toBe("trace-a@2"),
  );
  expect(
    fixture.state.experiment.draft!.variants.find(
      (v) => v.id === draft.comparison.candidate,
    )?.selector,
  ).toBe("trace-b@1");
});

it("refreshes execution inventory when the pair evidence is refreshed", async () => {
  const fixture = createEvalFixture({ prepared: true });
  const original = fixture.fetch;
  let deleted = false;
  fixture.fetch = async (request) => {
    const response = await original(request);
    if (deleted && new URL(request.url).pathname.endsWith("/executions")) {
      const body = await response.json();
      for (const item of body.items) item.available = false;
      return new Response(JSON.stringify(body), {
        status: response.status,
        headers: response.headers,
      });
    }
    return response;
  };
  const user = userEvent.setup();
  start(
    fixture,
    `/evals/experiments/experiment-1/pairs/${pairs.items[0]!.pairId}`,
  );
  await screen.findAllByRole("link", { name: "Run run-1" });
  deleted = true;
  await user.click(screen.getByRole("button", { name: "Refresh" }));
  await waitFor(() =>
    expect(screen.queryAllByRole("link", { name: "Run run-1" })).toHaveLength(
      0,
    ),
  );
});

it("shows persisted diagnostics after preparation returns the experiment to draft", async () => {
  const fixture = createEvalFixture();
  fixture.state.experiment.diagnostics = [
    { code: "eval_pin_mismatch", field: "", recovery: "duplicate" },
  ];
  const user = userEvent.setup();
  start(fixture, "/evals/experiments/experiment-1/setup");
  await screen.findByLabelText("Experiment name");
  expect(screen.getByRole("alert")).toHaveTextContent(
    "Preparation needs attention",
  );
  expect(screen.getByRole("alert")).toHaveTextContent(
    "Review the versions and equality policy",
  );
  await user.click(screen.getByRole("button", { name: "4. Readiness" }));
  await user.click(screen.getByText("Diagnostic details"));
  expect(screen.getByText("eval_pin_mismatch")).toBeVisible();
});

it("keeps preparation diagnostics visible while the prepared setup is unavailable", async () => {
  const fixture = createEvalFixture();
  fixture.state.experiment.state = "preparing";
  fixture.state.experiment.diagnostics = [
    { code: "eval_preparation_unavailable", field: "", recovery: "wait" },
  ];
  start(fixture, "/evals/experiments/experiment-1/setup");
  expect(await screen.findByRole("alert")).toHaveTextContent(
    "Preparation is temporarily unavailable",
  );
});

it("preserves typed capability separators and saves separate dataset capabilities", async () => {
  const fixture = createEvalFixture();
  const user = userEvent.setup();
  start(fixture, "/evals/datasets?project=evaluation-1");
  await user.click(
    await screen.findByRole("button", { name: "Create or import dataset" }),
  );
  await user.type(
    screen.getByLabelText("Dataset ID", { exact: true }),
    "typed-capabilities",
  );
  await user.type(screen.getByLabelText("Dataset name"), "Typed capabilities");
  await user.type(
    screen.getByLabelText("Visible task objective"),
    "Inspect the source",
  );
  const input = screen.getByLabelText("Required capabilities");
  await user.type(input, "linux, gpu");
  expect(input).toHaveValue("linux, gpu");
  await user.click(
    screen.getByRole("button", { name: "Save dataset revision" }),
  );
  await waitFor(() => {
    const saved = fixture.state.requests.find(
      (r) => r.method === "POST" && r.path.endsWith("/eval-datasets"),
    );
    expect(saved?.body).toMatchObject({
      cases: [expect.objectContaining({ requires: ["linux", "gpu"] })],
    });
  });
});

it("rejects an oversized dataset request before showing cases or sending POST", async () => {
  const fixture = createEvalFixture();
  const user = userEvent.setup();
  start(fixture, "/evals/datasets?project=evaluation-1");
  await user.click(
    await screen.findByRole("button", { name: "Create or import dataset" }),
  );
  const dataset = structuredClone(fixture.state.dataset);
  dataset.name = "Oversized import";
  dataset.cases[0]!.task.objective = "x".repeat(MAX_EVAL_DOCUMENT_BYTES);
  await user.upload(
    screen.getByLabelText("Import dataset JSON"),
    datasetFile(JSON.stringify(dataset)),
  );

  expect(
    await screen.findByText(/1 MiB evaluation document limit/u),
  ).toBeVisible();
  expect(
    screen.queryByText(/Oversized import · \d+ cases/u),
  ).not.toBeInTheDocument();
  expect(
    screen.queryByRole("button", { name: "Save dataset revision" }),
  ).not.toBeInTheDocument();
  expect(
    fixture.state.requests.some(
      (request) =>
        request.method === "POST" && request.path.endsWith("/eval-datasets"),
    ),
  ).toBe(false);
});

it("imports a pretty source over 1 MiB when its compact request fits", async () => {
  const fixture = createEvalFixture();
  const user = userEvent.setup();
  start(fixture, "/evals/datasets?project=evaluation-1");
  await user.click(
    await screen.findByRole("button", { name: "Create or import dataset" }),
  );
  const dataset = structuredClone(fixture.state.dataset);
  dataset.name = "Compact import";
  const compactBytes = new TextEncoder().encode(JSON.stringify(dataset)).length;
  const formattingBytes =
    new TextEncoder().encode(JSON.stringify(dataset, null, 2)).length -
    compactBytes;
  dataset.cases[0]!.task.objective += "x".repeat(
    MAX_EVAL_DOCUMENT_BYTES - compactBytes - Math.floor(formattingBytes / 2),
  );
  const source = JSON.stringify(dataset, null, 2);
  const file = datasetFile(source);
  expect(file.size).toBeGreaterThan(MAX_EVAL_DOCUMENT_BYTES);
  expect(
    new TextEncoder().encode(JSON.stringify({ ...dataset, privateChecks: [] }))
      .length,
  ).toBeLessThanOrEqual(MAX_EVAL_DOCUMENT_BYTES);
  await user.upload(screen.getByLabelText("Import dataset JSON"), file);

  expect(await screen.findByText(/Compact import · \d+ cases/u)).toBeVisible();
  await user.click(
    screen.getByRole("button", { name: "Save dataset revision" }),
  );
  await waitFor(() =>
    expect(
      fixture.state.requests.some(
        (request) =>
          request.method === "POST" && request.path.endsWith("/eval-datasets"),
      ),
    ).toBe(true),
  );
});

it("blocks an imported dataset when selected private checks exceed 1 MiB", async () => {
  const fixture = createEvalFixture();
  const user = userEvent.setup();
  start(fixture, "/evals/datasets?project=evaluation-1");
  await user.click(
    await screen.findByRole("button", { name: "Create or import dataset" }),
  );
  const dataset = structuredClone(fixture.state.dataset);
  dataset.name = "Private-heavy import";
  dataset.privateChecks = [
    {
      id: "long",
      revision: "r1",
      rubric: "x".repeat(MAX_EVAL_DOCUMENT_BYTES),
      expected: {},
    },
  ];
  await user.upload(
    screen.getByLabelText("Import dataset JSON"),
    datasetFile(JSON.stringify(dataset)),
  );
  expect(
    await screen.findByText(/Private-heavy import · \d+ cases/u),
  ).toBeVisible();

  await user.click(
    screen.getByLabelText(/Include private assessment rubrics/u),
  );
  expect(screen.getByText(/1 MiB evaluation document limit/u)).toBeVisible();
  expect(
    screen.getByRole("button", { name: "Save dataset revision" }),
  ).toBeDisabled();
  expect(
    screen.queryByText(/Private-heavy import · \d+ cases/u),
  ).not.toBeInTheDocument();

  await user.click(
    screen.getByLabelText(/Include private assessment rubrics/u),
  );
  expect(
    screen.getByRole("button", { name: "Save dataset revision" }),
  ).toBeEnabled();
});

it("shows both arms against all expected attempts, including missing and unscored evidence", async () => {
  const fixture = createEvalFixture({ prepared: true });
  start(fixture, "/evals/experiments/experiment-1/overview");
  const coverage = await screen.findByRole("region", {
    name: "Experiment coverage",
  });
  expect(within(coverage).getByText("Conclusion")).toBeVisible();
  expect(within(coverage).getByText("Inconclusive")).toBeVisible();
  const candidate = within(coverage)
    .getByRole("heading", { name: "B · Candidate" })
    .closest("section")!;
  expect(
    within(candidate).getByText("Scored").nextElementSibling,
  ).toHaveTextContent("3 / 4");
  expect(
    within(candidate).getByText("End-to-end passed").nextElementSibling,
  ).toHaveTextContent("2 / 4");
});
