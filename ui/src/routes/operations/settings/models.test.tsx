import { fireEvent, screen, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { conforming } from "../../../test/builders";
import { operationsSnapshot } from "../../../test/builders-execution";
import { renderApplication } from "../../../test/application";
import { json, preconditionFailed } from "../../../test/responses";
import type { Routes } from "../../../test/fake-server";
import type { ModelSelectionValue } from "../../../api/model-selection";

const digest = `sha256:${"b".repeat(64)}`;
const models = ["alpha", "beta"].map((id) => ({
  id,
  digest,
  upstreamModel: `test/${id}`,
  contextWindowTokens: id === "alpha" ? 128000 : 1000000,
  maxOutputTokens: 32768,
  policies: {
    planner: `planner_${id}@1`,
    worker: `worker_${id}@1`,
    summarizer: `summarizer_${id}@1`,
    audit_completion_worker: `audit_${id}@1`,
  },
}));
function response(selection: ModelSelectionValue, revision: string) {
  return json(
    conforming("ModelSelectionInfo", {
      configured: true,
      models,
      selection,
      revision,
      catalogDigest: digest,
      updatedAt: "2026-10-21T00:00:00Z",
    }),
    200,
    { ETag: `"${revision}:${digest.slice(7)}"`, "Cache-Control": "no-store" },
  );
}
const common: Routes = {
  "GET /v1/operations/snapshot": () => json(operationsSnapshot()),
  "GET /v1/settings/git-key": () => json({ configured: false }),
  "GET /v1/operations/settings/scheduler": () =>
    json(
      {
        maxConcurrentRuns: 1,
        revision: "1",
        updatedAt: "2026-10-21T00:00:00Z",
      },
      200,
      { ETag: '"1"' },
    ),
};

describe("Server model selection", () => {
  it("saves one default and independent role choices with the current ETag", async () => {
    let saved: ModelSelectionValue = { defaultModel: "alpha", roles: {} };
    let revision = "1";
    const { requests } = renderApplication("/operations/settings", {
      capabilities: ["user", "operations"],
      server: {
        ...common,
        "GET /v1/operations/settings/models": () => response(saved, revision),
        "PUT /v1/operations/settings/models": async ({ request }) => {
          expect(request.headers.get("If-Match")).toBe(
            `"1:${digest.slice(7)}"`,
          );
          saved = (await request.json()) as ModelSelectionValue;
          expect(saved).toEqual({
            defaultModel: "beta",
            roles: { summarizer: "alpha" },
          });
          revision = "2";
          return response(saved, revision);
        },
      },
    });
    const model = await screen.findByRole("combobox", {
      name: "Default model",
    });
    fireEvent.change(model, { target: { value: "beta" } });
    fireEvent.click(screen.getByText("Models by role · all use default"));
    fireEvent.change(
      screen.getByRole("combobox", { name: "Summarizer model" }),
      { target: { value: "alpha" } },
    );
    fireEvent.click(
      screen.getByRole("button", { name: "Save model defaults" }),
    );
    expect(
      await screen.findByText("Model defaults saved for new executions."),
    ).toBeInTheDocument();
    expect(model).toHaveValue("beta");
    expect(
      screen.getByRole("combobox", { name: "Summarizer model" }),
    ).toHaveValue("alpha");
    expect(
      screen.getByRole("button", { name: "Save model defaults" }),
    ).toBeDisabled();
    expect(
      requests.filter(
        (request) =>
          request.method === "PUT" &&
          new URL(request.url).pathname.endsWith("/models"),
      ),
    ).toHaveLength(1);
  });

  it("preserves a draft on a stale revision and reloads the saved value before reset", async () => {
    let changed = false;
    renderApplication("/operations/settings", {
      capabilities: ["user", "operations"],
      server: {
        ...common,
        "GET /v1/operations/settings/models": () =>
          response(
            { defaultModel: "alpha", roles: changed ? { worker: "beta" } : {} },
            changed ? "2" : "1",
          ),
        "PUT /v1/operations/settings/models": () => {
          changed = true;
          return preconditionFailed();
        },
      },
    });
    const model = await screen.findByRole("combobox", {
      name: "Default model",
    });
    fireEvent.change(model, { target: { value: "beta" } });
    fireEvent.click(
      screen.getByRole("button", { name: "Save model defaults" }),
    );
    expect(
      await screen.findByText(/Model settings changed on the server/),
    ).toBeInTheDocument();
    expect(model).toHaveValue("beta");
    expect(
      screen.getByRole("button", { name: "Save model defaults" }),
    ).toBeDisabled();
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Refresh from server" }),
      ).not.toBeDisabled(),
    );
    fireEvent.click(
      screen.getByRole("button", { name: "Use saved selection" }),
    );
    expect(model).toHaveValue("alpha");
    expect(screen.getByRole("combobox", { name: "Worker model" })).toHaveValue(
      "beta",
    );
  });
});
