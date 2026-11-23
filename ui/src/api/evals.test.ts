import { describe, expect, it } from "vitest";

import { createEvalFixture } from "../test/evals-fixture";
import { PublicAPI } from "./client";
import { saveEvalDraft } from "./evals";

/** A PublicAPI over the shared Eval fixture, signed in as its owner. */
function evalAPI(fixture: ReturnType<typeof createEvalFixture>): PublicAPI {
  const api = new PublicAPI(
    {
      uiVersion: "0.1.0",
      supportedApiVersions: ["contractor.public.v1"],
      apiBaseUrl: "http://127.0.0.1:8080",
    },
    (input) =>
      fixture.fetch(input instanceof Request ? input : new Request(input)),
  );
  api.csrf.replace("a".repeat(43));
  return api;
}

describe("Eval mutation headers", () => {
  it("sends If-Match only with a revision", async () => {
    const fixture = createEvalFixture();
    const api = evalAPI(fixture);
    const experiment = fixture.state.experiment;
    const body = { name: "Regression", draft: experiment.draft! };
    await saveEvalDraft(api, "evaluation-1", body, "eval-key");
    expect(fixture.state.requests.at(-1)).toMatchObject({
      method: "POST",
      key: "eval-key",
      etag: null,
    });
    const revision = experiment.revision;
    await saveEvalDraft(api, "evaluation-1", body, "eval-key-2", {
      id: experiment.experimentId,
      revision,
    });
    expect(fixture.state.requests.at(-1)).toMatchObject({
      method: "PATCH",
      key: "eval-key-2",
      etag: `"${revision}"`,
    });
  });

  it("rejects header values that fail validation before sending", async () => {
    const fixture = createEvalFixture();
    const sent = fixture.state.requests.length;
    await expect(
      saveEvalDraft(
        evalAPI(fixture),
        "evaluation-1",
        { name: "Regression", draft: fixture.state.experiment.draft! },
        "eval key",
      ),
    ).rejects.toThrow();
    expect(fixture.state.requests).toHaveLength(sent);
  });
});
