import { screen } from "@testing-library/react";
import { expect, it } from "vitest";
import { renderApplication } from "../../test/application";
import {
  newerPresetFixture,
  presetFixture,
} from "../../test/audit-presets-fixture";
import { fakeServer } from "../../test/fake-server";
import { json, page } from "../../test/responses";

it("lists every published Program version across catalog pages", async () => {
  const server = fakeServer({
    "GET /v1/audit-profiles/{name}/versions/{version}": ({ params }) =>
      json(params.version === "10" ? newerPresetFixture : presetFixture, 200, {
        ETag: `"${(params.version === "10" ? newerPresetFixture : presetFixture).ref.digest}"`,
      }),
    "GET /v1/audit-profiles": ({ url }) =>
      json(
        page(
          url.searchParams.has("cursor")
            ? [newerPresetFixture]
            : [presetFixture],
          url.searchParams.has("cursor") ? undefined : "second",
        ),
      ),
  });
  renderApplication("/catalog/audit-presets", { server });
  expect(await screen.findByRole("option", { name: "10" })).toBeInTheDocument();
  expect(screen.queryByText(/Unsupported/)).not.toBeInTheDocument();
});
it("shows the common input contract and actual Workflow binding", async () => {
  const server = fakeServer({
    "GET /v1/audit-profiles": () => json(page([presetFixture])),
    "GET /v1/audit-profiles/{name}/versions/{version}": () =>
      json(presetFixture, 200, { ETag: `"${presetFixture.ref.digest}"` }),
  });
  renderApplication("/catalog/audit-presets/source-review/1", { server });
  expect(
    await screen.findByRole("link", { name: /source-check@3/ }),
  ).toBeInTheDocument();
  expect(screen.getByText("Required")).toBeInTheDocument();
  expect(screen.queryByText("Inventory")).not.toBeInTheDocument();
});
