import { screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { EvalVariant } from "../../api/evals";
import { presetFixture } from "../../test/audit-presets-fixture";
import { workflowResource } from "../../test/builders-execution";
import { fakeServer } from "../../test/fake-server";
import { renderWithAPI } from "../../test/render";
import { json } from "../../test/responses";
import { VariantEditor } from "./variants";

describe("Eval variant input contracts", () => {
  for (const kind of ["workflow", "audit"] as const) {
    it(`renders the current ${kind} input contract without legacy fields`, async () => {
      const selector = kind === "audit" ? "source-review@1" : "trace-a@1";
      const value: EvalVariant = { id: "a", kind, selector };
      const onChange = vi.fn();
      const server = fakeServer({
        "GET /v1/audit-profiles/{name}/versions/{version}": () =>
          json(presetFixture, 200, { ETag: `"${presetFixture.ref.digest}"` }),
        "GET /v1/workflows/{name}/versions/{version}": () =>
          json(
            workflowResource("trace-a", "1", {
              inputs: {
                source: { required: true, mediaTypes: ["application/zip"] },
                notes: { required: false, mediaTypes: ["text/plain"] },
              },
            }),
          ),
      });
      renderWithAPI(
        <VariantEditor
          value={value}
          onChange={onChange}
          capabilities={undefined}
          label="A"
        />,
        { server },
      );
      const source = await screen.findByText("source", { exact: true });
      expect(source.closest("p")).toHaveTextContent(
        "Input source: required · application/zip",
      );
      expect(screen.getByLabelText("A version", { exact: true })).toHaveValue(
        selector,
      );
      if (kind === "workflow")
        expect(
          screen.getByText("notes", { exact: true }).closest("p"),
        ).toHaveTextContent("Input notes: optional · text/plain");
      expect(onChange).not.toHaveBeenCalled();
      expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    });
  }
});
