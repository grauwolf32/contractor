import { describe, expect, it } from "vitest";
import { fireEvent, screen, waitFor } from "@testing-library/react";
import { renderStart } from "./test-support";
import {
  profileFixture,
  sourceZip,
  withExecutionParameters,
} from "./test-fixtures";

const profile = withExecutionParameters(profileFixture("parameter-scan"), {
  server: { required: true },
  operations: { required: true },
  credential_id: { required: false },
});

describe("Pinned Audit execution parameters", () => {
  it("requires declared values and submits selection and credential ID without a settings artifact", async () => {
    const view = renderStart(
      "/checks/new?project=project_shop&type=parameter-scan",
      {
        profiles: [profile],
        materials: [sourceZip],
      },
    );
    const server = await screen.findByLabelText(/API server URL/);
    expect(
      screen.getByRole("button", { name: "Save as draft" }),
    ).toBeDisabled();
    fireEvent.change(server, { target: { value: "https://api.example.test" } });
    fireEvent.change(screen.getByLabelText(/Selected operations/), {
      target: { value: '["#/paths/~1pets/get"]' },
    });
    fireEvent.change(screen.getByLabelText(/Credential ID/), {
      target: { value: "scan-account" },
    });
    await view.user.click(
      screen.getByRole("button", { name: "Save as draft" }),
    );
    await waitFor(() =>
      expect(view.requests.some((r) => r.method === "POST")).toBe(true),
    );
    const request = view.requests.find((r) => r.method === "POST")!;
    expect(await request.clone().json()).toMatchObject({
      parameters: {
        server: "https://api.example.test",
        operations: '["#/paths/~1pets/get"]',
        credential_id: "scan-account",
      },
      inputs: { source: sourceZip.artifact },
    });
    expect(view.requests.some((r) => r.method === "PUT")).toBe(false);
  });

  it("blocks values over the UTF-8 byte limit", async () => {
    renderStart("/checks/new?project=project_shop&type=parameter-scan", {
      profiles: [profile],
      materials: [sourceZip],
    });
    fireEvent.change(await screen.findByLabelText(/API server URL/), {
      target: { value: "é".repeat(2049) },
    });
    fireEvent.change(screen.getByLabelText(/Selected operations/), {
      target: { value: '["#/paths/~1pets/get"]' },
    });
    expect(
      screen.getByRole("button", { name: "Save as draft" }),
    ).toBeDisabled();
  });
});
