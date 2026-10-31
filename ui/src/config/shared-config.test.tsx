import { act, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { fakeServer } from "../test/fake-server";
import { renderWithAPI } from "../test/render";
import { failure, json } from "../test/responses";
import { testSharedConfig } from "../test/shared-config";
import { useSharedConfig } from "./shared-config";
import { SharedConfigGate } from "../auth/shared-config-gate";
import { ArtifactFileDrop } from "../routes/artifacts/common";

function ProtectedMaterial() {
  const cap = useSharedConfig().limits.artifactPayloadMaxBytes;
  return (
    <>
      <h1>Protected material</h1>
      <p>Limit {cap}</p>
      <ArtifactFileDrop file={null} onSelect={() => undefined} />
    </>
  );
}

describe("UI policy bootstrap", () => {
  it("holds protected content until Server settings arrive, then displays their limit", async () => {
    let release!: (response: Response) => void;
    const server = fakeServer({
      "GET /v1/config": () =>
        new Promise<Response>((resolve) => {
          release = resolve;
        }),
    });
    renderWithAPI(
      <SharedConfigGate>
        <ProtectedMaterial />
      </SharedConfigGate>,
      { server },
    );
    expect(
      screen.queryByRole("heading", { name: "Protected material" }),
    ).not.toBeInTheDocument();
    await screen.findByRole("status");
    await act(async () => {
      release(json(testSharedConfig(1024 * 1024)));
    });
    expect(await screen.findByText("Limit 1048576")).toBeVisible();
    expect(
      screen.getByText(/choose one local file, up to 1 MiB/),
    ).toBeVisible();
    expect(server.sent("GET", "/v1/config")).toHaveLength(1);
  });

  it("blocks on failure and admits content only after explicit successful retry", async () => {
    const server = fakeServer({
      "GET /v1/config": () => failure(400, "invalid_request", "Try later"),
    });
    const { user } = renderWithAPI(
      <SharedConfigGate>
        <ProtectedMaterial />
      </SharedConfigGate>,
      { server },
    );
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Could not load Server settings",
    );
    expect(
      screen.queryByRole("heading", { name: "Protected material" }),
    ).not.toBeInTheDocument();
    server.route("GET /v1/config", () => json(testSharedConfig(1024 * 1024)));
    await user.click(screen.getByRole("button", { name: "Try again" }));
    expect(
      await screen.findByRole("heading", { name: "Protected material" }),
    ).toBeVisible();
    expect(server.sent("GET", "/v1/config")).toHaveLength(2);
  });
});
