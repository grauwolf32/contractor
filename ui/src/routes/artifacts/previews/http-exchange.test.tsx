import { fireEvent, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import captures from "../../../../../api/testdata/http-capture-cases.json";
import { HTTP_EXCHANGE_MEDIA_TYPE } from "../../../api/artifacts";
import type { HTTPExchangeDocument } from "../../../api/http-exchange";
import HTTPExchangePreview from "./http-exchange";
import { saveBlob } from "../../../app/download";
import { LoadedArtifactPreview } from "../loaded-preview";
import { renderWithAPI } from "../../../test/render";
import { ArtifactPreviewPanel } from "../preview";

vi.mock("../../../app/download", () => ({ saveBlob: vi.fn() }));

function document(
  name = "received_http_error_is_response",
): HTTPExchangeDocument {
  return {
    schema: "contractor.http.exchange.v1",
    capture: structuredClone(
      captures.cases.find((item) => item.name === name)!.value,
    ) as HTTPExchangeDocument["capture"],
    finding: { proposal_id: "proposal-test", receipt_id: "receipt-test" },
  };
}
function preview(value = document(), trusted = true) {
  return renderWithAPI(
    <HTTPExchangePreview
      document={value}
      context={{
        scope: { kind: "run", id: "run-test" },
        metadata: {
          artifact: {
            namespace: trusted ? "finding-proposals" : "worker",
            name: "http.exchange.proposal-test",
            revision: "exact",
          },
          mediaType: HTTP_EXCHANGE_MEDIA_TYPE,
          size: 1000,
          current: true,
          frozen: trusted,
          createdAt: "2026-10-13T00:00:00Z",
        },
      }}
    />,
  );
}
describe("HTTP exchange preview", () => {
  it("does not diagnose a transport failure from an unavailable Caido response", async () => {
    preview(document("caido_without_response"));
    await userEvent.click(screen.getByRole("tab", { name: "Response" }));
    expect(screen.getByText("No response captured.")).toBeVisible();
    expect(
      screen.getByText("Response was unavailable when captured"),
    ).toBeVisible();
    expect(screen.queryByText("Transport failed")).not.toBeInTheDocument();
  });
  it("shows exact request bytes, readable JSON and a Server-owned finding association", async () => {
    const value = document();
    value.capture.attempts[0]!.request.message = {
      kind: "structured",
      headers: [
        { name: "X-Duplicate", value: "first" },
        { name: "X-Duplicate", value: "second" },
      ],
      body_base64: btoa('{"hello":"world"}'),
    };
    preview(value);
    expect(
      screen.getByRole("link", { name: "Open possible issue" }),
    ).toHaveAttribute(
      "href",
      "/runs/run-test/artifacts/finding-proposals/proposal-test",
    );
    expect(
      within(screen.getByRole("tabpanel", { name: "Request" })).getAllByText(
        "X-Duplicate",
      ),
    ).toHaveLength(2);
    expect(screen.getByText(/"hello": "world"/).textContent).toBe(
      '{\n  "hello": "world"\n}',
    );
    await userEvent.click(screen.getByRole("tab", { name: "Response" }));
    expect(screen.getByText("HTTP 422 · Complete response")).toBeVisible();
  });
  it("does not infer a finding relationship from names in an ordinary namespace", () => {
    preview(document(), false);
    expect(
      screen.queryByRole("link", { name: "Open possible issue" }),
    ).not.toBeInTheDocument();
  });
  it.each(["transport_error", "timeout", "cancelled"])(
    "labels absent response: %s",
    async (name) => {
      preview(document(name));
      await userEvent.click(screen.getByRole("tab", { name: "Response" }));
      expect(screen.getByText("No response captured.")).toBeVisible();
      expect(screen.queryByText(/Complete response/)).not.toBeInTheDocument();
    },
  );
  it("labels an incomplete response without claiming a complete body", async () => {
    preview(document("incomplete_response"));
    await userEvent.click(screen.getByRole("tab", { name: "Response" }));
    expect(screen.getByText(/Incomplete response/)).toBeVisible();
    expect(
      screen.getByText(/saved body may be absent or partial/),
    ).toBeVisible();
  });
  it("distinguishes an empty response body from an absent response", async () => {
    preview(document("empty_response_body"));
    await userEvent.click(screen.getByRole("tab", { name: "Response" }));
    expect(
      within(screen.getByRole("tabpanel", { name: "Response" })).getByText(
        "Empty body (0 bytes).",
      ),
    ).toBeVisible();
    expect(screen.queryByText("No response captured.")).not.toBeInTheDocument();
  });
  it("preserves binary response bytes in the body download", async () => {
    const value = document("binary_response");
    preview(value);
    await userEvent.click(screen.getByRole("tab", { name: "Response" }));
    expect(screen.getByText(/Binary body/)).toBeVisible();
    await userEvent.click(
      screen.getByRole("button", { name: "Download response body" }),
    );
    const [blob, filename] = vi.mocked(saveBlob).mock.calls.at(-1)!;
    expect(blob.size).toBe(
      atob(
        (
          value.capture.attempts[0]!.response!.message as {
            body_base64: string;
          }
        ).body_base64,
      ).length,
    );
    expect(filename).toBe("response-body.bin");
  });
  it("renders raw Caido headers separately from binary bodies", async () => {
    const value = document("caido_raw");
    value.capture.attempts[0]!.response!.message = {
      kind: "raw",
      data_base64: btoa(
        "HTTP/1.1 403 Forbidden\r\nX-Test: value\r\n\r\n\x00\xff",
      ),
    };
    preview(value);
    await userEvent.click(screen.getByRole("tab", { name: "Response" }));
    expect(screen.getByText(/HTTP\/1.1 403 Forbidden/)).toBeVisible();
    expect(screen.getByText(/Binary body/)).toBeVisible();
    await userEvent.click(
      screen.getByRole("button", { name: "Download raw response" }),
    );
    expect(vi.mocked(saveBlob).mock.calls.at(-1)![1]).toBe("response.http");
  });
  it("keeps HTML as source and bounds body text", () => {
    const value = document();
    value.capture.attempts[0]!.request.message = {
      kind: "structured",
      headers: [],
      body_base64: btoa(
        '<img src="https://leak.invalid/"><script>window.bad = true</script>' +
          "x".repeat(9000),
      ),
    };
    const view = preview(value);
    expect(view.container.querySelector("img, script, iframe")).toBeNull();
    expect(screen.getByText(/Showing the first 8192 characters/)).toBeVisible();
    expect(
      within(screen.getByRole("tabpanel", { name: "Request" })).getByText(
        /<img src=/,
      ).textContent!.length,
    ).toBe(8192);
  });
  it("supports arrow-key message tabs and pages duplicate headers", async () => {
    const value = document();
    value.capture.attempts[0]!.request.message = {
      kind: "structured",
      headers: Array.from({ length: 120 }, (_, i) => ({
        name: "X-Item",
        value: String(i),
      })),
      body_base64: "",
    };
    preview(value);
    expect(
      within(screen.getByRole("tabpanel", { name: "Request" })).getAllByText(
        "X-Item",
      ),
    ).toHaveLength(50);
    await userEvent.click(screen.getByRole("button", { name: "Next headers" }));
    expect(screen.getByText("51–100 of 120")).toBeVisible();
    const request = screen.getByRole("tab", { name: "Request" });
    request.focus();
    fireEvent.keyDown(request, { key: "ArrowRight" });
    expect(screen.getByRole("tab", { name: "Response" })).toHaveFocus();
    expect(screen.getByRole("tabpanel", { name: "Response" })).toBeVisible();
  });
  it("shows a bounded malformed-artifact error", async () => {
    renderWithAPI(
      <LoadedArtifactPreview
        mediaType={HTTP_EXCHANGE_MEDIA_TYPE}
        source='{"bad":true}'
      />,
    );
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "invalid or unsupported",
    );
  });
  it("explains an oversized preview without fetching its payload", () => {
    const load = vi.fn();
    renderWithAPI(
      <ArtifactPreviewPanel
        metadata={{
          artifact: {
            namespace: "finding-proposals",
            name: "http.exchange.proposal-test",
            revision: "exact",
          },
          mediaType: HTTP_EXCHANGE_MEDIA_TYPE,
          size: 256 * 1024 + 1,
          current: true,
          frozen: true,
          createdAt: "2026-10-13T00:00:00Z",
        }}
        loadPreview={load}
        loadOnMountKey={["oversized-http"]}
        unavailableCopy="Download the saved exchange"
      />,
    );
    expect(screen.getByText(/exceeds the 256 KiB preview limit/)).toBeVisible();
    expect(load).not.toHaveBeenCalled();
  });
});
