import { webcrypto } from "node:crypto";
import { beforeAll, describe, expect, it, vi } from "vitest";

import { EXPORT_BYTES, publishedExport } from "../test/finding-export";
import { fakeServer } from "../test/fake-server";
import { testAPI } from "../test/render";
import { API_VERSION_HEADERS, bytes, json } from "../test/responses";
import {
  downloadFindingExport,
  publishFindingExport,
  type FindingExportRequest,
} from "./finding-exports";

beforeAll(() => vi.stubGlobal("crypto", webcrypto));
const selection: FindingExportRequest = {
  clientKey: "ui-export",
  format: "sarif",
  profile: "generic",
  includeDuplicates: false,
  findings: [
    { findingId: "one", revision: 1 },
    { findingId: "two", revision: 2 },
  ],
};

describe("Audit-local export transport", () => {
  it("validates exact explicit selections before making a request", async () => {
    const { api, server } = testAPI(fakeServer());
    await expect(
      publishFindingExport(api, "audit", {
        ...selection,
        findings: [selection.findings[0]!, selection.findings[0]!],
      }),
    ).rejects.toThrow("selection is invalid");
    expect(server.requests).toHaveLength(0);
  });
  it("sends the duplicate parameter and rejects inconsistent server counts", async () => {
    const published = await publishedExport();
    const { api, server } = testAPI(
      fakeServer({
        "POST /v1/audits/{auditId}/findings/exports": () =>
          json(published, 201),
      }),
    );
    expect(await publishFindingExport(api, "audit", selection)).toEqual(
      published,
    );
    expect(await server.requests[0]!.json()).toMatchObject({
      includeDuplicates: false,
    });
    await expect(
      publishFindingExport(api, "audit", {
        ...selection,
        includeDuplicates: true,
      }),
    ).rejects.toMatchObject({ code: "invalid_api_response" });
    expect(await server.requests[1]!.json()).toMatchObject({
      includeDuplicates: true,
    });
  });
  it("downloads all frozen parts and checks their exact bytes without another POST", async () => {
    const published = await publishedExport();
    const { api, server } = testAPI(
      fakeServer({}, () =>
        bytes(EXPORT_BYTES, "application/json", { ETag: '"r1"' }),
      ),
    );
    for (const part of ["content", "snapshot", "diagnostics"] as const) {
      const result = await downloadFindingExport(api, "audit", published, part);
      expect(result.blob.size).toBe(published.artifact.sizeBytes);
      expect(result.filename).toBe(
        part === "content" ? "audit-audit.sarif" : `audit-audit.${part}.json`,
      );
    }
    expect(server.requests.map((r) => r.method)).toEqual(["GET", "GET", "GET"]);
  });
  for (const variant of [
    "digest",
    "revision",
    "length",
    "stream-bound",
  ] as const) {
    it(`rejects changed ${variant} and cancels an oversized stream`, async () => {
      const published = await publishedExport();
      let cancelled = false;
      const stream = new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode(EXPORT_BYTES + "extra"));
        },
        cancel() {
          cancelled = true;
        },
      });
      const { api } = testAPI(
        fakeServer({}, () =>
          variant === "stream-bound"
            ? new Response(stream, {
                headers: {
                  ...API_VERSION_HEADERS,
                  "Content-Type": "application/json",
                  ETag: '"r1"',
                },
              })
            : bytes(
                variant === "digest"
                  ? EXPORT_BYTES.replace("2.1.0", "2.1.1")
                  : EXPORT_BYTES,
                "application/json",
                {
                  ETag: variant === "revision" ? '"r2"' : '"r1"',
                  ...(variant === "length" ? { "Content-Length": "0" } : {}),
                },
              ),
        ),
      );
      await expect(
        downloadFindingExport(api, "audit", published, "content"),
      ).rejects.toMatchObject({ code: "invalid_api_response" });
      if (variant === "stream-bound") expect(cancelled).toBe(true);
    });
  }
});
