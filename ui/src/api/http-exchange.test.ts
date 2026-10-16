import { describe, expect, it } from "vitest";
import captures from "../../../api/testdata/http-capture-cases.json";
import attempts from "../../../internal/auditdomain/testdata/finding-http-attempts.json";
import {
  canPreviewArtifact,
  HTTP_EXCHANGE_MEDIA_TYPE,
  MAXIMUM_PREVIEW_BYTES,
} from "./artifacts";
import { parseHTTPExchange } from "./http-exchange";

/** The selected-exchange artifact format the preview reads. */
const HTTP_EXCHANGE_SCHEMA = "contractor.http.exchange.v1";

/** Whether the preview accepts `capture` inside a selected-exchange artifact. */
function acceptsCapture(capture: unknown): boolean {
  return (
    parseHTTPExchange(
      JSON.stringify({
        schema: HTTP_EXCHANGE_SCHEMA,
        capture,
        finding: { proposal_id: "proposal-test", receipt_id: "receipt-test" },
      }),
    ) !== undefined
  );
}

describe("shared HTTP finding capture contract", () => {
  it.each(attempts)("matches the retained URL/status rules: $name", (item) => {
    const value = structuredClone(captures.cases[0]!.value);
    const attempt = value.attempts[0]!;
    if ("request" in attempt && attempt.request) {
      attempt.request.url =
        item.url +
        ("repeat" in item
          ? String(item.repeat).repeat(Number(item.count))
          : "");
    }
    if ("response" in attempt && attempt.response)
      attempt.response.status = item.status;
    expect(acceptsCapture(value)).toBe(item.valid);
  });
  it.each(captures.cases)("$name", ({ value, valid }) => {
    expect(acceptsCapture(value)).toBe(valid);
  });
  // The shared bounds, read from the fixture every language checks. Bodies
  // larger than a preview (LIM-UI-12) never reach the capture check, so the
  // body bounds (LIM-FND-7) are exercised by the Go and Python suites.
  it("applies the shared attempt bound (LIM-FND-9)", () => {
    const capture = structuredClone(captures.cases[0]!.value) as {
      attempts: unknown[];
    };
    const one = capture.attempts[0];
    capture.attempts = Array.from({ length: captures.limits.attempts }, () =>
      structuredClone(one),
    );
    expect(acceptsCapture(capture)).toBe(true);
    capture.attempts.push(structuredClone(one));
    expect(acceptsCapture(capture)).toBe(false);
  });
  it("applies the shared header block bound (LIM-FND-8)", () => {
    const capture = structuredClone(captures.cases[0]!.value) as {
      attempts: {
        request: { message: { headers: { name: string; value: string }[] } };
      }[];
    };
    const request = capture.attempts[0]!.request.message;
    const used = request.headers.reduce(
      (total, header) => total + header.name.length + header.value.length,
      0,
    );
    const filler = { name: "X-Fill", value: "" };
    request.headers.push(filler);
    filler.value = "v".repeat(
      captures.limits.headerBytes - used - filler.name.length,
    );
    expect(acceptsCapture(capture)).toBe(true);
    filler.value += "v";
    expect(acceptsCapture(capture)).toBe(false);
  });
});

const document = {
  schema: HTTP_EXCHANGE_SCHEMA,
  capture: captures.cases[0]!.value,
  finding: { proposal_id: "proposal-test", receipt_id: "receipt-test" },
};
describe("selected HTTP evidence artifact", () => {
  it("accepts the closed current artifact", () => {
    expect(parseHTTPExchange(JSON.stringify(document))).toEqual(document);
  });
  it("rejects unknown versions, fields and missing finding association", () => {
    for (const invalid of [
      { ...document, schema: "contractor.http.exchange.v0" },
      { ...document, unexpected: true },
      { ...document, finding: null },
      { ...document, finding: { ...document.finding, receipt_id: null } },
      {
        ...document,
        finding: { ...document.finding, proposal_id: "proposal\n" },
      },
      { ...document, capture: { ...document.capture, source: ["http"] } },
    ])
      expect(parseHTTPExchange(JSON.stringify(invalid))).toBeUndefined();
  });
  it("rejects repeated keys, trailing JSON and an oversized preview", () => {
    const source = JSON.stringify(document);
    expect(
      parseHTTPExchange(
        source.replace('"source":"http"', '"source":"http","source":"http"'),
      ),
    ).toBeUndefined();
    expect(parseHTTPExchange(source + "{}")).toBeUndefined();
    expect(
      parseHTTPExchange(" ".repeat(MAXIMUM_PREVIEW_BYTES) + source),
    ).toBeUndefined();
  });
  it("keeps preview admission bounded before any payload read", () => {
    const metadata = {
      artifact: {
        namespace: "finding-proposals",
        name: "http.exchange.proposal-test",
        revision: "exact",
      },
      mediaType: HTTP_EXCHANGE_MEDIA_TYPE,
      current: true,
      frozen: true,
      size: MAXIMUM_PREVIEW_BYTES,
      createdAt: "2026-10-13T00:00:00Z",
    };
    expect(canPreviewArtifact(metadata)).toBe(true);
    expect(
      canPreviewArtifact({ ...metadata, size: MAXIMUM_PREVIEW_BYTES + 1 }),
    ).toBe(false);
  });
});
