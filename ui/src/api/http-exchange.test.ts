import { describe, expect, it } from "vitest";
import captures from "../../../api/testdata/http-capture-cases.json";
import attempts from "../../../internal/auditdomain/testdata/finding-http-attempts.json";
import {
  canPreviewArtifact,
  HTTP_EXCHANGE_MEDIA_TYPE,
  MAXIMUM_PREVIEW_BYTES,
} from "./artifacts";
import {
  HTTP_CAPTURE_LIMITS,
  HTTP_EXCHANGE_SCHEMA,
  isHTTPCapture,
  parseHTTPExchange,
} from "./http-exchange";

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
    expect(isHTTPCapture(value)).toBe(item.valid);
  });
  it.each(captures.cases)("$name", ({ value, valid }) => {
    expect(isHTTPCapture(value)).toBe(valid);
  });
  it("pins shared body, header and attempt bounds", () => {
    for (const [key, value] of Object.entries(HTTP_CAPTURE_LIMITS)) {
      expect(captures.limits[key as keyof typeof captures.limits]).toBe(value);
    }
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
