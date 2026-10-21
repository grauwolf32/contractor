import { describe, expect, it } from "vitest";
import { readAuditArtifact } from "./audit-artifacts";
import { fakeServer } from "../test/fake-server";
import { bytes, failure } from "../test/responses";
import { testAPI } from "../test/render";

describe("exact Audit receipt content", () => {
  it("pins the receipt revision and package member", async () => {
    const seen: URL[] = [];
    const { api } = testAPI(
      fakeServer({
        "GET /v1/audits/{auditId}/artifacts": ({ url }) => {
          seen.push(url);
          return bytes('{"observations":[]}', "application/json");
        },
      }),
    );
    const read = await readAuditArtifact(
      api,
      "audit_one",
      "result/execution",
      "r1",
      "scanner-report",
      new AbortController().signal,
    );
    expect(await read.blob.text()).toBe('{"observations":[]}');
    expect(read.mediaType).toBe("application/json");
    expect(seen[0]!.searchParams.get("key")).toBe("result/execution");
    expect(seen[0]!.searchParams.get("revision")).toBe("r1");
    expect(seen[0]!.searchParams.get("member")).toBe("scanner-report");
  });

  it.each([undefined, "67108865", "-1", "invalid", "1", "100"])(
    "rejects incomplete or invalid byte lengths: %s",
    async (length) => {
      const { api } = testAPI(
        fakeServer({
          "GET /v1/audits/{auditId}/artifacts": () => {
            const response = bytes("saved", "text/plain");
            if (length === undefined) response.headers.delete("Content-Length");
            else response.headers.set("Content-Length", length);
            return response;
          },
        }),
      );
      await expect(
        readAuditArtifact(
          api,
          "audit_one",
          "prepare:execution:api",
          "r1",
          undefined,
          new AbortController().signal,
        ),
      ).rejects.toMatchObject({ code: "invalid_api_response" });
    },
  );

  it("retains the classified storage failure without replaying a mutation", async () => {
    let calls = 0;
    const { api } = testAPI(
      fakeServer({
        "GET /v1/audits/{auditId}/artifacts": () => {
          calls += 1;
          return failure(
            503,
            "storage_transaction_conflict",
            "Retry the read.",
            true,
          );
        },
      }),
    );
    await expect(
      readAuditArtifact(
        api,
        "audit_one",
        "result/execution",
        "r1",
        undefined,
        new AbortController().signal,
      ),
    ).rejects.toMatchObject({ code: "storage_transaction_conflict" });
    expect(calls).toBe(1);
  });
});
