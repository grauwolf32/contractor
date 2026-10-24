import { describe, expect, it } from "vitest";
import { fakeServer, invalidJSON } from "../test/fake-server";
import { testAPI } from "../test/render";
import { json, noContent, page } from "../test/responses";
import { scanSweep, sweepMember } from "../test/web-scans";
import {
  createScanSweep,
  getScanSweep,
  listScanSweepMembers,
  MAX_SWEEP_PROJECTS,
  purgeScanSweep,
} from "./scan-sweeps";

describe("durable sweep API adapter", () => {
  it("accepts the full scope boundary and rejects larger or empty/ambiguous scopes before writing", async () => {
    const server = fakeServer({
      "POST /v1/scan-sweeps": () => json(scanSweep(), 201),
      "DELETE /v1/scan-sweeps/{sweepId}": () => noContent(),
    });
    const { api } = testAPI(server);
    const ids = Array.from({ length: MAX_SWEEP_PROJECTS }, (_, i) => `p${i}`);
    await createScanSweep(api, { projectIds: ids }, "same-key");
    for (const request of [
      {},
      { projectIds: [...ids, "extra"] },
      { allProjects: true, projectIds: ["p1"] },
      { allProjects: true, parameters: { target: "https://other.example" } },
    ])
      await expect(createScanSweep(api, request, "same-key")).rejects.toThrow();
    expect(server.sent("POST", "/scan-sweeps")).toHaveLength(1);
    expect(
      server.sent("POST", "/scan-sweeps")[0]?.headers.get("Idempotency-Key"),
    ).toBe("same-key");
    await purgeScanSweep(api, "sweep-one");
    expect(server.sent("DELETE", "/sweep-one")).toHaveLength(1);
  });
  it("refuses inconsistent identities, counters, members and oversized bodies", async () => {
    const server = fakeServer({
      "GET /v1/scan-sweeps/{sweepId}": () =>
        json({ ...scanSweep(), sweepId: "another" }),
      "GET /v1/scan-sweeps/{sweepId}/members": () =>
        json(page([sweepMember(0), sweepMember(0)])),
    });
    const { api } = testAPI(server);
    await expect(getScanSweep(api, "sweep-one")).rejects.toMatchObject({
      code: "invalid_api_response",
    });
    await expect(listScanSweepMembers(api, "sweep-one")).rejects.toMatchObject({
      code: "invalid_api_response",
    });
    server.route("GET /v1/scan-sweeps/{sweepId}", () =>
      json({ ...scanSweep(), counts: { ...scanSweep().counts, total: 3 } }),
    );
    await expect(getScanSweep(api, "sweep-one")).rejects.toMatchObject({
      code: "invalid_api_response",
    });
    server.route("GET /v1/scan-sweeps/{sweepId}", () =>
      invalidJSON(" ".repeat(4 * 1024 * 1024 + 1)),
    );
    await expect(getScanSweep(api, "sweep-one")).rejects.toMatchObject({
      code: "invalid_api_response",
    });
  });
});
