import { describe, expect, it, vi } from "vitest";

import type { CreateRunRequest } from "../api/workflows";
import { canonicalRunRequest, RunDraftKeyring } from "./idempotency";

describe("Run draft idempotency", () => {
  it("canonicalizes object key order without coercing values", () => {
    const first: CreateRunRequest = {
      workflow: "workflow@1",
      parameters: { z: "", a: "value" },
      artifacts: {
        source: { namespace: "projects", name: "source", revision: "r1" },
      },
    };
    const second: CreateRunRequest = {
      artifacts: {
        source: { revision: "r1", name: "source", namespace: "projects" },
      },
      parameters: { a: "value", z: "" },
      workflow: "workflow@1",
    };
    expect(canonicalRunRequest(first)).toBe(canonicalRunRequest(second));
    expect(canonicalRunRequest(first)).toContain('"z":""');
  });

  it("reuses a key only for the exact canonical request", () => {
    const generate = vi
      .fn<() => string>()
      .mockReturnValueOnce("run-draft-1")
      .mockReturnValueOnce("run-draft-2");
    const keyring = new RunDraftKeyring(generate);
    const first: CreateRunRequest = {
      workflow: "workflow@1",
      parameters: { objective: "first" },
    };
    const equivalent: CreateRunRequest = {
      parameters: { objective: "first" },
      workflow: "workflow@1",
    };
    const changed: CreateRunRequest = {
      workflow: "workflow@1",
      parameters: { objective: "second" },
    };

    expect(keyring.keyFor(first)).toBe("run-draft-1");
    expect(keyring.keyFor(equivalent)).toBe("run-draft-1");
    expect(keyring.matches(equivalent)).toBe(true);
    expect(keyring.keyFor(changed)).toBe("run-draft-2");
    expect(keyring.matches(first)).toBe(false);
    expect(generate).toHaveBeenCalledTimes(2);
  });

  it("rejects an unsafe key generator", () => {
    const keyring = new RunDraftKeyring(() => "bad key");
    expect(() => keyring.keyFor({ workflow: "workflow@1" })).toThrow(
      "idempotency key",
    );
  });
});
