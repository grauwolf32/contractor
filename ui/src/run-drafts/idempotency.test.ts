import { describe, expect, it, vi } from "vitest";

import type { CreateRunRequest } from "../api/workflows";
import { canonicalRunRequest, RunDraftKeyring } from "./idempotency";

describe("Run draft idempotency", () => {
  it("canonicalizes object key order without coercing values", () => {
    const first: CreateRunRequest = {
      workflow: "workflow@1",
      labels: { purpose: "eval", "eval.id": "eval-01" },
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
      labels: { "eval.id": "eval-01", purpose: "eval" },
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

  it("rotates the key when one immutable metadata label changes", () => {
    const generate = vi
      .fn<() => string>()
      .mockReturnValueOnce("run-label-1")
      .mockReturnValueOnce("run-label-2");
    const keyring = new RunDraftKeyring(generate);
    const first: CreateRunRequest = {
      workflow: "workflow@1",
      labels: { "eval.id": "eval-01", "eval.leg": "a" },
    };
    const reordered: CreateRunRequest = {
      labels: { "eval.leg": "a", "eval.id": "eval-01" },
      workflow: "workflow@1",
    };
    const changed: CreateRunRequest = {
      workflow: "workflow@1",
      labels: { "eval.id": "eval-01", "eval.leg": "b" },
    };

    expect(keyring.keyFor(first)).toBe("run-label-1");
    expect(keyring.keyFor(reordered)).toBe("run-label-1");
    expect(keyring.keyFor(changed)).toBe("run-label-2");
  });

  it("never reuses one key across standalone and Project endpoints", () => {
    const generate = vi
      .fn<() => string>()
      .mockReturnValueOnce("run-endpoint-1")
      .mockReturnValueOnce("run-endpoint-2");
    const keyring = new RunDraftKeyring(generate);
    const request: CreateRunRequest = { workflow: "workflow@1" };

    expect(keyring.keyFor(request, "standalone")).toBe("run-endpoint-1");
    expect(keyring.keyFor(request, "project:project_example")).toBe(
      "run-endpoint-2",
    );
    expect(keyring.matches(request, "standalone")).toBe(false);
    expect(keyring.matches(request, "project:project_example")).toBe(true);
  });

  it("rejects an unsafe key generator", () => {
    const keyring = new RunDraftKeyring(() => "bad key");
    expect(() => keyring.keyFor({ workflow: "workflow@1" })).toThrow(
      "idempotency key",
    );
  });
});
