import { describe, expect, it, vi } from "vitest";

import { canonicalMutationRequest, MutationDraftKeyring } from "./idempotency";

describe("mutation draft idempotency", () => {
  it("reuses a key only for one exact canonical mutation", () => {
    const generate = vi
      .fn<() => string>()
      .mockReturnValueOnce("config-ui-1")
      .mockReturnValueOnce("config-ui-2");
    const keys = new MutationDraftKeyring<{ name: string; body: unknown }>(
      "config",
      generate,
    );
    const first = { name: "worker", body: { model: "m", limit: 2 } };
    const equivalent = { body: { limit: 2, model: "m" }, name: "worker" };
    const changed = { name: "worker", body: { model: "m", limit: 3 } };
    expect(keys.keyFor(first)).toBe("config-ui-1");
    expect(keys.keyFor(equivalent)).toBe("config-ui-1");
    expect(keys.matches(equivalent)).toBe(true);
    expect(keys.keyFor(changed)).toBe("config-ui-2");
    expect(generate).toHaveBeenCalledTimes(2);
  });

  it("omits undefined fields without coercing arrays or empty values", () => {
    expect(canonicalMutationRequest({ z: undefined, a: ["", 0, false] })).toBe(
      '{"a":["",0,false]}',
    );
  });

  it("rejects an unsafe generated key", () => {
    const keys = new MutationDraftKeyring("credential", () => "bad key");
    expect(() => keys.keyFor({ id: "one" })).toThrow("idempotency key");
  });
});
