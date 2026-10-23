import { describe, expect, it, vi } from "vitest";

import cases from "../../../api/testdata/public/idempotency-key-cases.json";
import { IDEMPOTENCY_KEY, requestHeaderViolation } from "../api/contract";
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

describe("shared Idempotency-Key cases", () => {
  const valid = [...cases.valid];
  const invalid = [...cases.invalid];
  for (const last of ["a", ".", "_", ":", "-"]) {
    const key = "a".repeat(cases.maxLength - 1) + last;
    valid.push(key);
    invalid.push(`a${key}`);
  }
  const operations = [
    "POST /v1/projects",
    "POST /v1/projects/{projectId}/eval-datasets",
    "POST /v1/eval-experiments/{id}/commands",
  ] as const;

  it("gives every shared key the Server's verdict before sending it", () => {
    expect(IDEMPOTENCY_KEY.maxLength).toBe(cases.maxLength);
    for (const key of valid) {
      expect(IDEMPOTENCY_KEY.test(key), key).toBe(true);
      expect(new MutationDraftKeyring("probe", () => key).keyFor({}), key).toBe(
        key,
      );
      for (const operation of operations) {
        expect(
          requestHeaderViolation(operation, "Idempotency-Key", key),
          `${operation} ${key}`,
        ).toBeUndefined();
      }
    }
    for (const key of invalid) {
      expect(IDEMPOTENCY_KEY.test(key), key).toBe(false);
      expect(
        () => new MutationDraftKeyring("probe", () => key).keyFor({}),
        key,
      ).toThrow("idempotency key");
      for (const operation of operations) {
        expect(
          requestHeaderViolation(operation, "Idempotency-Key", key),
          `${operation} ${key}`,
        ).toBeDefined();
      }
    }
  });
});
