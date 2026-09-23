import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { PublicAPIError } from "../../api/error";
import { MemoryStorage } from "../../test/storage";
import {
  finishMutation,
  mutationKey,
  RECOVERY_STORAGE_MESSAGE,
  recoverableMutation,
} from "./recovery";

class BlockedStorage extends MemoryStorage {
  override getItem(): string | null {
    throw new DOMException("blocked", "SecurityError");
  }
  override setItem(): void {
    throw new DOMException("blocked", "SecurityError");
  }
  override removeItem(): void {
    throw new DOMException("blocked", "SecurityError");
  }
}

const apiError = (status: number) =>
  new PublicAPIError({ status, code: "failed", message: "Request failed" });

beforeEach(() => {
  vi.stubGlobal("localStorage", new MemoryStorage());
});
afterEach(() => {
  vi.unstubAllGlobals();
});

describe("Eval mutation recovery", () => {
  it("reports unavailable browser storage before sending", async () => {
    vi.stubGlobal("localStorage", new BlockedStorage());
    const send = vi.fn();
    await expect(mutationKey("owner", "dataset", { a: 1 })).rejects.toThrow(
      RECOVERY_STORAGE_MESSAGE,
    );
    await expect(
      recoverableMutation("owner", "dataset", { a: 1 }, send),
    ).rejects.toThrow(RECOVERY_STORAGE_MESSAGE);
    expect(send).not.toHaveBeenCalled();
    await expect(
      finishMutation("owner", "dataset", { a: 1 }),
    ).resolves.toBeUndefined();
  });

  it("clears the key after a permanent client error", async () => {
    const request = { a: 1 };
    const keys: string[] = [];
    const fail = (error: Error) => (key: string) => {
      keys.push(key);
      return Promise.reject(error);
    };
    await expect(
      recoverableMutation("owner", "dataset", request, fail(apiError(409))),
    ).rejects.toThrow("Request failed");
    expect(localStorage.length).toBe(0);
    await expect(
      recoverableMutation("owner", "dataset", request, fail(apiError(503))),
    ).rejects.toThrow("Request failed");
    await expect(
      recoverableMutation("owner", "dataset", request, fail(apiError(429))),
    ).rejects.toThrow("Request failed");
    expect(localStorage.length).toBe(1);
    expect(keys[2]).toBe(keys[1]);
    expect(keys[1]).not.toBe(keys[0]);
    await expect(
      recoverableMutation("owner", "dataset", request, async (key) => key),
    ).resolves.toBe(keys[1]);
    expect(localStorage.length).toBe(0);
  });
});
