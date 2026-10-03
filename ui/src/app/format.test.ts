import { describe, expect, it } from "vitest";

import { compactDigest, compactId, middleTruncate } from "./format";

describe("identifier truncation", () => {
  it("keeps short values and shortens long ones around an ellipsis", () => {
    expect(middleTruncate("abcdef", 2, 2)).toBe("abcdef");
    expect(middleTruncate("abcdefghi", 2, 2)).toBe("ab…hi");
    expect(compactId("run-delete")).toBe("run-delete");
    expect(compactId("run_0123456789abcdef0123456789abcdef")).toBe(
      "run_01234567…89abcdef",
    );
    expect(compactDigest(`sha256:${"0".repeat(56)}12345678`)).toBe(
      "sha256:00000000…12345678",
    );
  });
});
