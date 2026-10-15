import { describe, expect, it } from "vitest";

import {
  compactDigest,
  compactId,
  compareVersions,
  formatAge,
  formatBytes,
  formatClock,
  formatDay,
  formatElapsed,
  formatRelative,
  formatSeconds,
  formatShortDateTime,
  formatTimeLimit,
  formatTimestamp,
  middleTruncate,
  plural,
} from "./format";

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

describe("counts and sizes", () => {
  it("pairs a count with its noun", () => {
    expect(plural(1, "check")).toBe("1 check");
    expect(plural(0, "check")).toBe("0 checks");
    expect(plural(1204, "endpoint")).toBe("1,204 endpoints");
    expect(plural(2, "entry", "entries")).toBe("2 entries");
    expect(plural(1.5, "hour")).toBe("1.5 hours");
  });

  it("formats byte sizes in binary units", () => {
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(70_350)).toBe("68.7 KiB");
    expect(formatBytes(3 * 1024 ** 2)).toBe("3.0 MiB");
    expect(formatBytes(1024 ** 3)).toBe("1.0 GiB");
    expect(formatBytes(2 * 1024 ** 4)).toBe("2.0 TiB");
  });

  it("formats measured seconds", () => {
    expect(formatSeconds(0.25)).toBe("0.25 s");
    expect(formatSeconds(1204.5)).toBe("1,204.5 s");
    expect(formatSeconds(1.23456)).toBe("1.235 s");
  });
});

describe("dates and times", () => {
  // Local times, so the expectations hold in every time zone.
  const evening = new Date(2026, 9, 4, 23, 51, 7);

  it("formats absolute and short timestamps and keeps other text", () => {
    expect(formatTimestamp(evening.toISOString())).toBe(
      "Oct 4, 2026, 11:51:07 PM",
    );
    expect(formatShortDateTime(evening.toISOString())).toBe("Oct 4, 23:51");
    expect(formatClock(evening)).toBe("23:51");
    expect(formatTimestamp("soon")).toBe("soon");
    expect(formatShortDateTime("soon")).toBe("soon");
  });

  it("names days relative to today", () => {
    const now = new Date(2026, 9, 5, 9, 0);
    expect(formatDay(new Date(2026, 9, 5, 1, 0).toISOString(), now)).toBe(
      "Today",
    );
    expect(formatDay(evening.toISOString(), now)).toBe("Yesterday");
    expect(formatDay(new Date(2026, 8, 1).toISOString(), now)).toBe("Sep 1");
    expect(formatDay(new Date(2025, 9, 4).toISOString(), now)).toBe(
      "Oct 4, 2025",
    );
    expect(formatDay("not a date", now)).toBeUndefined();
  });

  it("says how long ago a record was made, in minutes, hours or days", () => {
    const now = Date.parse("2026-09-22T12:00:00Z");
    expect(formatRelative("2026-09-22T11:59:30Z", now)).toBe("this minute");
    expect(formatRelative("2026-09-22T09:30:00Z", now)).toBe("2 hours ago");
    expect(formatRelative("2026-09-15T12:00:00Z", now)).toBe("7 days ago");
    expect(formatRelative("2026-09-22T15:00:00Z", now)).toBe("in 3 hours");
    expect(formatRelative("not a date", now)).toBeUndefined();
  });

  it("gives the compact age of a live observation to the second", () => {
    const now = Date.parse("2026-09-22T12:00:00Z");
    expect(formatAge(undefined, now)).toBe("Not observed");
    expect(formatAge("not a date", now)).toBe("Not observed");
    expect(formatAge("2026-09-22T11:59:58Z", now)).toBe("Just now");
    expect(formatAge("2026-09-22T11:59:48Z", now)).toBe("12s ago");
    expect(formatAge("2026-09-22T11:57:00Z", now)).toBe("3m ago");
    expect(formatAge("2026-09-22T10:00:00Z", now)).toBe("2h ago");
    expect(formatAge("2026-09-18T12:00:00Z", now)).toBe("4d ago");
  });
});

describe("durations", () => {
  it("formats elapsed time for quick scanning", () => {
    expect(formatElapsed(undefined)).toBe("—");
    expect(formatElapsed(700)).toBe("<1s");
    expect(formatElapsed(34_000)).toBe("34s");
    expect(formatElapsed(254_000)).toBe("4m 14s");
    expect(formatElapsed(7_440_000)).toBe("2h 4m");
    expect(formatElapsed(100_800_000)).toBe("1d 4h");
  });

  it("describes a time limit in words", () => {
    expect(formatTimeLimit(86400)).toBe("24 hours");
    expect(formatTimeLimit(604800)).toBe("7 days");
    expect(formatTimeLimit(5400)).toBe("1.5 hours");
    expect(formatTimeLimit(600)).toBe("10 minutes");
    expect(formatTimeLimit(36)).toBe("36 seconds");
    expect(formatTimeLimit(0)).toBe("no time limit");
  });
});

describe("version order", () => {
  it("orders numeric versions as numbers and others naturally", () => {
    expect(["2", "10", "9", "1.10", "1.9"].sort(compareVersions)).toEqual([
      "1.9",
      "1.10",
      "2",
      "9",
      "10",
    ]);
    expect(
      compareVersions("999999999999999999999", "999999999999999999998"),
    ).toBeGreaterThan(0);
    expect(compareVersions("beta-2", "beta-10")).toBeLessThan(0);
  });
});
