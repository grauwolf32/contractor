/*
 * The UI's one set of number, size, date, time and duration formatters
 * (src/ui/README.md, "Formatting"). The UI is in English, so every formatter
 * uses en-US conventions whatever the browser locale; dates and times show
 * in the browser's time zone.
 */

const counts = new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 });

/**
 * A count and its noun: "1 check", "3 checks", "1,204 endpoints",
 * "1.5 hours". The plural is `one` + "s" unless `many` is given.
 */
export function plural(count: number, one: string, many = `${one}s`): string {
  return `${counts.format(count)} ${count === 1 ? one : many}`;
}

/** "512 B", "68.7 KiB", "3.2 MiB", "1.0 GiB", "2.0 TiB". */
export function formatBytes(size: number): string {
  if (size < 1024) {
    return `${size} B`;
  }
  if (size < 1024 * 1024) {
    return `${(size / 1024).toFixed(1)} KiB`;
  }
  if (size >= 1024 ** 4) return `${(size / 1024 ** 4).toFixed(1)} TiB`;
  if (size >= 1024 ** 3) return `${(size / 1024 ** 3).toFixed(1)} GiB`;
  return `${(size / (1024 * 1024)).toFixed(1)} MiB`;
}

const absoluteTimestamp = new Intl.DateTimeFormat("en-US", {
  dateStyle: "medium",
  timeStyle: "medium",
});
const shortDateTime = new Intl.DateTimeFormat("en-US", {
  month: "short",
  day: "numeric",
  hour: "2-digit",
  minute: "2-digit",
  hourCycle: "h23",
});
const clock = new Intl.DateTimeFormat("en-US", {
  hour: "2-digit",
  minute: "2-digit",
  hourCycle: "h23",
});
const dayOfYear = new Intl.DateTimeFormat("en-US", {
  month: "short",
  day: "numeric",
});
const datedDay = new Intl.DateTimeFormat("en-US", {
  month: "short",
  day: "numeric",
  year: "numeric",
});
const relative = new Intl.RelativeTimeFormat("en", { numeric: "auto" });

function parseDate(value: string | Date): Date | undefined {
  const date = value instanceof Date ? value : new Date(value);
  return Number.isNaN(date.valueOf()) ? undefined : date;
}

/**
 * Absolute timestamp for detail metadata and hover titles:
 * "Oct 4, 2026, 11:51:07 PM". A value that is no date is shown as written.
 */
export function formatTimestamp(value: string): string {
  const date = parseDate(value);
  return date === undefined ? value : absoluteTimestamp.format(date);
}

/** "Oct 4, 23:51" for rows and timelines; a value that is no date as written. */
export function formatShortDateTime(value: string): string {
  const date = parseDate(value);
  return date === undefined ? value : shortDateTime.format(date);
}

/** "23:51" in local time. */
export function formatClock(value: Date): string {
  return clock.format(value);
}

function startOfDay(value: Date): number {
  return new Date(
    value.getFullYear(),
    value.getMonth(),
    value.getDate(),
  ).valueOf();
}

/** "Today", "Yesterday", "Oct 4" or, in another year, "Oct 4, 2025". */
export function formatDay(value: string, now: Date): string | undefined {
  const date = parseDate(value);
  if (date === undefined) return undefined;
  const days = Math.round((startOfDay(now) - startOfDay(date)) / 86_400_000);
  if (days === 0) return "Today";
  if (days === 1) return "Yesterday";
  return date.getFullYear() === now.getFullYear()
    ? dayOfYear.format(date)
    : datedDay.format(date);
}

/**
 * Time relative to `now` for records in lists and cards, in minutes, hours
 * or days: "this minute", "5 minutes ago", "in 2 hours", "yesterday".
 * Undefined for a value that is no date.
 */
export function formatRelative(value: string, now: number): string | undefined {
  const seconds = (Date.parse(value) - now) / 1000;
  if (!Number.isFinite(seconds)) return undefined;
  const magnitude = Math.abs(seconds);
  const unit =
    magnitude >= 86400 ? "day" : magnitude >= 3600 ? "hour" : "minute";
  const divisor = unit === "day" ? 86400 : unit === "hour" ? 3600 : 60;
  return relative.format(Math.round(seconds / divisor), unit);
}

/**
 * Compact age of a live observation, to the second: "Just now", "12s ago",
 * "3m ago", "2h ago", "4d ago"; "Not observed" without a valid time.
 */
export function formatAge(value: string | undefined, now: number): string {
  if (value === undefined) return "Not observed";
  const seconds = Math.max(0, Math.floor((now - Date.parse(value)) / 1000));
  if (!Number.isFinite(seconds)) return "Not observed";
  if (seconds < 5) return "Just now";
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

/** Elapsed time of a Run or step: "<1s", "45s", "3m 12s", "2h 5m", "1d 4h"; "—" when unknown. */
export function formatElapsed(milliseconds: number | undefined): string {
  if (milliseconds === undefined) {
    return "—";
  }
  const seconds = Math.floor(milliseconds / 1000);
  if (seconds < 1) {
    return "<1s";
  }
  if (seconds < 60) {
    return `${seconds}s`;
  }
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) {
    return `${minutes}m ${seconds % 60}s`;
  }
  const hours = Math.floor(minutes / 60);
  if (hours < 24) {
    return `${hours}h ${minutes % 60}m`;
  }
  const days = Math.floor(hours / 24);
  return `${days}d ${hours % 24}h`;
}

/**
 * A time limit in words: "36 seconds", "10 minutes", "1.5 hours", "7 days";
 * 0 is "no time limit".
 */
export function formatTimeLimit(seconds: number): string {
  if (seconds === 0) return "no time limit";
  if (seconds > 86400 && seconds % 86400 === 0)
    return plural(seconds / 86400, "day");
  if (seconds >= 3600) return plural(Math.round(seconds / 36) / 100, "hour");
  if (seconds % 60 === 0) return plural(seconds / 60, "minute");
  return plural(seconds, "second");
}

const measuredSeconds = new Intl.NumberFormat("en-US", {
  maximumFractionDigits: 3,
});

/** A measured duration in seconds: "0.25 s", "1,204.5 s". */
export function formatSeconds(value: number): string {
  return `${measuredSeconds.format(value)} s`;
}

/** Shortens a long value to its head and tail around an ellipsis. */
export function middleTruncate(
  value: string,
  head: number,
  tail: number,
): string {
  return value.length <= head + tail + 4
    ? value
    : `${value.slice(0, head)}…${value.slice(-tail)}`;
}

/** Compact form of a Run, Project or other long resource identifier. */
export function compactId(value: string): string {
  return middleTruncate(value, 12, 8);
}

/** Compact form of a "sha256:<hex>" digest. */
export function compactDigest(digest: string): string {
  return middleTruncate(digest, 15, 8);
}

/**
 * Version order, oldest first: dotted numbers compare part by part ("10"
 * after "9", "1.10" after "1.9", without a size limit); other versions
 * compare naturally ("beta-2" before "beta-10"). Published numeric versions
 * get this explicit order whatever the page order; other labels stay
 * selectable but imply no publication order.
 */
export function compareVersions(left: string, right: string): number {
  const numeric = /^\d+(?:\.\d+)*$/;
  if (numeric.test(left) && numeric.test(right)) {
    const a = left.split(".").map(BigInt);
    const b = right.split(".").map(BigInt);
    for (let index = 0; index < Math.max(a.length, b.length); index++) {
      const x = a[index] ?? 0n;
      const y = b[index] ?? 0n;
      if (x !== y) return x > y ? 1 : -1;
    }
  }
  return (
    left.localeCompare(right, "en", { numeric: true }) ||
    (left < right ? -1 : left > right ? 1 : 0)
  );
}
