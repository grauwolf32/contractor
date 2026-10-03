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

/**
 * Absolute timestamp for detail metadata. Every absolute date in the UI goes
 * through this formatter so the shape is stable across call sites and
 * browser locales; lists and cards use RecordedTime (relative) instead.
 */
export function formatTimestamp(value: string): string {
  const parsed = new Date(value);
  return Number.isNaN(parsed.valueOf())
    ? value
    : absoluteTimestamp.format(parsed);
}
