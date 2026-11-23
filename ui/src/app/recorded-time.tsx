import { formatClock, formatRelative, formatTimestamp } from "./format";
import { useEffect, useState } from "react";

/**
 * Relative timestamp for list rows and cards. Renders a `<time>` element whose
 * `dateTime` carries the source value and whose `title` holds the absolute
 * timestamp, so the exact moment is one hover away and tests can match on it.
 */
export function RecordedTime({
  value,
  className,
  display = "relative",
}: {
  value: string;
  className?: string;
  display?: "relative" | "clock";
}) {
  const [now, setNow] = useState(Date.now);
  useEffect(() => {
    if (display === "clock") return;
    const timer = setInterval(() => setNow(Date.now()), 60000);
    return () => clearInterval(timer);
  }, [display]);
  const exact = `${formatTimestamp(value)} · ${Intl.DateTimeFormat().resolvedOptions().timeZone}`;
  return (
    <time className={className} dateTime={value} title={exact}>
      {display === "clock"
        ? Number.isFinite(Date.parse(value))
          ? formatClock(new Date(value))
          : "Unknown date"
        : (formatRelative(value, now) ?? "Unknown date")}
    </time>
  );
}
