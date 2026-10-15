import { formatRelative, formatTimestamp } from "./format";
import { useEffect, useState } from "react";

/**
 * Relative timestamp for list rows and cards. Renders a `<time>` element whose
 * `dateTime` carries the source value and whose `title` holds the absolute
 * timestamp, so the exact moment is one hover away and tests can match on it.
 */
export function RecordedTime({
  value,
  className,
}: {
  value: string;
  className?: string;
}) {
  const [now, setNow] = useState(Date.now);
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 60000);
    return () => clearInterval(timer);
  }, []);
  const exact = `${formatTimestamp(value)} · ${Intl.DateTimeFormat().resolvedOptions().timeZone}`;
  return (
    <time className={className} dateTime={value} title={exact}>
      {formatRelative(value, now) ?? "Unknown date"}
    </time>
  );
}
