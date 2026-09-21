import { formatTimestamp } from "../routes/artifacts/common";
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
  const seconds = (Date.parse(value) - now) / 1000;
  const magnitude = Math.abs(seconds);
  const unit =
    magnitude >= 86400 ? "day" : magnitude >= 3600 ? "hour" : "minute";
  const divisor = unit === "day" ? 86400 : unit === "hour" ? 3600 : 60;
  const exact = `${formatTimestamp(value)} · ${Intl.DateTimeFormat().resolvedOptions().timeZone}`;
  return (
    <time className={className} dateTime={value} title={exact}>
      {Number.isFinite(seconds)
        ? new Intl.RelativeTimeFormat("en", { numeric: "auto" }).format(
            Math.round(seconds / divisor),
            unit,
          )
        : "Unknown date"}
    </time>
  );
}
