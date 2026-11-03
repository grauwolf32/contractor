import { Link, type To } from "react-router";
import type { Audit } from "../../../api/audits";
import { formatNumber } from "../../../app/format";
import { StatusGlyph } from "../../../ui";
import type { LegendEntry } from "./check-model";
import { describeStopReason, stopReasonTone } from "./stop-reason";

/**
 * The legend under a progress line: a swatch, the status and how many items
 * have it, each linking to the list filtered to it.
 */
export function ProgressLegend({
  entries,
  linkFor,
}: {
  entries: readonly LegendEntry[];
  linkFor: (group: LegendEntry["group"]) => To;
}) {
  if (entries.length === 0) return null;
  return (
    <ul className="checks-legend" aria-label="Legend">
      {entries.map((entry) => (
        <li key={entry.label}>
          <Link
            to={linkFor(entry.group)}
            aria-label={`${entry.label}: ${entry.count}`}
          >
            <span
              className="checks-swatch"
              data-tone={entry.tone}
              aria-hidden="true"
            />
            <span>{entry.label}</span>
            <span className="checks-legend-count">
              {formatNumber(entry.count)}
            </span>
          </Link>
        </li>
      ))}
    </ul>
  );
}

/** The stop reason of a check as one plain sentence. */
export function StopReasonBanner({ audit }: { audit: Audit }) {
  const stop = describeStopReason(audit);
  if (stop === null) return null;
  const tone = stopReasonTone(stop);
  return (
    <div
      className="checks-banner"
      data-tone={tone}
      role={stop.tone === "error" && !stop.deadline ? "alert" : "status"}
    >
      <StatusGlyph tone={tone} />
      <p>{stop.sentence}</p>
    </div>
  );
}
