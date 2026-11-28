import type { Audit } from "../../../api/audits";
import { StatusGlyph } from "../../../ui";
import {
  describeStopReason,
  stopReasonTone,
} from "../../../modules/checks/stop-reason";

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
