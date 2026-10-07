import type { UseQueryResult } from "@tanstack/react-query";
import { Link, type To } from "react-router";

import type { Audit, AuditWorkspace } from "../../../api/audits";
import { itemNoun, type ItemKind } from "../../../app/vocabulary";
import { IdChip, ProgressSegments, StatusGlyph } from "../../../ui";
import type { CheckLinks } from "./check-links";
import { timeLimitText } from "./check-format";
import {
  countLegend,
  countSegments,
  doneSummary,
  entryName,
  isConcluded,
  legend,
  legendSentence,
  workCounts,
  type CheckEntry,
  type LegendEntry,
} from "./check-model";
import { AuditControls } from "./controls";
import { describeStopReason, stopReasonTone } from "./stop-reason";

function plural(count: number, singular: string, many: string): string {
  return `${count.toLocaleString("en-US")} ${count === 1 ? singular : many}`;
}

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
              {entry.count.toLocaleString("en-US")}
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

/**
 * The check's progress header: how much is done, one segment per item in
 * list order with a legend, the lifecycle controls, what waits for the user
 * and the time limit. Counts come from the Server's workspace snapshot when
 * it has one; done counts concluded work (met, traced, issues found, not
 * applicable and excluded), never accepted issues or an approved report. A
 * list pinned to another revision than the snapshot is counted from its own
 * rows, so the header never mixes revisions (S19).
 */
export function CheckProgress({
  audit,
  projectName,
  projectId,
  kind,
  entries,
  partialList,
  pin,
  workspace,
  links,
}: {
  audit: Audit;
  projectName: string | undefined;
  projectId: string;
  kind: ItemKind;
  /** Every loaded item, in round order (not the filtered list). */
  entries: readonly CheckEntry[];
  /** More items exist than are loaded. */
  partialList: boolean;
  /** The revision the list is pinned to (`auditRevision`), if any. */
  pin: string | null;
  workspace: UseQueryResult<AuditWorkspace>;
  links: CheckLinks;
}) {
  const snapshot = workspace.data;
  // The list is pinned to another revision than the snapshot: its own rows
  // feed the whole header.
  const pinned =
    pin !== null &&
    entries.length > 0 &&
    snapshot !== undefined &&
    String(snapshot.auditRevision) !== pin;
  const source = pinned ? undefined : snapshot;
  const counts = source === undefined ? undefined : workCounts(source);
  const fromRows = entries.length > 0 && (pinned || !partialList);
  const total = counts?.total ?? entries.length;
  const legendEntries =
    audit.state === "draft" || total === 0
      ? []
      : fromRows
        ? legend(entries)
        : counts === undefined
          ? []
          : countLegend(counts);
  const segments = fromRows
    ? entries.map((entry) => ({
        tone: entry.status.tone,
        label: `${entryName(entry)}: ${entry.status.label}`,
      }))
    : countSegments(legendEntries);
  const done = source?.completedChecks ?? entries.filter(isConcluded).length;
  const noun = itemNoun(kind, total);
  const pluralNoun = itemNoun(kind, 2);
  const headline =
    audit.state === "draft"
      ? "Not started"
      : total === 0
        ? `No ${pluralNoun} yet`
        : doneSummary(done, total);
  const shown = legendEntries.filter((entry) => entry.count > 0);
  const summary =
    total === 0
      ? headline
      : `${done.toLocaleString("en-US")} of ${total.toLocaleString("en-US")} ${noun} done${shown.length === 0 ? "" : `: ${legendSentence(shown)}`}`;
  // Counts read from the snapshot pin the list they link to; counts read
  // from a pinned list keep its pin (S19).
  const linkPin = fromRows
    ? pin !== null && /^\d+$/u.test(pin)
      ? Number(pin)
      : undefined
    : snapshot?.auditRevision;
  const pending = snapshot?.pendingReviews ?? 0;
  const unreviewed = snapshot?.unreviewedFindings ?? 0;
  const timeLimit = timeLimitText(audit);
  return (
    <section className="checks-progress" aria-label="Check progress">
      <div className="checks-progress-top">
        <div className="checks-progress-summary">
          <span className="checks-progress-count">{headline}</span>
          {segments.length === 0 ? null : (
            <ProgressSegments segments={segments} label={summary} />
          )}
        </div>
        <div className="checks-progress-actions">
          <AuditControls audit={audit} projectName={projectName} menu />
          {unreviewed > 0 ? (
            <Link
              className="ui-btn"
              data-variant="primary"
              to={`/issues?${new URLSearchParams({ project: projectId, state: "proposed" }).toString()}`}
            >
              Review {plural(unreviewed, "possible issue", "possible issues")}
            </Link>
          ) : null}
        </div>
      </div>
      <div className="checks-progress-bottom">
        <ProgressLegend
          entries={legendEntries}
          linkFor={(group) => links.group(group, linkPin)}
        />
        <div className="checks-progress-facts">
          {timeLimit === undefined ? null : <span>{timeLimit}</span>}
          <span className="checks-progress-id">
            <span>Check ID</span>
            <IdChip value={audit.auditId} label="check ID" />
          </span>
        </div>
      </div>
      {pinned ? (
        <p className="checks-quiet">
          {`Counts describe revision ${pin}, the one the list shows${
            partialList
              ? `: the ${entries.length.toLocaleString("en-US")} ${pluralNoun} loaded so far`
              : ""
          }.`}
        </p>
      ) : fromRows || !partialList ? null : (
        <p className="checks-quiet">
          Counts cover the whole check; the list shows the{" "}
          {entries.length.toLocaleString("en-US")} {pluralNoun} loaded so far.
        </p>
      )}
      <StopReasonBanner audit={audit} />
      {pending > 0 || unreviewed > 0 ? (
        <p className="checks-progress-waiting">
          {pending > 0 ? (
            <Link
              to={links.deep("reviews", {
                state: "pending",
                ...(snapshot === undefined
                  ? {}
                  : { auditRevision: String(snapshot.auditRevision) }),
              })}
            >
              {plural(
                pending,
                "decision waiting for you",
                "decisions waiting for you",
              )}
            </Link>
          ) : null}
          {unreviewed > 0 ? (
            <Link
              to={links.deep("findings", {
                verdict: "unreviewed",
                ...(snapshot === undefined
                  ? {}
                  : { auditRevision: String(snapshot.auditRevision) }),
              })}
            >
              {plural(
                unreviewed,
                "possible issue not reviewed yet",
                "possible issues not reviewed yet",
              )}
            </Link>
          ) : null}
        </p>
      ) : null}
      {workspace.error === null ? null : (
        <div className="checks-notice" data-tone="warning" role="status">
          <p>
            {snapshot === undefined
              ? "The counts of this check could not be loaded."
              : "The counts could not be refreshed; they show the last snapshot."}
          </p>
          <button
            className="ui-btn"
            data-size="xs"
            type="button"
            disabled={workspace.isFetching}
            onClick={() => void workspace.refetch()}
          >
            {workspace.isFetching ? "Loading…" : "Try again"}
          </button>
        </div>
      )}
    </section>
  );
}
