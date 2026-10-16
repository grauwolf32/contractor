import { ContextLink } from "../../app/context-navigation";
import { DetailPane, EmptyState } from "../../ui";
import { CheckDetail } from "./detail-check";
import { IssueDetail, ReportDetail, ReviewDetail } from "./detail-decisions";
import { Overview, type DetailContext } from "./detail-parts";
import { RunDetail } from "./detail-run";
import type { InboxRef, InboxRow, InboxSectionId } from "./model";
import { issuePath, reportPath, runPath } from "./present";

// ---------- Nothing selected, or not listed ----------

/** The page of an item that is not listed, from its reference alone. */
function referencePage(ref: InboxRef): { to: string; label: string } {
  switch (ref.kind) {
    case "issue":
      return {
        to: issuePath(ref.auditId, ref.findingId),
        label: "Open full review",
      };
    case "report":
      return { to: reportPath(ref.auditId), label: "Open report" };
    case "review":
    case "check":
      return {
        to: `/checks?check=${encodeURIComponent(ref.auditId)}`,
        label: "Open the check",
      };
    case "run":
      return { to: runPath(ref.runId), label: "Open Run" };
  }
}

function NotListed({
  selected,
  loading,
}: {
  selected: InboxRef;
  loading: boolean;
}) {
  const page = referencePage(selected);
  return (
    <DetailPane>
      {loading ? (
        <p className="ui-caption" role="status">
          Loading…
        </p>
      ) : (
        <EmptyState
          title="This item is no longer in your Inbox"
          action={
            <ContextLink returnLabel="Inbox" to={page.to}>
              {page.label}
            </ContextLink>
          }
        >
          It may have been decided, finished or removed since it was listed.
        </EmptyState>
      )}
    </DetailPane>
  );
}

/** The detail pane for the selected item. */
export function InboxDetail({
  selected,
  row,
  loading,
  context,
  counts,
}: {
  selected: InboxRef | undefined;
  /** The selected item's first row in the list, when it is listed. */
  row: InboxRow | undefined;
  /** The lists that could hold the selected item are still loading. */
  loading: boolean;
  context: DetailContext;
  counts: Record<InboxSectionId, string>;
}) {
  if (selected === undefined)
    return <Overview context={context} counts={counts} />;
  if (selected.kind === "run")
    return (
      <RunDetail
        key={selected.runId}
        runId={selected.runId}
        row={row?.type === "run" ? row : undefined}
        context={context}
      />
    );
  if (selected.kind === "check") {
    const check =
      row?.type === "check"
        ? row.check
        : context.data.checks.checks.find(
            (candidate) => candidate.audit.auditId === selected.auditId,
          );
    if (check !== undefined)
      return (
        <CheckDetail
          key={check.audit.auditId}
          check={check}
          row={row?.type === "check" ? row : undefined}
          context={context}
        />
      );
  }
  if (row?.type === "issue")
    return <IssueDetail key={row.key} row={row} context={context} />;
  if (row?.type === "review")
    return <ReviewDetail key={row.key} row={row} context={context} />;
  if (row?.type === "report")
    return <ReportDetail key={row.key} row={row} context={context} />;
  return <NotListed selected={selected} loading={loading} />;
}
