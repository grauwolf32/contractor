import { useCallback, useEffect, useRef, useState } from "react";
import { useLocation, useNavigate, useSearchParams } from "react-router";

import { useDocumentTitle } from "../../app/document-title";
import { locationDestination } from "../../app/navigation";
import { PaneLayout, useListNavigation } from "../../ui";
import { useAnnouncement } from "../decisions/announcement";
import { useInboxData, type InboxData } from "./data";
import { InboxDetail, type DetailContext } from "./detail";
import { InboxList } from "./list";
import {
  distinctCount,
  inboxSearch,
  inboxSubtitle,
  nextToDecide,
  parseRefKey,
  refKey,
  type InboxRef,
  type InboxRow,
  type InboxSectionId,
} from "./model";
import { rowPage } from "./present";

import "./inbox.css";

function plural(count: number, one: string, many: string): string {
  return `${count.toLocaleString("en-US")} ${count === 1 ? one : many}`;
}

/** One line per section for the overview shown when nothing is selected. */
function overviewCounts(
  decide: number,
  data: InboxData,
): Record<InboxSectionId, string> {
  const { model } = data;
  const unblock = distinctCount(model.unblock);
  const ready = distinctCount(model.ready);
  const checks = model.running.length;
  const runs = data.activeRuns.data?.items.length ?? 0;
  return {
    decide:
      decide === 0
        ? "Nothing to decide"
        : `${plural(decide, "needs", "need")} your decision`,
    unblock:
      unblock === 0
        ? "Nothing is stuck"
        : plural(unblock, "is stuck", "are stuck"),
    ready:
      ready === 0
        ? "Nothing finished in the last 7 days"
        : `${ready} finished in the last 7 days`,
    running:
      checks === 0 && runs === 0
        ? "Nothing is running"
        : [
            checks === 0
              ? ""
              : plural(checks, "check is running", "checks are running"),
            runs === 0
              ? ""
              : `${data.activeRunCount ?? runs} ${runs === 1 ? "Run" : "Runs"} in progress`,
          ]
            .filter((part) => part !== "")
            .join(", "),
  };
}

/** Whether the lists that could hold this item are still loading. */
function listing(ref: InboxRef, data: InboxData): boolean {
  switch (ref.kind) {
    case "issue":
    case "review":
      return data.pending.decide;
    case "report":
      return data.reports.isPending;
    case "check":
      return data.checks.isPending;
    case "run":
      return false;
  }
}

/**
 * The Inbox at "/" (UUS:26-27): decisions first, then blocked work, finished
 * results and running checks, across every project. The selected item is
 * the `?item=` query, so Back and links restore it; J / K move through the
 * items, Enter opens the item's own page, and C / R / E decide a possible
 * issue in place.
 */
export function InboxRoute() {
  useDocumentTitle("Inbox");
  const data = useInboxData();
  const { model } = data;
  const navigate = useNavigate();
  const location = useLocation();
  const [params] = useSearchParams();
  const selected = parseRefKey(params.get("item"));
  const selectedKey = selected === undefined ? undefined : refKey(selected);
  const index =
    selectedKey === undefined
      ? -1
      : model.order.findIndex((row) => row.key === selectedKey);
  const row = index < 0 ? undefined : model.order[index];

  // "Decision recorded" outlives the decision bar, which leaves with its item.
  const { text: recorded, announce, clear } = useAnnouncement();
  // The item focus moves to once it shows; "" is the overview.
  const [focusKey, setFocusKey] = useState<string | undefined>();

  const select = useCallback(
    (
      target: InboxRow | undefined,
      options: { replace?: boolean; focus?: boolean } = {},
    ) => {
      setFocusKey(options.focus === true ? (target?.key ?? "") : undefined);
      void navigate(
        {
          pathname: "/",
          search: target === undefined ? "" : inboxSearch(target.ref),
        },
        { replace: options.replace ?? false },
      );
    },
    [navigate],
  );
  // A move by the user: the last decision's message no longer applies.
  const move = useCallback(
    (target: InboxRow | undefined, options: { replace?: boolean } = {}) => {
      clear();
      select(target, options);
    },
    [clear, select],
  );

  // The decide list as last rendered: the decided item may have left it by
  // the time the Server's answer arrives.
  const decideRows = useRef(model.decide);
  useEffect(() => {
    decideRows.current = model.decide;
  });
  const onDecided = useCallback(
    (key: string, outcome: string) => {
      select(nextToDecide(decideRows.current, key), {
        replace: true,
        focus: true,
      });
      announce(`Decision recorded: ${outcome}.`);
    },
    [announce, select],
  );

  const { containerProps } = useListNavigation({
    count: model.order.length,
    index,
    onMove: (next) => move(model.order[next], { replace: true }),
    onOpen: (at) => {
      const target = model.order[at];
      if (target === undefined) return;
      void navigate(rowPage(target), {
        state: {
          returnTo: locationDestination(location),
          returnLabel: "Inbox",
          returnState: location.state,
        },
      });
    },
  });

  const decideCount = data.summary.needsDecision ?? model.decide.length;
  const settled = !data.pending.decide || model.order.length > 0;
  const subtitle = settled
    ? inboxSubtitle({
        decide: decideCount,
        unblock: distinctCount(model.unblock),
        ready: distinctCount(model.ready),
        running: model.running.length,
      })
    : "Checking what needs you…";

  const position = row?.section === "decide" ? model.decide.indexOf(row) : -1;
  const context: DetailContext = {
    data,
    previous: index > 0 ? model.order[index - 1] : undefined,
    next: index < 0 ? undefined : model.order[index + 1],
    nextDecision: position < 0 ? undefined : model.decide[position + 1],
    onSelect: move,
    onDecided,
    focusKey,
    recorded,
  };

  // A fragment: the shell lays out a PaneLayout that is the content's child.
  return (
    <>
      <p className="ui-visually-hidden" role="status">
        {recorded}
      </p>
      <PaneLayout
        listLabel="Inbox"
        detailLabel="Selected item"
        showDetail={selected !== undefined}
        backLink={{ to: "/", label: "Inbox" }}
        list={
          <InboxList
            data={data}
            subtitle={subtitle}
            selectedKey={selectedKey}
            containerProps={containerProps}
          />
        }
        detail={
          <InboxDetail
            selected={selected}
            row={row}
            loading={selected !== undefined && listing(selected, data)}
            context={context}
            counts={overviewCounts(decideCount, data)}
          />
        }
      />
    </>
  );
}
