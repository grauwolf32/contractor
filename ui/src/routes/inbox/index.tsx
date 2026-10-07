import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
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
  rowIndex,
  type InboxRef,
  type InboxRow,
  type InboxSectionId,
} from "./model";
import { rowPage } from "./present";

import "./inbox.css";

/** The overview, where focus requests and announcements name no item. */
const OVERVIEW = "";

/** The row the user chose for an item that is listed twice. */
interface RowChoice {
  key: string;
  section: InboxSectionId;
}

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
      return data.checks.isPending || data.reports.pending;
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
  // The URL names the item. A check listed twice (stuck and running) is
  // selected in the row the user chose, otherwise in its first row.
  const [choice, setChoice] = useState<RowChoice | undefined>();
  const index =
    selectedKey === undefined
      ? -1
      : rowIndex(
          model.order,
          selectedKey,
          choice?.key === selectedKey ? choice.section : undefined,
        );
  const row = index < 0 ? undefined : model.order[index];
  const position = row?.section === "decide" ? model.decide.indexOf(row) : -1;

  // Where the page is: the selected item, or the overview.
  const place = selectedKey ?? OVERVIEW;

  // "Decision recorded" outlives the decision bar, which leaves with its
  // item. It shows where it was announced: on the item the decision moved
  // to, or where the user already was when the answer came.
  const { text: announced, announce, clear } = useAnnouncement();
  const [announcedAt, setAnnouncedAt] = useState<string | undefined>();
  // The item (or OVERVIEW) that takes focus once it shows after a decision.
  // The move happens once: opening that item again leaves focus alone.
  const [focusKey, setFocusKey] = useState<string | undefined>();
  const onFocused = useCallback(() => setFocusKey(undefined), []);
  // Any other change of selection (a row, a key, Back, a link) ends both.
  // Adjusted while rendering, as React recommends over an effect: the
  // decision sets them for the place it moves to before the URL follows.
  const [lastPlace, setLastPlace] = useState(place);
  if (lastPlace !== place) {
    setLastPlace(place);
    if (announcedAt !== place) setAnnouncedAt(undefined);
    if (focusKey !== undefined && focusKey !== place) setFocusKey(undefined);
  }
  const recorded = announcedAt === place ? announced : "";

  // A decision's answer can arrive after the user moved on or left the
  // Inbox: these hold what the page last showed, and whether it still does.
  const selectedKeyRef = useRef(selectedKey);
  const decideRows = useRef(model.decide);
  // Where the selected decision sat in Decide: a refresh can remove it
  // before the Server answers, and the item after it moved into its place.
  const decisionPosition = useRef<{ key: string; index: number } | undefined>(
    undefined,
  );
  useLayoutEffect(() => {
    selectedKeyRef.current = selectedKey;
    decideRows.current = model.decide;
    if (selectedKey !== undefined && position >= 0)
      decisionPosition.current = { key: selectedKey, index: position };
  });
  const mounted = useRef(false);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  const select = useCallback(
    (
      target: InboxRow | undefined,
      options: { replace?: boolean; focus?: boolean } = {},
    ) => {
      setChoice(
        target === undefined
          ? undefined
          : { key: target.key, section: target.section },
      );
      setFocusKey(
        options.focus === true ? (target?.key ?? OVERVIEW) : undefined,
      );
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
  // A row the user clicked; its link puts the item in the URL.
  const choose = useCallback(
    (target: InboxRow) => {
      clear();
      setFocusKey(undefined);
      setChoice({ key: target.key, section: target.section });
    },
    [clear],
  );

  const announceAt = useCallback(
    (at: string, text: string) => {
      setAnnouncedAt(at);
      announce(text);
    },
    [announce],
  );
  const onDecided = useCallback(
    (key: string, outcome: string, title: string) => {
      // The user left the Inbox while the decision was on its way.
      if (!mounted.current) return;
      const current = selectedKeyRef.current;
      if (current !== key) {
        // They moved on in the Inbox: they stay where they are, and the
        // message names what was decided.
        announceAt(
          current ?? OVERVIEW,
          `Decision recorded on “${title}”: ${outcome}.`,
        );
        return;
      }
      const former = decisionPosition.current;
      const next = nextToDecide(
        decideRows.current,
        key,
        former?.key === key ? former.index : undefined,
      );
      announceAt(next?.key ?? OVERVIEW, `Decision recorded: ${outcome}.`);
      select(next, { replace: true, focus: true });
    },
    [announceAt, select],
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

  const context: DetailContext = {
    data,
    previous: index > 0 ? model.order[index - 1] : undefined,
    next: index < 0 ? undefined : model.order[index + 1],
    nextDecision: position < 0 ? undefined : model.decide[position + 1],
    onSelect: move,
    onDecided,
    focusKey,
    onFocused,
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
            selected={row}
            onChoose={choose}
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
