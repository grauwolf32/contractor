# Shared UI building blocks (`src/ui`)

The V3B list + detail primitives, buttons, dialogs, facts and fields. Import
them from `src/ui`:

```tsx
import { Button, ListPane, ListRow, PaneLayout, StatusGlyph } from "../../ui";
```

Pages use these instead of writing their own: lint rejects a hand-written
`ui-btn` button, `role="alertdialog"` or inline `<svg>` icon outside `src/ui`
and `src/app` (see [Gates](#gates)). Shared helpers that are not components
(icons, formatting, polling, write conflicts, late navigation, vocabulary)
live in `src/app` and are listed under [App helpers](#app-helpers).

Styles live in `src/ui/ui.css` (imported once from `src/main.tsx`, classes
start with `ui-`). Colours come only from `src/app/theme.css` tokens, so every
component works in the light, dark and black themes. The visual reference is
`docs/design/ui/mockups/v3b/*.html`; the rules are in
`docs/design/ui/v3b-build-contract.md`.

Optional props also accept `undefined`, so values can be passed through under
`exactOptionalPropertyTypes`.

## Layout

### `PaneLayout`

```ts
PaneLayout(props: {
  list: ReactNode;
  detail: ReactNode;
  showDetail: boolean;
  backLink?: { to: To; label: string };
  listLabel: string;
  detailLabel: string;
})
```

Two `<section>`s named `listLabel` and `detailLabel`, separated by a 1 px
hairline.

- **Above 1100 px:** the list is 360 px wide.
- **821–1100 px:** the list is 300 px wide.
- **Above 820 px:** the frame is `calc(100dvh - var(--shell-topbar-height, 0px))`
  tall and each pane scrolls on its own.
- **≤ 820 px:** one pane at a time in normal page flow. The list shows by
  default; with `showDetail` the detail shows, with `← backLink.label` above it.
  Opening the detail moves focus to the detail pane. Going back moves focus
  to the row the user was on, which scrolls it into view: the selected row if
  the selection stays, else the row that was selected last while the detail
  showed (Back usually clears the selection), else what had focus in the
  list. When there is no such row, or it has left the list, focus goes to the
  list pane, scrolled to its top.

Selection belongs in the URL. `showDetail` is usually "an item is selected".

```tsx
<PaneLayout
  listLabel="Possible issues"
  detailLabel="Review"
  showDetail={findingId !== undefined}
  backLink={{ to: "/issues", label: "Back to possible issues" }}
  list={<IssuesList />}
  detail={
    finding ? (
      <IssueDetail finding={finding} />
    ) : (
      <EmptyState title="Choose a possible issue" />
    )
  }
/>
```

The shell sets `--shell-topbar-height` (0 when the content starts at the top)
and gives pane pages a content area without padding.

### `ListPane`

```ts
ListPane(props: {
  title?: ReactNode;
  titleAs?: "h1" | "h2";      // default "h1"
  titleRef?: Ref<HTMLHeadingElement>; // the title can take focus (tabIndex -1)
  subtitle?: ReactNode;
  actions?: ReactNode;        // right of the title
  header?: ReactNode;         // replaces title/subtitle/actions
  toolbar?: ReactNode;        // under the title, e.g. FilterChips
  footer?: ReactNode;         // bottom: key hints, Technical details
  children: ReactNode;
  "aria-label"?: string;      // renders a <section>; omit inside PaneLayout
})
```

Rows and section headings line up with the pane edges through
`--ui-pad-start` / `--ui-pad-end`. A direct `EmptyState` or `TechnicalDetails`
child is padded to match; give other direct children the same inline padding.

`actions` sit right of the title and wrap under it when the pane is narrow.
They never grow past the pane, also at 320 px: the actions block shrinks
(`flex: 0 1 auto`, `min-width: 0`, `max-width: 100%`), and so do its direct
children and any select, input or textarea in it. A select sizes to the pane
instead of to its longest option, so it needs no width cap of its own. Long
button labels do not wrap there; keep them short.

With `titleRef` the title heading gets `tabIndex={-1}` and the ref, so a page
can move focus to it once what had focus went away, for example after the
last item of a filtered list was decided. It shows a focus ring only for
keyboard users. With `header`, give the `DetailHeader` the ref instead.

```tsx
const title = useRef<HTMLHeadingElement>(null);
// …once the selection left the list:
title.current?.focus();

<ListPane title="Possible issues" titleRef={title}>
  …
</ListPane>;
```

```tsx
<ListPane
  title="Inbox"
  subtitle="2 need you, 1 check is running."
  footer={
    <>
      <Kbd>J</Kbd> <Kbd>K</Kbd> move
    </>
  }
>
  <ListSection title="Decide" count={1} aside="only you can confirm">
    …
  </ListSection>
</ListPane>
```

### `ListSection`

```ts
ListSection(props: {
  title?: ReactNode;          // omit for a plain list
  count?: number | string;
  aside?: ReactNode;
  titleAs?: "h2" | "h3";      // default "h2"
  "aria-label"?: string;      // names the <ul> itself
  "aria-keyshortcuts"?: string; // keys declared on the <ul>
  children: ReactNode;        // ListRow elements
})
```

A `<section>` labelled by its small uppercase heading, then
`<ul role="list">`. `aria-label` and `aria-keyshortcuts` go on the `<ul>`, so
a page can name a list without a title and declare the keys that move
through it (J / K, arrows, Enter) on the list itself; the visible hint is a
`Kbd` line, usually in the pane's footer. To declare them on every row's
focusable link or button instead, use `ListRow`'s `ariaKeyShortcuts`.

```tsx
<ListSection title="Endpoints" count={5} aside="under /workshop/api/">
  {rows}
</ListSection>

<div {...containerProps}>
  <ListSection aria-keyshortcuts="J K ArrowDown ArrowUp Enter">{rows}</ListSection>
</div>
```

### `ListRow`

```ts
ListRow(props: {
  to?: To;                    // title is a router Link
  onSelect?: () => void;      // title is a button; with `to`, also runs on click
  selected?: boolean;         // aria-current="true", tint and 3 px bar
  glyph?: ReactNode;          // usually <StatusGlyph />
  title: ReactNode;
  meta?: ReactNode;           // an array is joined with "·"
  trailing?: ReactNode;
  clamp?: 1 | 2 | false;      // title lines, default 2
  children?: ReactNode;       // inline actions, progress
  id?: string;
  ariaKeyShortcuts?: string;  // aria-keyshortcuts of the link or button
})
```

Renders an `<li>`. The title link or button covers the whole row (one click
target, one focus ring). Inline actions in `children` and `trailing` are
separate controls above it: never put a button inside the title. `meta` is
for text and identifiers: an `IdChip` (or a link) there stays clickable, but
actions belong in `children` or `trailing`. The glyph is decorative, so say
the status in words in `meta`. With a method chip, keep a space before the
path so the name reads "GET /path". `ariaKeyShortcuts` declares the keys that
act on the row on its link or button, where screen readers announce them on
focus (e.g. `"J K ArrowDown ArrowUp Home End Enter"` with
`useListNavigation`); a row without `to` or `onSelect` has no focusable
control and ignores it.

```tsx
<ListRow
  to={`/checks?check=${check.id}`}
  selected={check.id === selectedId}
  glyph={<StatusGlyph tone="blocked" />}
  title={
    <>
      <MethodChip method="PUT" /> service_request/{"{id}"}
    </>
  }
  meta={[<strong key="state">Blocked</strong>, "AI response length limit"]}
>
  <button type="button" className="ui-btn" data-size="xs">
    Open the run
  </button>
</ListRow>
```

### `FilterChips`

```ts
FilterChips<T extends string>(props: {
  label: string;              // group name
  options: readonly { value: T; label: string; count?: number | string }[];
  value: T;
  onChange: (value: T) => void;
})
```

`role="group"` of `aria-pressed` pill buttons. Each button is named
"label count", e.g. "Needs review 1". Pressing the chip that is already
pressed changes nothing: `onChange` runs only for another value, so a page
keeps its cursor, open rows and other state without guarding against it.

```tsx
<FilterChips
  label="Filter by status"
  value={state}
  onChange={setState}
  options={[
    { value: "proposed", label: "Needs review", count: 1 },
    { value: "confirmed", label: "Confirmed", count: 0 },
  ]}
/>
```

### `DetailPane`

```ts
DetailPane(props: {
  header?: ReactNode;         // usually DetailHeader
  footer?: ReactNode;         // pinned to the bottom above 820 px, e.g. DecisionBar
  children: ReactNode;        // padded body, 20 px gaps between children
  "aria-label"?: string;      // renders a <section>; omit inside PaneLayout
})
```

```tsx
<DetailPane header={<DetailHeader title={finding.title} />} footer={<DecisionBar … />}>
  <FindingSummary finding={finding} />
</DetailPane>
```

Where the footer is pinned (above 820 px wide and 560 px tall), content
that shows a recorded decision carries the `ui-footer-record` class: it is
capped at `min(45vh, 24rem)` and scrolls on its own, so a long reason never
covers the pane, and it draws the divider above it so the line stays put
while it scrolls (its first block drops its own divider). The decision
components (`src/routes/decisions`) set the class themselves when they show
a `DecisionRecord` or a current decision without the bar, and keep the
capped content usable from the keyboard: a record without a control of its
own is a tab stop while it scrolls, and actions inside it stay in view. The
pending `DecisionBar` never carries the class and stays whole. Pages need no
cap of their own.

### `DetailHeader`

```ts
DetailHeader(props: {
  breadcrumb?: readonly { label: ReactNode; to?: To; state?: unknown }[];
  title: ReactNode;
  titleAs?: "h1" | "h2";      // default "h2"
  titleRef?: Ref<HTMLHeadingElement>; // the title can take focus (tabIndex -1)
  eyebrow?: ReactNode;        // the page kind on a line above the title
  status?: ReactNode;         // usually StatusChip
  meta?: ReactNode;
  actions?: ReactNode;
})
```

Breadcrumb items with `to` are links (with `state`, the link carries
location state, such as the origin page's own way back); a last item
without `to` gets `aria-current="page"`. `eyebrow` names the kind of page
in small capitals above the title ("Workflow Run"). The title is 20 px in a list pane and up to 28 px in a
wide detail pane. It also serves as a list pane `header`, for example the
check page's breadcrumb, h1 and state. `titleRef` works as in `ListPane`: the
heading gets `tabIndex={-1}`, so the page can focus it when the control that
had focus (a closed panel, the item it showed) went away.

```tsx
<DetailHeader
  breadcrumb={[{ label: "Checks", to: "/checks" }, { label: project.name }]}
  title="API endpoint trace"
  titleAs="h1"
  status={
    <StatusChip tone="progress" size="sm">
      Running
    </StatusChip>
  }
  meta={<IdChip value={audit.id} label="check ID" />}
/>
```

## Status

`StatusTone` (from `src/app/status-tone.ts`, re-exported here) is one of
`done`, `partial`, `progress`, `blocked`, `idle`, `review`, `warning`,
`success`, `info` and `neutral`. Glyph and chip colours: done/success green,
partial/review/warning amber, progress/info blue, blocked red, idle/neutral
grey. Status is never colour alone: always show the word.

### `StatusGlyph`

```ts
StatusGlyph(props: { tone: StatusTone; size?: number /* 16 */; label?: string })
```

Inline stroke SVG: done is a check in a circle, partial a half-filled
circle, progress an open arc, blocked a slashed circle, idle a dashed circle,
review a diamond with "!", warning a triangle with "!", success a check, info
an "i" in a circle, neutral a dot. Without `label` it is `aria-hidden`; with
one it is `role="img"` named by the label.

```tsx
<StatusGlyph tone="partial" />
<StatusGlyph tone="blocked" label="Blocked" />
```

### `StatusChip`

```ts
StatusChip(props: { tone: StatusTone; children: ReactNode; glyph?: boolean /* true */; size?: "sm" | "md" /* "md" */ })
```

```tsx
<StatusChip tone="review">Possible issue, needs your review</StatusChip>
```

### `ProgressSegments`

```ts
ProgressSegments(props: {
  segments: readonly { tone: StatusTone; label: string }[];
  label: string;              // accessible summary of the whole line
  size?: "sm" | "md";         // 4 px or 8 px (default) bars
})
```

One `role="img"` element named by `label`. Up to 60 items it shows
equal-width bars, one per item in list order, 3 px apart (2 px above 40
items). Above 60 items (an ASVS check has about 280 requirements) per-item
bars would vanish in a list row, so consecutive items of one tone merge into
one bar as wide as their share, without gaps. List order and proportions
stay, and the bar's hover title names its labels with counts ("Met (12)").
Colours: done/success `--m-done`, partial/review/warning `--m-partial`,
progress/info `--m-progress`, blocked `--m-blocked`, idle/neutral `--m-idle`.

```tsx
<ProgressSegments
  label="0 of 5 endpoints done: 1 partially traced, 1 blocked, 1 in progress, 2 not checked yet"
  segments={items.map((item) => ({
    tone: coverageTone(item),
    label: coverageLabel(item),
  }))}
/>
```

## Chips and hints

### `MethodChip`

```ts
MethodChip(props: { method: string })   // upper-cased, monospace, on --chrome
```

```tsx
<>
  <MethodChip method="get" /> /workshop/api/mechanic/mechanic_report
</>
```

### `Kbd`

```ts
Kbd(props: { children: ReactNode })
```

On a control that declares `aria-keyshortcuts`, wrap hints in
`<span aria-hidden="true">` so the accessible name stays clean.

```tsx
<button type="button" aria-keyshortcuts="J">
  Next item{" "}
  <span aria-hidden="true">
    <Kbd>J</Kbd>
  </span>
</button>
```

### `IdChip`

```ts
IdChip(props: { value: string; label: string; display?: string; wrap?: boolean })
```

Shows `display`, or the value shortened to its first 8 and last 4
characters (`shortenId`). Values up to 16 characters are kept whole. The
full value is the hover title. Text too long for its line ends in an
ellipsis; with `wrap` it breaks anywhere onto more lines instead, for full
identifiers that must stay readable where there is no hover (a stage
execution ID on a phone). The copy button is named "Copy " + label. It
uses `navigator.clipboard.writeText` and announces "Copied". When the
Clipboard API is missing or refuses, it selects the full value in a hidden
element and says "Press Ctrl+C to copy" ("Press ⌘+C to copy" on Apple
platforms, from `modKeyLabel()`). Both messages use an `aria-live="polite"`
region.

```tsx
<IdChip value={audit.id} label="check ID" />
<IdChip value={`${ref.name}@${ref.version}`} display={`${ref.name}@${ref.version}`} label="workflow version" />
<IdChip value={attempt.stageExecutionId} display={attempt.stageExecutionId} label="stage execution ID" wrap />
```

## Content

### `TechnicalDetails`

```ts
TechnicalDetails(props: {
  summary?: string;           // default "Technical details"
  description?: ReactNode;
  children: ReactNode;
  defaultOpen?: boolean;
  className?: string;         // added next to "ui-tech", e.g. a spec hook
  onToggle?: (open: boolean) => void;
})
```

A native `<details>` styled as a quiet link. Revisions, digests, rounds,
slots, tokens and allocations go here. `onToggle` runs with the new state
whenever it opens or closes, for example to render costly content only while
it is open. The chevron turns only for its own summary, so a closed
disclosure inside an open one keeps pointing right.

```tsx
<TechnicalDetails description="Steps, limits and AI usage, for admins and debugging.">
  <dl>…</dl>
</TechnicalDetails>

<TechnicalDetails summary="Detailed counters" className="ops-counters" onToggle={setOpen}>
  {open ? <Counters /> : null}
</TechnicalDetails>
```

### `ActivityLog`

```ts
ActivityLog(props: {
  entries: readonly { id: string; time?: string | Date; tone?: StatusTone; title?: ReactNode; text?: ReactNode }[];
  "aria-label": string;
})
```

An ordered timeline: time column, glyph, text. A `Date` or ISO string shows
as local HH:MM (`<time>`, full timestamp on hover); other text such as "now"
is shown as written. Entries without a time are steps of the entry above and
get a small dot. Timed entries without a tone use `neutral`.

```tsx
<ActivityLog
  aria-label="Activity on this endpoint"
  entries={[
    {
      id: "a",
      time: attempt.finishedAt,
      tone: "partial",
      title: "Stopped with a partial trace.",
      text: reason,
    },
    { id: "b", text: "Compared sibling views." },
  ]}
/>
```

### `EmptyState`

```ts
EmptyState(props: { title: ReactNode; children?: ReactNode; action?: ReactNode })
```

```tsx
<EmptyState
  title="Nothing needs you"
  action={<Link to="/checks/new">Start a check</Link>}
>
  Possible issues arrive here as checks find them.
</EmptyState>
```

## Decisions

### `DecisionBar`

```ts
DecisionBar(props: {
  options: readonly { id: string; label: string; shortcut?: string; tone?: "primary" | "secondary" | "danger"; icon?: ReactNode }[];
  selected?: string;
  onSelect: (id: string) => void;
  severity?: {
    options: readonly { value: string; label: string }[];
    value?: string;
    onChange: (value: string) => void;
    required: boolean;
    label?: string;           // default "Severity"
    hint?: ReactNode;
  };
  rationale: { value: string; onChange: (value: string) => void; label?: string /* "Why" */; placeholder?: string; maxLength?: number };
  onSubmit: () => void;
  submitLabel?: string;       // default "Record decision"
  pending?: boolean;
  error?: ReactNode;          // role="alert" under the bar
  disabledReason?: string;    // disables deciding (not Next) and says why
  more?: ReactNode;           // after the verdicts, e.g. a More menu
  next?: { label: string; shortcut?: string; onNext: () => void };
  extra?: ReactNode;          // full width above the reason, e.g. a duplicate picker
  "aria-label"?: string;      // default "Your decision"
})
```

A `<section aria-label="Your decision">` laid out like the mockup's sticky
bar. Put it in `DetailPane`'s `footer` to pin it.

- **Severity:** a `role="radiogroup"` of native radios, so arrow keys work.
  It has `aria-required` when required.
- **Verdicts:** `aria-pressed` buttons with `aria-keyshortcuts` and a Kbd hint.
  A click or the verdict's key selects it and moves focus to the reason.
- **Reason:** a `<textarea>` labelled "Why". Ctrl+Enter or ⌘+Enter records.
- **Record:** disabled until a verdict is chosen, the reason is not blank and
  a required severity is set. A line under the bar says what is missing
  ("Choose a decision.", "Choose severity.", "Write a short reason.") and
  describes the button.
- **`pending`:** locks every control, turns off the keys and makes `more`
  and `extra` inert. Keyboard focus stays in the bar until the page moves it
  after the answer, as a browser drops focus from a control that turns
  disabled or inert: the reason only turns read-only and Record only
  `aria-disabled` and `aria-busy` (pressing it again sends nothing). When
  recording starts with focus on a control that does lock, or on the page
  body, focus moves to Record in the same commit (Safari and Firefox on
  macOS do not focus a clicked button, so a verdict can keep it).
- **`next`:** a quiet button. Its key is bound as well; if the page already
  binds the same key to the same action, only one handler runs.
- **Narrow panes:** the verdict, Next and Record buttons shrink and wrap
  their labels instead of overflowing (long `next.label` or `submitLabel`
  texts stay whole). In bars up to 34rem wide, and on touch-only devices, the
  key hints are hidden; `aria-keyshortcuts` stays.

```tsx
<DecisionBar
  options={[
    {
      id: "true_positive",
      label: "Confirm issue",
      shortcut: "c",
      tone: "primary",
    },
    { id: "false_positive", label: "Not an issue", shortcut: "r" },
    { id: "needs_evidence", label: "Needs evidence", shortcut: "e" },
  ]}
  selected={verdict}
  onSelect={setVerdict}
  severity={
    verdict === "true_positive"
      ? {
          options: SEVERITIES,
          value: severity,
          onChange: setSeverity,
          required: true,
        }
      : undefined
  }
  rationale={{ value: reason, onChange: setReason }}
  onSubmit={() =>
    decide.mutate({ verdict, severity, rationale: reason.trim() })
  }
  pending={decide.isPending}
  error={decide.error?.message}
  more={<ActionMenu label="More decisions">…</ActionMenu>}
  next={{ label: "Next item", shortcut: "j", onNext: selectNext }}
/>
```

## Keyboard

### `useShortcuts`

```ts
useShortcuts(
  bindings: Record<
    string,
    | ((event: KeyboardEvent) => void)
    | { handler: (event: KeyboardEvent) => void; when?: (event: KeyboardEvent) => boolean }
  >,
  options?: { enabled?: boolean; allowInDialog?: boolean },
): void
```

One `keydown` listener on `document`. Keys are lowercase `KeyboardEvent.key`
names: `"j"`, `"?"`, `"enter"`, `"escape"`, `"arrowdown"`, `"space"`. Use
`"mod+k"` for Ctrl on Windows and Linux and ⌘ on macOS. On layouts that type
non-Latin letters (such as Cyrillic), letter bindings also match the physical
key, so J stays J.

A binding is skipped when:

- the event was already handled (`defaultPrevented`) or is part of an IME
  composition;
- it is a plain-key binding and Ctrl, ⌘ or Alt is held (Shift is allowed, for
  "?");
- focus is in a text input, textarea, select or contenteditable region, except
  for `"escape"` and `mod+…` bindings. Checkboxes and radios are not text
  inputs;
- a focused control uses the key itself (Enter, Space, arrows, Home, End,
  Page Up and Page Down on links, buttons, inputs and widgets);
- an element with `aria-modal="true"` exists, unless `allowInDialog` is set.
  The handler then only reacts to keys pressed inside a dialog;
- the binding's own `when` returns false. It is asked last, right before the
  binding would take the event.

A binding that fires calls `preventDefault`, so when two hooks bind the same
key the first listener wins. A skipped or declined event is left alone: no
`preventDefault`, so a later listener or the browser acts on it. Handlers can
change on every render without re-subscribing.

```tsx
useShortcuts({ "mod+k": openPalette, "?": showHelp });

// Page-wide Ctrl/⌘+Enter that leaves a focused link its own Ctrl/⌘+Enter
// (open in a new tab).
useShortcuts({
  "mod+enter": {
    when: (event) =>
      !(event.target instanceof Element && event.target.closest("a[href]")),
    handler: start,
  },
});
```

### `useListNavigation`

```ts
useListNavigation(options: {
  count: number;
  index: number;              // -1 when nothing is selected
  onMove: (index: number) => void;
  onOpen?: (index: number) => void;
  enabled?: boolean;
}): { containerProps: { ref: RefCallback<HTMLElement>; onKeyDown: (event) => void } }
```

J and K work anywhere on the page. ↓, ↑, Home, End and Enter work while
focus is inside the element that gets `containerProps` (spread them on a DOM
element that wraps the rows). Moves clamp at both ends and do not call
`onMove` when nothing changes. With nothing selected, J and K select the
first item. After a move, the selected row (the element with
`aria-current="true"`, as ListRow renders it) scrolls into view. Focus moves
to it when focus was in the list. Enter on the selected row, or on the
container itself, calls `onOpen`. Enter on other links and buttons, such as
inline actions, keeps their own action. Use one `useListNavigation` per page.

```tsx
const { containerProps } = useListNavigation({
  count: items.length,
  index: items.findIndex((item) => item.id === selectedId),
  onMove: (next) => {
    const item = items[next];
    if (item) navigate(`/issues/${item.auditId}/${item.id}`);
  },
});
return (
  <div {...containerProps}>
    <ListSection title="Needs review">{rows}</ListSection>
  </div>
);
```

### Helpers

- `modKeyLabel()`: `"⌘"` on Apple platforms, else `"Ctrl"`, for key hints.
  `isApplePlatform()` is the underlying test.
- `shortenId(value)`: the IdChip short form.

## Buttons

### `Button`

```ts
Button(props: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "danger" | "ghost" | "link"; // default "secondary"
  size?: "xs" | "sm" | "md";      // 28, 32 or 38 px; default "md"
  icon?: ReactNode;               // before the text, usually <Icon />
  type?: "button" | "submit" | "reset"; // default "button"
  pending?: boolean;              // disabled + aria-busy
  pendingLabel?: ReactNode;       // the text while pending ("Saving…")
  ref?: Ref<HTMLButtonElement>;
})
```

The `.ui-btn` look as a component. It is `type="button"` unless `type`
says otherwise, so it never submits a form by accident. Use `primary` once
per page for the main action, `danger` for destructive ones, `ghost` for
quiet toolbar actions and `link` for an action that reads as text.

```tsx
<Button
  variant="primary"
  size="sm"
  icon={<Icon name="plus" />}
  pending={create.isPending}
  pendingLabel="Starting…"
  onClick={() => create.mutate()}
>
  Start check
</Button>
```

Adopting it from markup is mechanical: `className="ui-btn"` becomes
`Button`, `data-variant` and `data-size` become `variant` and `size`, a
leading `<Icon>` moves into `icon`, and `disabled={x.isPending}` with a
ternary label becomes `pending` and `pendingLabel`. Other classes stay in
`className`.

### `IconButton`

```ts
IconButton(props: Omit<ButtonHTMLAttributes<HTMLButtonElement>, "children"> & {
  label: string;                  // accessible name and hover title
  icon: ReactNode;
  variant?: ButtonVariant; size?: ButtonSize; pending?: boolean;
})
```

A square button with only an icon (`data-icon-only`). `label` is required,
so an icon-only control always has a name.

```tsx
<IconButton
  label={`Delete Run ${run.runId}`}
  icon={<Icon name="delete" />}
  variant="ghost"
  size="sm"
  onClick={onDelete}
/>
```

### `ButtonLink`

```ts
ButtonLink(props: LinkProps & {
  variant?: ButtonVariant; size?: ButtonSize; icon?: ReactNode;
  returnLabel?: string;           // a ContextLink: the target offers "Back to …"
})
```

Navigation that looks like a button ("Open check", "Start a check").

```tsx
<ButtonLink size="sm" to={checkPath(audit.projectId, audit.auditId)}>
  Open check
</ButtonLink>
```

## Dialogs

### `AlertDialog`

```ts
AlertDialog(props: {
  title: ReactNode;
  eyebrow?: ReactNode;            // small line above the title
  children: ReactNode;            // the description (aria-describedby)
  details?: ReactNode;            // facts, a type-to-confirm Field, notices
  error?: ReactNode;              // above the buttons, e.g. <ErrorNotice />
  confirmLabel: string;
  pendingLabel?: string;          // confirm text while pending
  cancelLabel?: string;           // default "Cancel"
  tone?: "danger" | "primary";    // default "danger": danger button and border
  pending?: boolean;
  confirmDisabled?: boolean;      // e.g. until a name is typed
  closeLabel?: string;            // adds a × button with this name
  dismissOnBackdrop?: boolean;    // default false
  className?: string;             // panel modifier, e.g. a width
  onConfirm: () => void;
  onCancel: () => void;
})
```

Every confirmation: removals, Stop and Delete of a check, Continue from
failed stage, Retry model connection, discarding a Run draft. It is a modal
`role="alertdialog"` on the shared `Dialog` (`src/app/dialog.tsx`): focus
starts on the safe action, Tab stays inside, Escape and Cancel close it, and
focus returns to the opener. Nothing is sent before the confirm button.
While `pending`, every control is disabled and Escape, the backdrop and the
close button do nothing.

```tsx
{
  confirming ? (
    <AlertDialog
      eyebrow="Permanent action"
      title="Delete completed Run?"
      confirmLabel="Delete Run"
      pendingLabel="Deleting…"
      pending={deletion.isPending}
      error={
        deletion.error === null ? null : <ErrorNotice error={deletion.error} />
      }
      onCancel={() => setConfirming(false)}
      onConfirm={() => deletion.mutate(runId)}
    >
      <p>This permanently removes the Run and its history.</p>
    </AlertDialog>
  ) : null;
}
```

Other modal content (forms, pickers) uses `Dialog` and `DialogHeader` from
`src/app/dialog.tsx` with the `ui-dialog ui-panel` classes.

## Facts and fields

### `Facts`

```ts
Facts(props: {
  items: readonly { term: ReactNode; value: ReactNode; key?: string; wide?: boolean }[];
  layout?: "grid" | "row" | "list" | "rows"; // default "grid" (columns of >= 10rem)
  size?: "sm" | "md";
  technical?: readonly Fact[];      // behind a Technical details disclosure
  technicalSummary?: string;
  className?: string;
  "aria-label"?: string;
})
```

A `<dl>` of term and value pairs, each pair a `<div>`. A fact whose value
is `undefined`, `null` or `false` is left out (0 and "" stay), so pages list
what they know without conditionals; with no fact left it renders nothing.
A `wide` fact takes a row of its own (a repository URL, a revision).

- **`grid`:** columns of at least 10rem; terms above values.
- **`row`:** pairs wrap on one line.
- **`list`:** pairs stacked.
- **`rows`:** terms in one column (8–13rem) beside the values, a hairline
  between pairs, values in regular weight: configuration and record
  details. It turns into one column at 640 px, or when a `.ui-facts-frame`
  around it is narrower than 30rem (a side panel, a nested value; a frame
  nested in such a list is then indented).

```tsx
<Facts
  layout="row"
  items={[
    { term: "Project", value: project.name },
    {
      term: "Started",
      value: audit.startedAt && formatShortDateTime(audit.startedAt),
    },
  ]}
  technical={[
    {
      term: "Check ID",
      value: <IdChip value={audit.auditId} label="check ID" />,
    },
  ]}
/>
```

### `Field`

```ts
Field(props: {
  label: ReactNode;
  hint?: ReactNode;
  error?: ReactNode;              // aria-invalid and an alert under the hint
  required?: boolean;             // required + aria-required, "*" in the label
  optional?: boolean;             // "(optional)" in the label
  id?: string;                    // the control's id; generated otherwise
  className?: string;
  children: (control: {
    id: string; "aria-describedby"?: string; "aria-invalid"?: true;
    "aria-required"?: true; required?: true;
  }) => ReactNode;
})
```

Label, control, hint and error, wired for assistive technology. Spread the
argument on the control; inputs, selects and textareas inside a Field get
the shared form-control look and focus ring (`src/app/ui-primitives.css`).

```tsx
<Field label="Project name" hint="Shown in lists." error={nameError} required>
  {(control) => (
    <input
      {...control}
      value={name}
      onChange={(event) => setName(event.currentTarget.value)}
    />
  )}
</Field>
```

## App helpers

Shared code that is not a component lives in `src/app`:

- **Icons, `src/app/icon.tsx`:** `<Icon name size? strokeWidth? label? className? />`
  draws every stroke icon of the UI (navigation, actions, chevrons,
  decisions, material kinds). It is decorative unless `label` names it.
  Add new icons there instead of an inline `<svg>`; charts and drawings
  are not icons.
- **Formatting, `src/app/format.ts`:** `plural(count, one, many?)`,
  `formatNumber(value, maxFractionDigits = 2, minFractionDigits = 0)`
  (counts and measurements, "1,204"), `formatCompact` ("1.2K", "1.5M"),
  `formatBytes`, `formatTimestamp` (absolute, hover titles),
  `formatShortDateTime` ("Oct 4, 23:51"), `formatClock`, `formatDay`
  ("Today", "Yesterday", "Oct 4"), `formatRelative` (records, minutes and
  up: `RecordedTime` uses it), `formatAge` (live observations to the
  second), `formatElapsed` (Run and step durations), `formatTimeLimit` (a
  limit in words), `formatSeconds` (measured seconds), `compactId`,
  `compactDigest`, `middleTruncate` and `compareVersions`. Every Intl
  formatter of the UI is defined there, with en-US conventions.
- **Query states, `src/app/query-view.tsx`:** `<QueryView query={useQuery(…)}>`
  renders loading ("Loading…" unless `loading` replaces it; `null` shows
  nothing), the error panel when nothing is loaded (with an `errorHint` line
  under it), a stale-data warning after a failed refetch, the `empty` state
  for data that `isEmpty` selects, and the content. For an infinite query a
  failed next page keeps the loaded pages and shows its error
  (`continuationContext`) under them, retrying only that page. Try again
  (`retryLabel`) calls the query's own `refetch` unless `onRetry` replaces
  it; `renderError(error, retry, stale)` replaces the error panel and the
  stale warning for an area whose errors need their own words, such as the
  Evals' revision conflicts.
- **Polling, `src/app/polling.ts`:** `pollInterval({ interval, settled?,
settledInterval?, maxInterval? })` returns a `refetchInterval` function:
  the base interval while the data can change, stop (or the slow
  `settledInterval`) once `settled(data)` holds, doubling waits while reads
  fail (up to `maxInterval`, 60 s by default) and no polling after a
  permanent refusal such as 404. A page that shows related data invalidates
  it when it sees the settled transition.

  ```ts
  refetchInterval: pollInterval<RunStatus>({
    interval: 2_000,
    settled: (run) => isTerminalRunState(run.state),
  }),
  ```

- **Write conflicts, `src/app/conflict.ts` and `conflict-notice.tsx`:**
  `writeConflict(error)` is `"stale"` for 412 (the revision changed first),
  `"conflict"` for 409 (the current state rules the request out) and
  undefined otherwise; `<ConflictNotice error subject="The check" />` says
  so under a form. Neither retries; the page reads the resource again.
- **Late navigation, `src/app/page-location.ts`:** `useGuardedNavigate()`
  returns a `navigate` that moves only while the user is still on the page
  that rendered it. Every navigation after an await or in a mutation
  callback uses it (AGENTS.md 3.8).
- **Vocabulary, `src/app/vocabulary.ts`:** labels and tones of Audit,
  coverage, finding, review and Run states. Run states keep the API words
  (Initializing, Pending, Running, Waiting, Cancelling, Succeeded, Failed,
  Cancelled); `runStatePhrase` and `runRecoveryReason` give the sentence
  forms. `labelOf(table, value)` and `wordOf(table, value)` read a value
  this client does not know yet as `readableValue(value)` ("release_pending"
  is "Release pending"). Material kinds are classified once in
  `src/routes/artifacts/kinds.ts`.

## Utility classes

- **`.ui-btn`:** buttons and links styled as in the mockups; use `Button`,
  `IconButton` or `ButtonLink` rather than writing the attributes.
  - `data-variant`: `primary` (filled with `--primary`), `secondary` (the
    default look), `danger`, `ghost`, `link` (reads as text).
  - `data-size`: `sm` (32 px) or `xs` (28 px); the default is 38 px.
  - `data-icon-only`: square, as wide as it is tall.
  - `aria-pressed="true"` gets the selected look; `aria-busy="true"` the
    progress cursor; disabled buttons, and those only `aria-disabled="true"`
    so they keep focus, are dimmed and show no hover.
  - Pressing moves it down 1 px; colours fade in 140 ms.
- **`.ui-stack`:** a column of children with one gap (12 px; `data-gap`
  1, 2, 4 or 6 picks another step of the spacing scale).
- **`.ui-toolbar`:** a wrapping row of controls (8 px gap; `data-gap` as
  above; `data-justify="between"` or `"end"`).
- **`.ui-panel`:** a bordered card on the page surface; dialogs are panels
  too (`ui-dialog ui-panel`).
- **`.ui-badge`:** a small neutral pill: a version, a scope, "Latest".
- **`.ui-control`:** a select or input outside a `Field` or form (a list
  filter, a material choice, a version picker): line border, surface
  background, focus ring. `data-size="sm"` is 32 px; `data-size="xs"` is a
  28 px monospace control as wide as its value (up to 12rem); the default
  is 38 px. Controls inside a `Field` get the form look without it.
- **`.ui-caption`:** secondary text (muted, 13 px).
- **`.ui-mono`:** identifiers, paths and digests inline in text.
- **`.ui-dialog`, `.ui-dialog-heading`, `.ui-dialog-actions`,
  `.ui-dialog-close`:** the modal panel of `Dialog`; below 640 px it is a
  bottom sheet.
- **`.ui-visually-hidden`:** hides text visually but keeps it for screen
  readers.
- **`.ui-footer-record`:** content of a `DetailPane` footer that shows a
  recorded decision. Where the footer is pinned it is capped and scrolls on
  its own (see `DetailPane`). The decision components set it and keep the
  content reachable from the keyboard while it scrolls.

## Tokens used

From `src/app/theme.css`, added for V3B:

| Tokens                             | Use                                      |
| ---------------------------------- | ---------------------------------------- |
| `--chrome`, `--chrome-2`           | Chips and the copy hover                 |
| `--faint`                          | Placeholders and faint text              |
| `--sel`, `--sel-bar`, `--sel-line` | Selected rows, pressed buttons and chips |
| `--code-bg`, `--code-hl`           | Code blocks and highlighted lines        |
| `--badge-bg`, `--badge-ink`        | Rail count badge                         |
| `--m-*`                            | Progress segments                        |

Also used: the base surface, line, text, status and primary tokens.

## Style scale

Sizes come from the scale in `src/app/theme.css`: the shared sheets use it,
and route CSS moves onto it as each area adopts the building blocks. The
steps were taken from the values pages used most, so mapping an old value
onto the nearest step changes at most half a pixel; round halves up (12.5 px
to 13 px).

| Tokens                                          | Values                                                                  |
| ----------------------------------------------- | ----------------------------------------------------------------------- |
| `--space-0-5` … `--space-12`                    | 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 32, 48 px (`--space-N` is N × 4 px) |
| `--radius-xs` … `--radius-2xl`, `--radius-full` | 4, 6, 8, 10, 12, 14 px; 999 px for pills                                |
| `--text-2xs` … `--text-xl`                      | 11, 12, 13, 14, 16, 20, 24 px                                           |
| `--control-xs`, `-sm`, `-md`, `-lg`             | 28, 32, 38 px button heights; 44 px touch target                        |

Breakpoints cannot be custom properties, so they are four fixed widths that
`make lint` enforces (`scripts/check-css.mjs`): **480, 640, 820 and 1100
px**. A narrow layout is `@media (max-width: N px)`; the wide side of the
same breakpoint is `(min-width: N+1 px)`. 820 px is the one-pane switch of
`PaneLayout` and the rail; 1100 px narrows the list pane; 640 px turns
dialogs into bottom sheets. Container queries (`@container`) size
components and may use any width.

Reduced motion is handled once in `ui.css`: every transition and animation
finishes at once, so pages need no `prefers-reduced-motion` rules of their
own unless they change more than motion.

Route CSS loads after the shared sheets, so shared classes never need
`!important`, and shared rules live only in `src/app/theme.css`,
`src/styles.css`, `src/app/ui-primitives.css` and `src/ui/ui.css` (eager),
never in a lazily loaded route stylesheet.

A bare `<button>` has no look of its own: `src/styles.css` only resets it
(no padding, border or background, the surrounding text colour and font, a
pointer cursor, `not-allowed` while disabled). `.ui-btn` carries the whole
`Button` look, including the colour transition, the 1 px press and the
dimmed disabled state. Any other button (a tab, a list row, a chip, a menu
item, a canvas handle) gets its look from its own class, so no rule has to
undo a default: write what the control needs, not `min-height: 0`,
`transform: none` or a `:not(:disabled):hover` that only restores a
background.

## Gates

`make lint` runs them as the `ui-gates` target (it needs `make
ui-install`); `pnpm lint` runs the ESLint part. They fail on:

- **Hand-written building blocks** (ESLint rule `building-blocks/use-shared`,
  `scripts/eslint-building-blocks.mjs`): outside `src/ui` and `src/app`, a
  `<button>` with the `ui-btn` class, `role="alertdialog"` or an inline
  `<svg>`. `scripts/building-blocks-baseline.json` lists the files that
  still have them, with counts; a file may not exceed its count, and a
  count higher than the file needs fails too, so lower it in the change
  that adopts the building block. Charts and drawings are listed under
  `drawings` with a reason.
- **Unreferenced CSS** (`scripts/check-css.mjs`): a class that no
  production source names in a string or template literal, and an `@media`
  width outside the breakpoints. A class built at run time
  (`diff-line-${kind}`) or rendered by a library goes into
  `scripts/css-allowlist.json` with its source; stale entries fail.
- **Needless exports** (`scripts/check-exports.mjs`): an export of a
  production module that no other production module imports, whether only
  tests import it or nothing does (an export used only in its own module
  should not be exported). Test helpers live in `src/test` or a
  `test-support.tsx` file, and production code must not import them.
  `scripts/exports-baseline.txt` lists tolerated entries against an open
  task; an entry that is no longer needed fails, so it only shrinks.
