# Shared UI building blocks (`src/ui`)

The V3B list + detail primitives. Import everything from `src/ui`:

```tsx
import { ListPane, ListRow, PaneLayout, StatusGlyph } from "../../ui";
```

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

- **≥ 1100 px:** the list is 360 px wide.
- **821–1099 px:** the list is 300 px wide.
- **Above 820 px:** the frame is `calc(100dvh - var(--shell-topbar-height, 0px))`
  tall and each pane scrolls on its own.
- **≤ 820 px:** one pane at a time in normal page flow. The list shows by
  default; with `showDetail` the detail shows, with `← backLink.label` above it.
  Switching panes moves focus to the pane that appears (or back to the
  selected row).

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
  children: ReactNode;        // ListRow elements
})
```

A `<section>` labelled by its small uppercase heading, then
`<ul role="list">`.

```tsx
<ListSection title="Endpoints" count={5} aside="under /workshop/api/">
  {rows}
</ListSection>
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
})
```

Renders an `<li>`. The title link or button covers the whole row (one click
target, one focus ring). Inline actions in `children` and `trailing` are
separate controls above it: never put a button inside the title. The glyph is
decorative, so say the status in words in `meta`. With a method chip, keep a
space before the path so the name reads "GET /path".

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
"label count", e.g. "Needs review 1".

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

### `DetailHeader`

```ts
DetailHeader(props: {
  breadcrumb?: readonly { label: ReactNode; to?: To }[];
  title: ReactNode;
  titleAs?: "h1" | "h2";      // default "h2"
  status?: ReactNode;         // usually StatusChip
  meta?: ReactNode;
  actions?: ReactNode;
})
```

Breadcrumb items with `to` are links; a last item without `to` gets
`aria-current="page"`. The title is 20 px in a list pane and up to 28 px in a
wide detail pane. It also serves as a list pane `header`, for example the
check page's breadcrumb, h1 and state.

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

One `role="img"` element with equal-width bars, one per item in list order,
3 px apart. Gaps shrink above 40 and 80 items. Colours: done/success `--m-done`,
partial/review/warning `--m-partial`, progress/info `--m-progress`, blocked
`--m-blocked`, idle/neutral `--m-idle`.

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
<MethodChip method="get" /> / workshop / api / mechanic / mechanic_report;
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
IdChip(props: { value: string; label: string; display?: string })
```

Shows `display`, or the value shortened to its first 8 and last 4
characters (`shortenId`). Values up to 16 characters are kept whole. The
full value is the hover title. The copy button is named "Copy " + label. It
uses `navigator.clipboard.writeText` and announces "Copied". When the
Clipboard API is missing or refuses, it selects the full value in a hidden
element and says "Press Ctrl+C to copy". Both messages use an
`aria-live="polite"` region.

```tsx
<IdChip value={audit.id} label="check ID" />
<IdChip value={`${ref.name}@${ref.version}`} display={`${ref.name}@${ref.version}`} label="workflow version" />
```

## Content

### `TechnicalDetails`

```ts
TechnicalDetails(props: { summary?: string /* "Technical details" */; description?: ReactNode; children: ReactNode; defaultOpen?: boolean })
```

A native `<details>` styled as a quiet link. Revisions, digests, rounds,
slots, tokens and allocations go here.

```tsx
<TechnicalDetails description="Steps, limits and AI usage, for admins and debugging.">
  <dl>…</dl>
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
- **`pending`:** disables every control, turns off the keys, makes the reason
  read-only (focus stays) and makes `more` and `extra` inert.
- **`next`:** a quiet button. Its key is bound as well; if the page already
  binds the same key to the same action, only one handler runs.

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
  bindings: Record<string, (event: KeyboardEvent) => void>,
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
  The handler then only reacts to keys pressed inside a dialog.

A binding that fires calls `preventDefault`, so when two hooks bind the same
key the first listener wins. Handlers can change on every render without
re-subscribing.

```tsx
useShortcuts({ "mod+k": openPalette, "?": showHelp });
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
  onMove: (next) =>
    navigate(`/issues/${items[next].auditId}/${items[next].id}`),
});
return (
  <div {...containerProps}>
    <ListSection title="Needs review">{rows}</ListSection>
  </div>
);
```

### Helpers

- `isTextEntryTarget(target)`: true for text inputs, textareas, selects and
  contenteditable regions.
- `modKeyLabel()`: `"⌘"` on Apple platforms, else `"Ctrl"`, for key hints.
  `isApplePlatform()` is the underlying test.
- `shortenId(value)`: the IdChip short form.
- `clockTime(value)`: the ActivityLog time.

## Utility classes

- **`.ui-btn`:** buttons and links styled as in the mockups.
  - `data-variant`: `primary` (filled with `--primary`), `secondary` (the
    default look), `danger`, `ghost`.
  - `data-size`: `sm` (32 px) or `xs` (28 px); the default is 38 px.
  - `aria-pressed="true"` gets the selected look.
- **`.ui-visually-hidden`:** hides text visually but keeps it for screen
  readers.

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
