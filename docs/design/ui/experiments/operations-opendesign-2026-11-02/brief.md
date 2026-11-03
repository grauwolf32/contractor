# Contractor · Operations overview · design brief

User chose Operations after approving the New check design. This is a new
preview for review, not a production transfer. Use the same app font families,
neutral surfaces, theme tokens and supplied icons as the recent Contractor UI.

## What the operator needs

1. Scan current slot distribution and whether something merits inspection.
2. Understand Runtime process occupancy and temporary allocation roles.
3. Distinguish a reserved assignment, an observed prepared role, draining and
   fenced/reconciliation states without interpreting them as a successful Run.
4. Find the exact process/allocation/Run and their technical facts on demand.
5. Reach all Operations sections through clearly labelled navigation.

The current real React page is flat: Execution readiness, a slot legend and
three process/role tiles. The proposed functional seed uses the same snapshot
facts, adds bounded read-only drill-downs and supplies genuine current captures.
Give it a clear visual rhythm; avoid a grid of interchangeable cards or a wall
of warnings. This should look like a calm, precise operational workspace.

## Suggested visual emphasis (agent proposal)

- Keep Operations heading concise with refresh and a small live-transport badge.
- Label the internal section navigation. Distinguish observation from Setup.
  Collapse it cleanly on phones; do not leave an icon-only mystery rail.
- Use a compact summary band; slot distribution is the visual anchor. Give the
  inspection area enough emphasis to find real reconciliation facts quickly.
- Use existing state colors for slot segments, icons and pills, always with text.
- Runtime process rows and current allocation rows should feel related but
  distinct. Make their disclosures easy to discover. Keep details secondary.
- Separate authoritative allocation phase from Runtime observation in every row.
- Use spacing, alignment, dividers and type weight before adding decoration.
- No neon, invented graphs, decorative gauges, background art or new assets.

## Non-negotiable data boundaries (OPS-R20–43)

Only the supplied OperationsSnapshot fixtures define the values. They validate
against the current public API schema. Keep the controller and fixtures byte
identical. Do not invent queue sizes, CPU/GPU data, historical trends, recent
activity, durable principal labels, throughput, capacity percentages or events.

- One process has one slot. Idle is not proof of compatible Run capacity.
- Snapshot process entries are not the inventory of durable saved principals.
- Observed `allocated`/`prepared` proves preparation, not a Worker call in flight.
- Slot states: idle, busy, reserved, draining, fenced. They are not Run outcomes.
- Reconciliation count matches current Overview: a reported reason OR differing
  observed/authoritative allocation IDs. A reservation awaiting observation is
  explicitly qualified, not claimed to be a failed allocation.
- Live connection is transport freshness, not overall deployment health.
- Read failure has unknown counts, never fabricated zero or green health.
- Reconnecting preserves the last snapshot and its warning. Empty is explicit.
- IDs, accepted heartbeat, confirmed lease and exact capabilities remain visible
  in disclosures; missing heartbeat/lease remain “Not reported”.
- Live metrics stay incomplete until final reports; do not turn default zeros
  into a performance dashboard. Full capabilities are provenance, not endpoint
  health or compatibility with every Workflow.
- This preview does not expose execution/configuration mutation actions.

## Input and output

Read only reference/operations-brief.md and reference/operations-seed.html.
Reference/current-attention-1440.png and current-attention-390.png show the real
React baseline. Work only in this new project. Preserve all previous projects,
files and conversations. Do not read Skills, discover other folders, install
anything, use remote assets, change the model or call external endpoints.

The seed is 63 KiB of self-contained HTML. It includes exact shared theme CSS,
shared icon markup, a single local controller, and schema-valid normal,
attention, empty, reconnecting and read-error fixtures. CSS is your contribution:
write NEW `opendesign.css`, around 180–260 readable lines of responsive overrides.
Use only existing CSS custom properties for colors and the supplied font
families. Keep all essential data and controls visible/reachable; do not use
pseudo-content to invent metrics. Use 1100/820/640/480px breakpoints as needed.

Then use Python to insert that CSS as a NEW `<style id="opendesign">` immediately
before `</head>` of the seed, without changing any other byte. Write the result
as NEW `index-operations-v1.html`. Write NEW `rationale.md` in 4–6 concrete
sentences describing your hierarchy and mobile treatment. No dependency work,
browser QA, long planning or edits of the seed. Stop after producing these files.

We will run browser checks externally in light/dark/black at 1440/1100/390/320px,
all five cases, disclosures, labelled navigation, exact counts and zero overflow.
