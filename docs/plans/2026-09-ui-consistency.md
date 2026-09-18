# UI improvements — implementation and review plan

Requested sequence, 2026-09-19:

1. Implement the approved Project header and shared Workflow cards; commit on main.
   - Contextual Project actions and a compact secondary menu.
   - One Workflow family per card; explicit selected version, latest numeric
     version by default, previous versions available, no silent selection change.
   - Catalog contracts remain visible in a compact layout; Project cards show
     artifact matching, ambiguity and the selected version's primary outputs.
   - Preserve exact-version links, Run drafts, all published versions and pagination.
2. Inventory all route families, record consistency findings, apply shared UI fixes
   and commit on main. Review headings, actions, navigation, forms, tables,
   loading/error/empty states, mobile layout and keyboard interaction.
3. Create a separate usability branch. Map every route family to US-01…US-11,
   record evidence and priorities, implement actionable journey improvements,
   validate and deploy that branch to the stand. Keep separately scoped Evals
   experiment work distinct from the currently supported label-based grouping.

Validation: targeted behavior tests for version selection, exact links and
context actions; full UI tests, typecheck, lint/format and production build;
read-only browser review against stand data on desktop/mobile. Deployment
preserves Server, Runtime processes and the current scheduler state.

## Progress

- Planning: approved design and existing User Stories/specifications reviewed.
- Implementation and review findings will be recorded below as each stage completes.

### Stage 1 — implemented

Shared WorkflowCard and version grouping now serve both Project and Catalog.
Catalog reads complete inventory with cursor-cycle validation; Project matching
uses the same complete Workflow list. Numeric components compare as arbitrary
precision integers. Opaque labels have no Latest badge. Older explicit choices
survive refresh, and configuration opens the exact selected version.

Project actions moved to their sections; confirmed deletion remains in a compact
menu. Catalog cards expose compact contracts without placeholder descriptions.
No immutable published definition was rewritten to invent presentation metadata.

Verification: 53 files / 340 tests passed; typecheck, targeted ESLint, production
build and diff whitespace checks passed. Live read-only browser review: 15 catalog
families from 33 versions, desktop/mobile catalog fits, no JS errors or writes.
Project mobile has pre-existing horizontal overflow; tracked for stage 2.
