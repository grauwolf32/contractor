# Shared UI modules

Modules own reusable data queries and subject-specific rules. They consume the
public API and shared infrastructure; they do not import routes, widgets,
visual controls or router state. Routes supply selected IDs, cursors and
filters, then compose widgets with the returned data. Modules do not redefine
Server contracts or add a separate store.

- `checks`: shared Audit/check reads and display rules. See its README for
  query and route ownership.

`scripts/layer-boundaries.test.mjs` checks the production import boundary,
including lazy imports. Keep concrete public files rather than a universal
barrel; imports should reveal the required responsibility.
