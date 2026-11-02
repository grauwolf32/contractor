# Review of the React transfer

Raw OpenDesign v11 and v12 outputs are preserved unchanged here. The actual
React UI, measured snapshots, prompts, event logs and original/intermediate
HTML remain in the private host archive recorded in review.json. The HTML
comparison refers to that archive’s local reference directory.

The integrated Issue route applies the reviewed presentation through its own
components and shared Artifact/decision owners. Generated claims were checked
against full source: v12’s statement that duplicate styles were absent was
incorrect (the brief held an excerpt). The implementation resets existing
card styles explicitly and retains lazy reads, independent decisions, paging
and stale-error handling. Forced hidden-node rules and display:contents were
not ported. User refinements after generation remain authoritative.
