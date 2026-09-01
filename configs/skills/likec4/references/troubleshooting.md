---
description: "Common LikeC4 mistakes and fixes — read when hitting validation errors, unexpected rendering, or a failing eval answer."
---

# Common Mistakes & Debugging (LikeC4 DSL)

Load this file when encountering validation errors, unexpected rendering, or when an eval answer is failing.

## Syntax Errors

| Error | Cause | Fix |
|-------|-------|-----|
| "Identifier PAYMENT.API not found" | Dots in identifier name | Use `payment-api` not `payment.api`; dots are FQN separators only |
| "Unknown kind SERVICE" | Kind not in specification | Define `element service { ... }` in the specification block |
| "Duplicate FQN cloud.backend" | Element ID repeated under the same parent | Rename one element; each sibling must have a unique identifier |
| "Expected property TYPE after include *" | Malformed filter predicate | Use `include * where kind is component`, not `include * component` |
| "Invalid relationship kind async-cache" | Kind not found in specification | Define `relationship async-cache { ... }` in specification first |

## Model & Hierarchy

| Error | Cause | Fix |
|-------|-------|-----|
| Element shows but relationships don't render | Relationship references FQN incorrectly | Use exact FQN matching the model hierarchy |
| "Can't define relationship from parent to child" | Direct parent-child relationships are forbidden | Move the relationship outside the parent element or use implicit notation |
| Child element not visible from other files | Referencing by short name instead of FQN | Import or use full FQN: `cloud.backend.api` |
| Extend block adds duplicate tags | Tags stack on merge | Use consistent tag names; duplicates are not deduplicated automatically |

## View Predicates

| Error | Cause | Fix |
|-------|-------|-----|
| `include *` shows grandchildren | Confusing `*` with `**` | `*` = direct children only; use `**` for recursive descent |
| Relationships disappear in scoped view | Neighbors not explicitly included | Add `include -> scope` or `include <-> scope` to pull in inbound/outbound sources |
| "WHERE predicate not matching any elements" | Tag or kind name is case-sensitive | Use exact case: `#Critical` ≠ `#critical` |
| Element included but styled differently | Global vs. local style conflict | Local view styles override global; audit style rules order in the view block |

## Deployment

| Error | Cause | Fix |
|-------|-------|-----|
| `instanceOf` doesn't resolve | Instance refers to wrong FQN | Use the exact logical model FQN, not the deployment node identifier |
| "Undefined DEPLOYMENT_KIND" | Kind referenced but not in specification | Define `deploymentNode vm { ... }` in specification |
| Deployment relationship inherits unexpectedly | Logical model edges are inherited automatically | Suppress inherited relationships explicitly in the deployment view if unwanted |

## Dynamic Views

| Error | Cause | Fix |
|-------|-------|-----|
| Parallel block renders incorrectly | Nested `parallel { parallel { ... } }` | Flatten: put all concurrent steps in a single `parallel { }` block |
| Response arrows (`<-`) show wrong direction | Chaining mixes `->` and `<-` inconsistently | Use symmetric chains: `a -> b -> c` then `c <- b <- a` for returns |
| `navigateTo` link doesn't work | Target view name does not exist | Ensure the target view name exists in the same project |
| `variant sequence` is ignored | Wrong keyword or spelling | Use exact: `variant sequence` (not `type`, `mode`, or `sequence` alone) |

## Validation & Import

| Condition | Meaning | Required action |
|---|---|---|
| `validatorAvailable` is false | The Runtime Agent did not advertise the `validate_likec4` capability | Return retryable failure; do not treat the empty issue list as clean |
| `validatorExecutionError` is non-null | The installed validator failed, timed out, or returned invalid output | Return retryable failure and preserve the reported execution reason |
| `issuesTruncated` is true | More diagnostics existed than the bounded result could return | Repair the visible root causes, but never claim the document is clean until a later call returns `valid: true` |

## Performance & Large Models

| Symptom | Cause | Action |
|---|---|---|
| Validation takes >30s | The bounded validator invocation times out | Simplify the single document or return retryable failure; do not bypass validation |
| A requested export fails | Export is not exposed by the assigned Toolset | Report the unsupported capability; do not claim an image was generated |
| One model is too large to repair safely | A whole-document rewrite would be risky | Use bounded reads and unique replacements; return retryable failure if a safe repair is impossible |

## Debugging Workflow

When encountering errors, follow these steps in order:

1. **Run exact-artifact validation:** persist the current document, then call `validate_likec4()` with no arguments. Continue only when `valid` is true; unavailable, failed, or truncated validation is not clean.

2. **Check FQN integrity:** Find the identifier from the error message and verify it matches the model hierarchy exactly.

3. **Isolate predicate issues:** Copy the failing `include`/`exclude` rule into a new minimal test view to confirm predicate semantics in isolation.

4. **Validate specification first:** Comment out `model`, `deployment`, and `views`; validate the `specification` block alone. If it passes, uncomment the next block and repeat.

5. **Use `extend` strategically:** When enriching an existing element, use `extend FQN { }` rather than re-declaring it; re-declarations cause duplicate FQN errors.

## Skill Best Practices

1. **Always start with project structure understanding.** Use the exact source artifact and supplied analysis reports before changing the architecture model.
2. **For relationship ambiguity, always include KIND + TITLE.** When source/target/kind match multiple relationships, include title in the matcher to avoid silently targeting the wrong one.
3. **Response discipline on strict prompts.** Output one final answer first (no alternatives) when the prompt says "exact", "minimal", or "paste-ready".
4. **Validate after each persisted phase.** Call `validate_likec4()` only after `load_likec4` or `write_likec4`; it validates the current exact artifact.
5. **Use precise references.** Prefer an FQN when a short nested identifier could be ambiguous.
6. **Keep specification stable.** In Contractor it lives in the same document, but changes can invalidate the whole model; edit it deliberately.
7. **Test wildcard semantics locally.** When unsure whether `*` or `**` is correct, add a minimal temporary view to the current document, validate it, then remove it before success.
