# UI Dialog migration inventory

This inventory records the modal implementations inspected for V37-001. The
shared primitive is `ui/src/app/dialog.tsx`; new modal work should use it rather
than add another document-level Escape listener or focus trap.

| Surface                          | Previous behavior                                                                        | V37 ownership                                                                                  |
| -------------------------------- | ---------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Project Workflow launcher        | Inline modal markup, document Escape listener, no focus entry/containment or restoration | Migrated in V37-001                                                                            |
| Git repository import            | Body portal with a private focus trap and document Escape listener                       | Migrated in V37-001; this is the nested child used by the Run form                             |
| Skill ZIP upload                 | Private focus trap, scroll lock and restoration                                          | Follow-up migration; behavior is already bounded and V37-001 does not change Skill publication |
| Project Artifact upload          | Inline modal and document Escape listener                                                | Migrated in V37-003 when the Project header gained the direct Add sources action               |
| HTTP target editor               | Inline modal and pending-aware document Escape listener                                  | Follow-up migration; credential and target semantics remain unchanged                          |
| Project deletion                 | Inline `alertdialog` with pending-aware Escape                                           | Migrated in V37-003; typed confirmation and caller-owned pending refusal remain unchanged      |
| completed Run deletion           | Inline `alertdialog` with pending-aware Escape                                           | Follow-up migration; retain terminal-release and deletion rules                                |
| Audit cancel/delete confirmation | No shared confirmation at the V37 baseline                                               | V37-005 introduces it using the shared primitive                                               |

The primitive renders each layer into a direct `document.body` portal. The
application and lower layers become `inert` and `aria-hidden`, body scrolling is
locked once for the stack, and only the newest layer handles Tab or Escape.
Escape calls `onRequestClose`; it does not decide whether pending work may be
cancelled. Focus returns to the live trigger, then to the remaining parent
Dialog, then to a visible document control. The Dialog boundary also stops a
handled portal form submission from reaching an enclosing React form.

Backdrop activation does not close a Dialog. This avoids accidental data loss
and leaves close/discard policy with each caller. A later migration must preserve
the existing pending, confirmation, redaction and mutation semantics of the
surface being moved.
