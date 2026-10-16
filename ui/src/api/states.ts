import { schemaEnum } from "./contract";
import type { components } from "./generated/public";

/**
 * The one UI definition of Audit and Run state lists. Every list is typed by
 * the generated enum, and states.test.ts ties each one to the OpenAPI
 * document; pages import these names instead of writing states out.
 */

type AuditState = components["schemas"]["AuditState"];
type WorkflowRunState = components["schemas"]["WorkflowRunState"];
type NonTerminalRunState = components["schemas"]["NonTerminalWorkflowRunState"];
type TerminalRunState = Exclude<WorkflowRunState, NonTerminalRunState>;

/** Every Run state, in contract order. */
export const RUN_STATES = schemaEnum<WorkflowRunState>("WorkflowRunState");

/** Run states that can still change: the states the Queue lists. */
export const NON_TERMINAL_RUN_STATES = schemaEnum<NonTerminalRunState>(
  "NonTerminalWorkflowRunState",
);

/** Run states that never change again. */
export const TERMINAL_RUN_STATES: readonly TerminalRunState[] = Object.freeze(
  RUN_STATES.filter(
    (state): state is TerminalRunState =>
      !(NON_TERMINAL_RUN_STATES as readonly string[]).includes(state),
  ),
);

/** Every Audit state, in contract order. */
export const AUDIT_STATES = schemaEnum<AuditState>("AuditState");

/** Audits that ran and can hold possible issues: all but draft and deleting. */
export const AUDIT_STATES_WITH_FINDINGS: readonly AuditState[] = Object.freeze(
  AUDIT_STATES.filter((state) => state !== "draft" && state !== "deleting"),
);

/** Audits working or writing their report; they can still propose issues. */
export const WORKING_AUDIT_STATES = [
  "active",
  "finalizing",
] as const satisfies readonly AuditState[];

/** Audits running or waiting for the user: the Inbox's checks. */
export const INBOX_AUDIT_STATES = [
  "active",
  "waiting_review",
] as const satisfies readonly AuditState[];

/** Audits the Inbox lists in one of its sections. */
export const INBOX_LISTED_AUDIT_STATES = [
  "active",
  "waiting_review",
  "paused",
  "finalizing",
  "completed",
  "failed",
] as const satisfies readonly AuditState[];

/**
 * Started Audits that are not stopping or ended: they accept Stop, can ask
 * for decisions and keep live workspace counters.
 */
export const OPEN_AUDIT_STATES = [
  "active",
  "waiting_review",
  "paused",
  "finalizing",
] as const satisfies readonly AuditState[];

/**
 * Started Audits that neither wait for the user nor have ended: what the
 * Checks list calls running.
 */
export const IN_PROGRESS_AUDIT_STATES = [
  "active",
  "paused",
  "finalizing",
  "cancelling",
] as const satisfies readonly AuditState[];

/** Live Audits a summary names, in display order: running first. */
export const LIVE_AUDIT_STATES = [
  "active",
  "finalizing",
  "cancelling",
  "waiting_review",
] as const satisfies readonly AuditState[];

/** Audits that can have a report: waiting, finishing or ended. */
export const REPORT_AUDIT_STATES = [
  "waiting_review",
  "finalizing",
  "completed",
  "cancelled",
  "failed",
] as const satisfies readonly AuditState[];

/** Audits that ended without completing. */
export const STOPPED_AUDIT_STATES = [
  "cancelled",
  "failed",
] as const satisfies readonly AuditState[];

/** Audits that ended, or are being deleted after they ended or stopped. */
export const ENDED_AUDIT_STATES = [
  "completed",
  ...STOPPED_AUDIT_STATES,
  "deleting",
] as const satisfies readonly AuditState[];

/** Audits that change without the user acting, so views poll them. */
export const CHANGING_AUDIT_STATES = [
  "active",
  "waiting_review",
  "finalizing",
  "cancelling",
  "deleting",
] as const satisfies readonly AuditState[];

/** Whether `state` is one of `states`, for a narrowly typed list. */
export function isStateIn<S extends string>(
  states: readonly S[],
  state: string,
): state is S {
  return (states as readonly string[]).includes(state);
}
