// Decision and finding components shared by Inbox, Issues, Checks and
// Reports (docs/design/ui/v3b-build-contract.md §7). Styles: ./decisions.css,
// imported by the components.
export { FindingSummary, type FindingSummaryProps } from "./finding-summary";
export {
  FindingDecision,
  type FindingDecisionNext,
  type FindingDecisionProps,
} from "./finding-decision";
export {
  ActionDecision,
  ReportDecision,
  type ActionDecisionProps,
  type ReportDecisionProps,
} from "./review-decision";
export { DecisionRecord } from "./decision-record";
export type {
  AuditActionDecisionResult,
  AuditFindingDecisionResult,
  AuditReviewDecision,
} from "./model";
