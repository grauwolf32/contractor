package auditstore

import "strings"

// The scan functions read each entity's columns in one fixed order. The lists
// below hold that order once, and every statement selects an entity's columns
// through its helper, qualified by the table or alias it reads, so a new
// column is added here and in the scan function only.

// auditColumnList is the scanAudit order. The event sequence is the last
// sequence number used, one less than the next one stored.
var auditColumnList = []string{
	"audit_id", "owner_id", "project_id",
	"profile_name", "profile_version", "profile_digest", "profile_snapshot", "input_selection",
	"baseline_snapshot", "state", "revision", "current_round_id", "dispatch_state", "hold_state",
	"deadline_at", "max_rounds", "batch_size", "max_items_per_round", "max_items_total",
	"max_submitted_runs", "max_item_run_attempts", "max_evidence_bytes",
	"reserved_run_count", "submitted_run_count", "outstanding_run_count", "retained_evidence_bytes",
	"next_event_sequence - 1", "stop_reason_code", "stop_reason_message",
	"created_at", "updated_at", "started_at", "finished_at", "deletion_requested_at", "paused_at", "phase",
}

// executionColumnList is the scanExecution order.
var executionColumnList = []string{
	"execution_id", "audit_id", "round_id", "role", "workflow_role", "role_attempt",
	"manifest_ref", "manifest_digest", "submission_key", "request_digest", "run_id",
	"state", "terminal_outcome", "terminal_run_generation", "terminal_run_sequence",
	"terminal_observed_at", "run_provenance", "run_deleted_at", "created_at", "updated_at",
	"preparation_snapshot", "preparation_outputs",
}

// roundColumnList is the scanRound order.
var roundColumnList = []string{
	"round_id", "audit_id", "ordinal", "manifest_ref", "manifest_digest",
	"state", "expected_item_count", "revision", "created_at", "updated_at",
}

// itemColumnList is the scanItem order.
var itemColumnList = []string{
	"item_id", "audit_id", "round_id", "item_key", "ordinal", "kind", "subject_key",
	"task_ref", "task_digest", "origin", "workflow_role", "state",
	"approval_kind", "approval_subject_digest",
	"final_disposition", "accepted_result_ref", "accepted_result_digest",
	"last_execution_item_id", "created_at", "updated_at",
}

// receiptColumn is one column of a collection receipt: the AuditExecution
// column that stores it and the name a receipt row gives it.
type receiptColumn struct {
	source, name string
	// detail marks the source output and retained references that
	// scanReceiptSummary leaves out.
	detail bool
}

// receiptColumnList is the scanReceipt order; without its detail columns it
// is the scanReceiptSummary order.
var receiptColumnList = []receiptColumn{
	{source: "collection_receipt_id", name: "receipt_id"},
	{source: "audit_id", name: "audit_id"},
	{source: "execution_id", name: "execution_id"},
	{source: "run_id", name: "run_id"},
	{source: "terminal_outcome", name: "terminal_outcome"},
	{source: "terminal_run_generation", name: "terminal_run_generation"},
	{source: "terminal_run_sequence", name: "terminal_run_sequence"},
	{source: "collection_disposition", name: "disposition"},
	{source: "collection_source_output_ref", name: "source_output_ref", detail: true},
	{source: "collection_source_output_digest", name: "source_output_digest", detail: true},
	{source: "collection_retained_refs", name: "retained_refs", detail: true},
	{source: "collection_error_code", name: "error_code"},
	{source: "collection_request_digest", name: "request_digest"},
	{source: "collected_at", name: "created_at"},
}

// qualifiedColumns joins columns, each qualified by the table or alias prefix.
func qualifiedColumns(prefix string, columns []string) string {
	qualified := make([]string, len(columns))
	for index, column := range columns {
		qualified[index] = prefix + "." + column
	}
	return strings.Join(qualified, ", ")
}

func prefixedAuditColumns(prefix string) string {
	return qualifiedColumns(prefix, auditColumnList)
}

func prefixedExecutionColumns(prefix string) string {
	return qualifiedColumns(prefix, executionColumnList)
}

func prefixedRoundColumns(prefix string) string {
	return qualifiedColumns(prefix, roundColumnList)
}

func prefixedItemColumns(prefix string) string {
	return qualifiedColumns(prefix, itemColumnList)
}

// prefixedReceiptColumns selects receipt-named columns, such as those
// collectionReceiptProjection returns, in scanReceipt order.
func prefixedReceiptColumns(prefix string) string {
	names := make([]string, len(receiptColumnList))
	for index, column := range receiptColumnList {
		names[index] = column.name
	}
	return qualifiedColumns(prefix, names)
}

// collectionReceiptProjection selects a collected AuditExecution aliased as
// its collection receipt, in scanReceipt order.
func collectionReceiptProjection(alias string) string {
	projected := make([]string, len(receiptColumnList))
	for index, column := range receiptColumnList {
		projected[index] = alias + "." + column.source + " AS " + column.name
	}
	return strings.Join(projected, ", ")
}

// receiptSummaryColumns selects a collected AuditExecution's receipt in
// scanReceiptSummary order, without its source output and retained
// references.
func receiptSummaryColumns(alias string) string {
	sources := make([]string, 0, len(receiptColumnList))
	for _, column := range receiptColumnList {
		if !column.detail {
			sources = append(sources, column.source)
		}
	}
	return qualifiedColumns(alias, sources)
}

// withColumns lets the scan function of one entity's column list also read
// the columns a statement selects before and after that list.
type withColumns struct {
	row           scanner
	before, after []any
}

func (r withColumns) Scan(dest ...any) error {
	all := make([]any, 0, len(r.before)+len(dest)+len(r.after))
	all = append(all, r.before...)
	all = append(all, dest...)
	return r.row.Scan(append(all, r.after...)...)
}
