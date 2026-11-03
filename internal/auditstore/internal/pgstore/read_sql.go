package pgstore

// SQL statements for read.go.

// listItemsPageSQL returns one page of the items of owner $1's Audit $2 in
// round ordinal, item ordinal and item ID order. It filters before the
// keyset: only round $3, state $4 and subject key $5, each when it is not
// NULL. ($6, $7, $8) is the (round ordinal, item ordinal, item ID) cursor of
// the previous page; a NULL $6 starts at the first page. At most $9 rows.
// Each of the Audit's rounds, at most auditprofile.MaxAuditRounds (32, the
// audits_max_rounds_check), contributes at most $9 candidates in item order
// through audit_items_round_ordinal, so a page reads at most 32 * $9 items
// however late its cursor; a sparse subject filter still reads every item it
// rejects. A state other than settled reads through listItemsStatePageSQL
// instead. Used by QueryStore.ListItemsPage.
var listItemsPageSQL = itemPageSQL(`candidate.round_id = round.round_id`)

// listItemsStatePageSQL is listItemsPageSQL for a state $4 other than
// settled: each round's candidates come from audit_items_dispatch_idx, which
// holds only unsettled items, so a sparse state such as awaiting_review reads
// only its own items and not every item of each round.
// Used by QueryStore.ListItemsPage.
var listItemsStatePageSQL = itemPageSQL(`candidate.audit_id = $2 AND candidate.state = $4
          AND candidate.state <> 'settled' AND candidate.round_id = round.round_id`)

// itemPageSQL is the item page of listItemsPageSQL whose candidates of each
// round satisfy access. The outer read walks the owner's Audit's rounds in
// ordinal order, from the cursor's round on; each round contributes its
// first $9 items after the cursor that pass the filters, and the page keeps
// the first $9 of those candidates. Items of one round have distinct
// ordinals, so the per-round keyset is a range on the item ordinal; the exact
// row comparison then drops the cursor item itself.
func itemPageSQL(access string) string {
	return `
SELECT ` + prefixedItemColumns("item") + `
  FROM audits AS audit
  JOIN audit_rounds AS round ON round.audit_id = audit.audit_id
 CROSS JOIN LATERAL (
       SELECT candidate.*
         FROM audit_items AS candidate
        WHERE ` + access + `
          AND candidate.ordinal >= CASE WHEN round.ordinal = $6 THEN $7 ELSE 0 END
          AND ($4::text IS NULL OR candidate.state = $4)
          AND ($5::text IS NULL OR candidate.subject_key = $5)
          AND ($6::integer IS NULL OR (round.ordinal, candidate.ordinal, candidate.item_id) > ($6, $7, $8))
        ORDER BY candidate.ordinal, candidate.item_id
        LIMIT $9) AS item
 WHERE audit.owner_id = $1 AND audit.audit_id = $2
   AND round.state <> 'proposed'
   AND ($3::text IS NULL OR round.round_id = $3)
   AND ($6::integer IS NULL OR round.ordinal >= $6)
 ORDER BY round.ordinal, item.ordinal, item.item_id
 LIMIT $9`
}

// uncollectedExecutionStatesSQL lists the execution states before
// collection, every auditstore.ExecutionState but collected
// (TestUncollectedExecutionStatesSQLMatchesDeclaration), which is also the
// predicate of audit_executions_reconcile_idx. Reads name these states rather
// than write state <> 'collected': the planner does not reason with the state
// CHECK, so only the named list proves that the partial index holds every
// row the read asks for. Used by uncollectedExecutionSQL and
// ClaimStore.listWindowExecutions.
const uncollectedExecutionStatesSQL = `('intent', 'submitted', 'collecting')`

// uncollectedExecutionSQL holds when the Audit aliased audit has an execution
// that is not collected yet. It is one probe of audit_executions_reconcile_idx,
// where state <> 'collected' read every collected execution of the Audit
// before it could conclude that none is left. Used by proposeReportSQL,
// commitReportSQL and purgeClaimedAuditSQL.
const uncollectedExecutionSQL = `EXISTS (
           SELECT 1 FROM audit_executions AS execution
            WHERE execution.audit_id = audit.audit_id
              AND execution.state IN ` + uncollectedExecutionStatesSQL + `)`

// listItemAttemptsSQL lists the execution attempts of items $3 in owner $1's
// Audit $2, each with its execution's role, terminal outcome, Run provenance
// and Run-deleted flag, ordered by item and attempt. $4 is one more than the
// projection bound so the caller can detect overflow.
// Used by QueryStore.ListItemAttempts.
var listItemAttemptsSQL = `
SELECT member.execution_item_id, member.execution_id, member.item_id,
       member.item_attempt, execution.role, member.state,
       member.collection_disposition, member.result_ref, member.result_digest,
       execution.terminal_outcome, execution.run_id, execution.run_provenance,
       execution.run_deleted_at IS NOT NULL AS run_deleted,
       member.created_at, member.collected_at
  FROM audit_execution_items AS member
  JOIN audit_executions AS execution USING (execution_id, audit_id)
  JOIN audits AS audit USING (audit_id)
 WHERE audit.owner_id = $1 AND audit.audit_id = $2
   AND member.item_id = ANY($3::text[])
 ORDER BY member.item_id, member.item_attempt, member.execution_item_id
 LIMIT $4`

// listCoverageSQL pages the coverage of round $2's items in Audit $1 together
// with each item's ordinal and task, keyset-paginated by item ordinal after $3
// and limited to $4 rows. Used by QueryStore.ListCoverage.
var listCoverageSQL = `
SELECT item.audit_id, item.round_id, item.item_id, item.ordinal,
       item.item_key, item.subject_key, item.coverage_status,
       item.coverage_requested, item.coverage_completed, item.coverage_gaps,
       item.coverage_rationale, item.coverage_result_ref, item.coverage_result_digest,
       item.coverage_updated_at, item.task_ref, item.task_digest
  FROM audit_items AS item
 WHERE item.audit_id = $1 AND item.round_id = $2
   AND ` + acceptedItemSQL("item") + `
   AND item.ordinal > $3
 ORDER BY item.ordinal, item.item_id LIMIT $4`
