package pgstore

// SQL statements for read.go.

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
