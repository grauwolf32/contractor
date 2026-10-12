// Package rundeletion permanently deletes a released terminal WorkflowRun
// together with the Audit, finding and Artifact state that refers to it, in
// one transaction. It composes steps owned by runstore, auditstore,
// findingintake and artifacts, each changing only its own tables, so none of
// those stores depends on another for a Run deletion.
package rundeletion

import (
	"context"
	"fmt"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditstore/hookstore"
	"github.com/grauwolf32/contractor/internal/findingintake"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
)

// Reason names the lifecycle gate that keeps a Run from being deleted.
type Reason string

const (
	RunNotTerminal           Reason = "run_not_terminal"
	AllocationReleasePending Reason = "allocation_release_pending"
	AuditCollectionPending   Reason = "audit_collection_pending"
)

// NotDeletableError identifies the lifecycle gate that rejected a hard
// deletion without exposing any owner information. It is a runstore
// conflict, so callers that only distinguish conflicts keep doing so.
type NotDeletableError struct {
	RunID  string
	Reason Reason
}

func (e *NotDeletableError) Error() string {
	return fmt.Sprintf("WorkflowRun %q is not deletable: %s", e.RunID, e.Reason)
}

func (e *NotDeletableError) Unwrap() error { return runstore.ErrConflict }

// Deleter deletes Runs on a pool, in its own retrying transaction, or inside
// a caller's pgx.Tx so Project deletion can compose it atomically.
type Deleter struct {
	db persistencepostgres.DBTX
}

func New(db persistencepostgres.DBTX) *Deleter {
	return &Deleter{db: db}
}

// DeleteReleasedTerminalRun permanently removes one owner-visible terminal
// Run and its Run-owned data.
func (d *Deleter) DeleteReleasedTerminalRun(ctx context.Context, ownerID, runID string) error {
	return persistencepostgres.Within(ctx, d.db, pgx.TxOptions{}, func(tx pgx.Tx) error {
		return deleteReleasedTerminalRun(ctx, tx, ownerID, runID)
	})
}

func deleteReleasedTerminalRun(ctx context.Context, tx pgx.Tx, ownerID, runID string) error {
	runs := runstore.NewPostgresTxStore(tx)
	gates, err := runs.LockRunForDeletion(ctx, ownerID, runID)
	if err != nil {
		return err
	}
	if !runstore.RunLifecycleTerminal.Includes(gates.State) {
		return &NotDeletableError{RunID: runID, Reason: RunNotTerminal}
	}
	if gates.AllocationReleasePending {
		return &NotDeletableError{RunID: runID, Reason: AllocationReleasePending}
	}
	// ImportIntoAudit holds the source Run FOR KEY SHARE before creating a
	// destination hold. Our Run lock makes this destination set stable. Take
	// Audit locks before execution, retention and Artifact locks, matching
	// import and Audit purge rather than adding an Audit lock at the end.
	audits := hookstore.NewRunDeletionTx(tx)
	auditIDs, err := audits.LockRunDeletionAudits(ctx, ownerID, runID)
	if err != nil {
		return err
	}
	if gates.PublicationMode == runstore.PublicationAuditManaged {
		if gates.AuditExecutionID == nil {
			return fmt.Errorf("delete WorkflowRun %q: invalid Audit authority", runID)
		}
		marked, err := audits.MarkExecutionRunDeleted(ctx, *gates.AuditExecutionID, runID)
		if err != nil {
			return err
		}
		if !marked {
			return &NotDeletableError{RunID: runID, Reason: AuditCollectionPending}
		}
	}
	// Finding receipt identity and safe provenance outlive an ordinary source
	// Run. Imported proposals already have destination-Audit bindings; every
	// other proposal becomes a durable discarded tombstone before Run-owned
	// pins and bindings are purged below.
	if err := findingintake.MarkSourceRunDeleted(ctx, tx, runID); err != nil {
		return fmt.Errorf("dispose WorkflowRun %q finding proposals: %w", runID, err)
	}
	// The Project output publication receipts name the Run's output
	// revisions, so they go before the purge; the published outputs stay.
	if err := runs.DeleteLockedRunOutputPublications(ctx, runID); err != nil {
		return err
	}
	purger, err := artifacts.NewPostgresPurger(tx)
	if err != nil {
		return fmt.Errorf("prepare WorkflowRun %q Artifact purge: %w", runID, err)
	}
	if err := purger.PurgeRun(ctx, runID); err != nil {
		return fmt.Errorf("purge WorkflowRun %q Artifacts: %w", runID, err)
	}
	if err := runs.DeleteLockedRun(ctx, ownerID, runID); err != nil {
		return err
	}
	// Only external Run availability changed: finding assessments and pending
	// review subjects keep their own revisions. The Audit revision invalidates
	// every enclosing read/cursor, including terminal Audits.
	// While finalizing, UpdatedAt is also the frozen report's generatedFrom.
	// Keep it stable so a retry after a failed revision CAS reuses the exact
	// immutable report bytes; Run availability is absent from that payload.
	if err := audits.InvalidateDeletedRunProjections(ctx, auditIDs); err != nil {
		return fmt.Errorf("invalidate deleted WorkflowRun %q Audit projections: %w", runID, err)
	}
	return nil
}

// RunDeletionBlocker reports the first durable lifecycle gate for an
// owner-visible Run without mutating it, or nil when the Run is deletable.
// The answer is advisory; deletion repeats every check under a row lock in
// one transaction.
func (d *Deleter) RunDeletionBlocker(ctx context.Context, ownerID, runID string) (*Reason, error) {
	blockers, err := d.blockers(ctx, ownerID, []string{runID})
	if err != nil {
		return nil, err
	}
	reason, ok := blockers[runID]
	if !ok {
		return nil, fmt.Errorf("inspect WorkflowRun %q deletion: %w", runID, runstore.ErrNotFound)
	}
	return reason, nil
}

// DeletableRuns reports, for at most one public page (pagelimit.MaxRows) of
// the owner's Runs, which ones RunDeletionBlocker would report as deletable.
// A Run that does not exist or belongs to another owner is reported as not
// deletable.
func (d *Deleter) DeletableRuns(ctx context.Context, ownerID string, runIDs []string) (map[string]bool, error) {
	blockers, err := d.blockers(ctx, ownerID, runIDs)
	if err != nil {
		return nil, err
	}
	result := make(map[string]bool, len(runIDs))
	for _, runID := range runIDs {
		reason, ok := blockers[runID]
		result[runID] = ok && reason == nil
	}
	return result, nil
}

// blockers maps each existing owner-visible Run to its first deletion gate,
// nil when none applies, in the order deletion checks them.
func (d *Deleter) blockers(ctx context.Context, ownerID string, runIDs []string) (map[string]*Reason, error) {
	gates, err := runstore.NewPostgresStore(d.db).ReadRunDeletionGates(ctx, ownerID, runIDs)
	if err != nil {
		return nil, err
	}
	result := make(map[string]*Reason, len(gates))
	// workflow_runs_audit_execution_unique keeps one Run per AuditExecution.
	runByExecution := make(map[string]string)
	for runID, gate := range gates {
		switch {
		case !runstore.RunLifecycleTerminal.Includes(gate.State):
			result[runID] = reasonOf(RunNotTerminal)
		case gate.AllocationReleasePending:
			result[runID] = reasonOf(AllocationReleasePending)
		case gate.PublicationMode != runstore.PublicationAuditManaged:
			result[runID] = nil
		case gate.AuditExecutionID == nil:
			result[runID] = reasonOf(AuditCollectionPending)
		default:
			runByExecution[*gate.AuditExecutionID] = runID
		}
	}
	collected, err := hookstore.NewRunDeletion(d.db).CollectedExecutionRuns(ctx, runByExecution)
	if err != nil {
		return nil, err
	}
	for executionID, runID := range runByExecution {
		if collected[executionID] {
			result[runID] = nil
		} else {
			result[runID] = reasonOf(AuditCollectionPending)
		}
	}
	return result, nil
}

func reasonOf(reason Reason) *Reason { return &reason }
