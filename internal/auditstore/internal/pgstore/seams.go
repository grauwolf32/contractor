// Package pgstore is the PostgreSQL implementation of the Audit store. It
// keeps one store type per seam, so a caller holds only the commands and reads
// of the seam it works in: the claim-bound Controller commands (ClaimStore),
// the owner's lifecycle and start commands (OwnerStore), the review journal
// (ReviewStore), the reads (QueryStore), and the narrow entry points other
// contexts use (FindingEvents, RunDeletion, ProjectDeletion,
// CredentialHolds). A step whose row locks must last until the caller commits
// exists only on a seam's transaction-bound type (PER-R18). The package is
// internal: callers reach a seam through the auditstore subpackage that
// exports it.
package pgstore

import (
	"context"

	"github.com/grauwolf32/contractor/internal/auditstore"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// store is the database handle and the helpers that more than one seam
// shares. Every seam type embeds it.
type store struct{ db persistencepostgres.DBTX }

// ClaimStore holds the commands the Audit controller runs under a live
// controller claim: claims, transitions, Round acceptance, execution intents,
// Run binding, terminal observation, collection, report finalization,
// settlement and purge. Each command checks the claim epoch it is given.
type ClaimStore struct{ store }

func NewClaimStore(db persistencepostgres.DBTX) *ClaimStore {
	return &ClaimStore{store{db: db}}
}

// OwnerStore holds the owner's Audit lifecycle commands: draft creation, the
// start that materializes the first Round or begins preparation, pause and
// cancellation, resume and deletion, each replayed by its idempotency key.
type OwnerStore struct{ store }

func NewOwnerStore(db persistencepostgres.DBTX) *OwnerStore {
	return &OwnerStore{store{db: db}}
}

// OwnerTxStore is an OwnerStore bound to the caller's transaction, with the
// owner steps whose row locks must last until the caller commits.
type OwnerTxStore struct {
	*OwnerStore
	tx pgx.Tx
}

func NewOwnerTxStore(tx pgx.Tx) *OwnerTxStore {
	return &OwnerTxStore{OwnerStore: NewOwnerStore(tx), tx: tx}
}

// ReviewStore holds the review journal: finding triage requests and
// decisions, item action decisions, report candidate decisions and the
// review events they record.
type ReviewStore struct{ store }

func NewReviewStore(db persistencepostgres.DBTX) *ReviewStore {
	return &ReviewStore{store{db: db}}
}

// ReviewTxStore is a ReviewStore bound to the caller's transaction, with the
// review steps whose row locks must last until the caller commits.
type ReviewTxStore struct {
	*ReviewStore
	tx pgx.Tx
}

func NewReviewTxStore(tx pgx.Tx) *ReviewTxStore {
	return &ReviewTxStore{ReviewStore: NewReviewStore(tx), tx: tx}
}

// QueryStore holds the Audit reads: Audits, Rounds, items and their
// attempts, executions, coverage, artifact links, preparation and the report
// build's pages.
type QueryStore struct{ store }

func NewQueryStore(db persistencepostgres.DBTX) *QueryStore {
	return &QueryStore{store{db: db}}
}

// FindingEvents records the Audit events of finding intake decisions that
// finding intake makes outside a collection.
type FindingEvents struct{ store }

func NewFindingEvents(db persistencepostgres.DBTX) *FindingEvents {
	return &FindingEvents{store{db: db}}
}

// RunDeletion is what Run deletion reads and changes of the Audits whose
// executions bound the Run.
type RunDeletion struct{ store }

func NewRunDeletion(db persistencepostgres.DBTX) *RunDeletion {
	return &RunDeletion{store{db: db}}
}

// RunDeletionTx is a RunDeletion bound to the caller's transaction, with the
// Audit locks that must last until the Run is deleted.
type RunDeletionTx struct {
	*RunDeletion
	tx pgx.Tx
}

func NewRunDeletionTx(tx pgx.Tx) *RunDeletionTx {
	return &RunDeletionTx{RunDeletion: NewRunDeletion(tx), tx: tx}
}

// ProjectDeletion lets Project deletion request the deletion of the
// Project's Audits.
type ProjectDeletion struct{ store }

func NewProjectDeletion(db persistencepostgres.DBTX) *ProjectDeletion {
	return &ProjectDeletion{store{db: db}}
}

// CredentialHolds lets the credential lifecycle find the Audits whose
// dispatch hold keeps a credential in use.
type CredentialHolds struct{ store }

func NewCredentialHolds(db persistencepostgres.DBTX) *CredentialHolds {
	return &CredentialHolds{store{db: db}}
}

// Get reads ownerID's Audit.
func (s *QueryStore) Get(ctx context.Context, ownerID, auditID string) (auditstore.Audit, error) {
	return s.get(ctx, ownerID, auditID)
}

// GetRound reads one Round of an Audit.
func (s *QueryStore) GetRound(ctx context.Context, auditID, roundID string) (auditstore.Round, error) {
	return s.getRound(ctx, auditID, roundID)
}

// ListExecutionItems reads the members of one execution in batch order.
func (s *QueryStore) ListExecutionItems(ctx context.Context, executionID string) ([]auditstore.ExecutionItem, error) {
	return s.listExecutionItems(ctx, executionID)
}

// GetReportCandidate reads the frozen report candidate of an Audit's
// pending report review.
func (s *QueryStore) GetReportCandidate(ctx context.Context, auditID string) (auditstore.ReportCandidate, error) {
	return s.getReportCandidate(ctx, auditID)
}

// AppendReviewEvents reserves a contiguous sequence range and records all
// events atomically. Every event advances the Audit revision once, preserving
// the single-event contract; a rejected event rolls back the entire statement.
func (s *ReviewStore) AppendReviewEvents(ctx context.Context, events []auditstore.ReviewEventParams) error {
	return s.appendReviewEvents(ctx, events)
}
