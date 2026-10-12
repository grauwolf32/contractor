// Package hookstore is the Audit store seam of the entry points other
// contexts use: finding intake records its decisions as Audit events, Run
// deletion invalidates what Audits derived from a deleted child Run, Project
// deletion requests the deletion of the Project's Audits, and the credential
// lifecycle finds the Audits whose dispatch hold keeps a credential in use.
// Each entry point is its own type, so a context holds only its own.
package hookstore

import (
	"github.com/grauwolf32/contractor/internal/auditstore/internal/pgstore"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// FindingEvents records the Audit events of finding intake decisions made
// outside a collection.
type FindingEvents = pgstore.FindingEvents

// RunDeletion is what Run deletion reads and changes of the Audits that
// bound the Run.
type RunDeletion = pgstore.RunDeletion

// RunDeletionTx is a RunDeletion bound to the caller's transaction, with the
// Audit locks that must last until the Run is deleted (PER-R18).
type RunDeletionTx = pgstore.RunDeletionTx

// ProjectDeletion requests the deletion of a deleting Project's Audits.
type ProjectDeletion = pgstore.ProjectDeletion

// CredentialHolds lists the Audits whose dispatch hold keeps a credential in
// use.
type CredentialHolds = pgstore.CredentialHolds

func NewFindingEvents(db persistencepostgres.DBTX) *FindingEvents {
	return pgstore.NewFindingEvents(db)
}

func NewRunDeletion(db persistencepostgres.DBTX) *RunDeletion {
	return pgstore.NewRunDeletion(db)
}

func NewRunDeletionTx(tx pgx.Tx) *RunDeletionTx {
	return pgstore.NewRunDeletionTx(tx)
}

func NewProjectDeletion(db persistencepostgres.DBTX) *ProjectDeletion {
	return pgstore.NewProjectDeletion(db)
}

func NewCredentialHolds(db persistencepostgres.DBTX) *CredentialHolds {
	return pgstore.NewCredentialHolds(db)
}
