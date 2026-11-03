// Package hookstore is the Audit store seam of the entry points other
// contexts use: finding intake locks the Audit it decides for and its
// Project's finding correlation fence and records its decisions as Audit
// events, Run deletion invalidates what Audits derived from a deleted child
// Run, Project deletion requests the deletion of the Project's Audits, the
// credential lifecycle finds the Audits whose dispatch hold keeps a credential
// in use, and the Eval store matches the Audits it created to their creation
// requests. Each entry point is its own type, so a context holds only its own.
package hookstore

import (
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/auditstore/internal/pgstore"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// FindingEvents records the Audit events of finding intake decisions made
// outside a collection.
type FindingEvents = pgstore.FindingEvents

// FindingEventsTx is a FindingEvents bound to finding intake's transaction,
// with the Audit lock and the Project finding correlation fence it holds until
// it commits and the direct verification it commits under that lock (PER-R18).
type FindingEventsTx = pgstore.FindingEventsTx

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

// AuditCreations matches an Audit to the owner's create request that made it.
type AuditCreations = pgstore.AuditCreations

// ClaimedArtifacts publishes material for round preparation and import under
// an explicit live claim, without exposing controller commands.
type ClaimedArtifacts = pgstore.ClaimedArtifacts

func NewClaimedArtifacts(db persistencepostgres.DBTX, storage *artifactpostgres.Storage) *ClaimedArtifacts {
	return pgstore.NewClaimedArtifacts(db, storage)
}

func NewFindingEvents(db persistencepostgres.DBTX, storage *artifactpostgres.Storage) *FindingEvents {
	return pgstore.NewFindingEvents(db, storage)
}

func NewFindingEventsTx(tx pgx.Tx, storage *artifactpostgres.Storage) *FindingEventsTx {
	return pgstore.NewFindingEventsTx(tx, storage)
}

func NewRunDeletion(db persistencepostgres.DBTX, storage *artifactpostgres.Storage) *RunDeletion {
	return pgstore.NewRunDeletion(db, storage)
}

func NewRunDeletionTx(tx pgx.Tx, storage *artifactpostgres.Storage) *RunDeletionTx {
	return pgstore.NewRunDeletionTx(tx, storage)
}

func NewProjectDeletion(db persistencepostgres.DBTX, storage *artifactpostgres.Storage) *ProjectDeletion {
	return pgstore.NewProjectDeletion(db, storage)
}

func NewCredentialHolds(db persistencepostgres.DBTX, storage *artifactpostgres.Storage) *CredentialHolds {
	return pgstore.NewCredentialHolds(db, storage)
}

func NewAuditCreations(db persistencepostgres.DBTX, storage *artifactpostgres.Storage) *AuditCreations {
	return pgstore.NewAuditCreations(db, storage)
}
