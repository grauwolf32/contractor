package postgres

import (
	"context"
	"fmt"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/helpers/randomid"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// runRevisionLimitConstraint names the database rejection of LIM-ART-6 raised
// by contractor_count_run_artifact_revisions.
const runRevisionLimitConstraint = "artifact_scopes_run_revision_limit"

// isRunRevisionLimit reports whether err is the database rejection of a
// revision beyond a RunScope's limit: a stored bound the classifier reports,
// named by the revision limit's constraint.
func isRunRevisionLimit(err error) bool {
	return persistencepostgres.Classify(err) == persistencepostgres.ClassLimitExceeded &&
		persistencepostgres.ConstraintName(err) == runRevisionLimitConstraint
}

// exactHoldConstraint names the database rejection of a strict exact-reference
// hold on a revision that does not exist (DEL-R63), raised by
// contractor_artifact_hold_apply or strict lineage source validation. It means
// the exact revision a writer read is gone (ART-R124).
const exactHoldConstraint = "artifact_exact_hold"

// IsExactRevisionMissing reports whether err rejects a reference to an exact
// revision that no longer exists. A strict holder's writer that is not an
// ArtifactStore operation reports it as its own not-found outcome
// (FAIL-DEL-7).
func IsExactRevisionMissing(err error) bool {
	if persistencepostgres.Classify(err) != persistencepostgres.ClassReferenceViolation {
		return false
	}
	constraint := persistencepostgres.ConstraintName(err)
	return constraint == exactHoldConstraint
}

// storeError classifies the failure of a statement that creates scopes,
// bindings or revisions. The definite rejections every such statement can
// meet map to their sentinels: a Run beyond its revision limit
// (ErrRunRevisionLimit), a deleting Project scope (ErrScopeDeleting), a scope
// whose owner does not exist (ErrInvalidScope), an exact source revision that
// no longer exists (ErrArtifactNotFound) and, when conflict is not nil,
// a binding another writer created first (conflict). Anything else keeps its
// cause.
func storeError(what string, err error, conflict *artifacts.ConflictError) error {
	if isRunRevisionLimit(err) {
		return fmt.Errorf("%s: %w", what, artifacts.ErrRunRevisionLimit)
	}
	if IsExactRevisionMissing(err) {
		return fmt.Errorf("%s: %w", what, artifacts.ErrArtifactNotFound)
	}
	switch persistencepostgres.Classify(err) {
	case persistencepostgres.ClassDeleting:
		return fmt.Errorf("%s: %w", what, artifacts.ErrScopeDeleting)
	case persistencepostgres.ClassReferenceViolation:
		return fmt.Errorf("%s: %w", what, artifacts.ErrInvalidScope)
	case persistencepostgres.ClassDuplicate:
		if conflict != nil {
			return conflict
		}
	}
	return fmt.Errorf("%s: %w", what, err)
}

// Repository uses only the supplied DBTX. Passing a pgx.Tx makes every
// operation transaction-scoped without this repository starting a nested
// transaction. Its Repository methods trust their arguments, which Service
// validated.
type Repository struct {
	db    persistencepostgres.DBTX
	newID func(string) (string, error)
}

func NewRepository(db persistencepostgres.DBTX) *Repository {
	return &Repository{db: db, newID: randomid.New}
}

var _ artifacts.Repository = (*Repository)(nil)

// TxRepository is a Repository bound to the caller's
// transaction, for writes that must commit together with the version they
// describe.
type TxRepository struct {
	*Repository
	tx pgx.Tx
}

// NewTxRepository binds a Repository to the caller's
// transaction tx.
func NewTxRepository(tx pgx.Tx) *TxRepository {
	return &TxRepository{Repository: NewRepository(tx), tx: tx}
}

// ensureScope creates the row of scope unless it exists. It is a statement of
// its own on purpose: under READ COMMITTED a concurrent INSERT ... DO NOTHING
// that waited for the winning transaction is visible to the binding statement
// that follows, while a CTE of that statement would keep the pre-wait snapshot
// and could falsely report a conflict.
func (r *Repository) ensureScope(ctx context.Context, scope artifacts.Scope) error {
	if _, err := r.db.Exec(ctx, `
INSERT INTO artifact_scopes (scope_kind, scope_id)
VALUES ($1, $2)
ON CONFLICT DO NOTHING`, scope.Kind(), scope.ID()); err != nil {
		return storeError("create artifact scope", err, nil)
	}
	return nil
}

// PreparePayload prepares bytes and verifies an existing physical generation
// before constructing a transaction-bound repository for publication.
func (r *Repository) PreparePayload(ctx context.Context, payload artifacts.Payload) (artifacts.Payload, error) {
	return PreparePayload(ctx, r.db, payload)
}
