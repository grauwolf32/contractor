package artifacts

import (
	"context"
	"fmt"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/randomid"
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
func storeError(what string, err error, conflict *ConflictError) error {
	if isRunRevisionLimit(err) {
		return fmt.Errorf("%s: %w", what, ErrRunRevisionLimit)
	}
	if IsExactRevisionMissing(err) {
		return fmt.Errorf("%s: %w", what, ErrArtifactNotFound)
	}
	switch persistencepostgres.Classify(err) {
	case persistencepostgres.ClassDeleting:
		return fmt.Errorf("%s: %w", what, ErrScopeDeleting)
	case persistencepostgres.ClassReferenceViolation:
		return fmt.Errorf("%s: %w", what, ErrInvalidScope)
	case persistencepostgres.ClassDuplicate:
		if conflict != nil {
			return conflict
		}
	}
	return fmt.Errorf("%s: %w", what, err)
}

// PostgresRepository uses only the supplied DBTX. Passing a pgx.Tx makes every
// operation transaction-scoped without this repository starting a nested
// transaction. Its Repository methods trust their arguments, which Service
// validated.
type PostgresRepository struct {
	db    persistencepostgres.DBTX
	newID func(string) (string, error)
}

func NewPostgresRepository(db persistencepostgres.DBTX) *PostgresRepository {
	return &PostgresRepository{db: db, newID: randomid.New}
}

var _ Repository = (*PostgresRepository)(nil)

// PostgresTxRepository is a PostgresRepository bound to the caller's
// transaction, for writes that must commit together with the version they
// describe.
type PostgresTxRepository struct {
	*PostgresRepository
	tx pgx.Tx
}

// NewPostgresTxRepository binds a PostgresRepository to the caller's
// transaction tx.
func NewPostgresTxRepository(tx pgx.Tx) *PostgresTxRepository {
	return &PostgresTxRepository{PostgresRepository: NewPostgresRepository(tx), tx: tx}
}

// ensureScope creates the row of scope unless it exists. It is a statement of
// its own on purpose: under READ COMMITTED a concurrent INSERT ... DO NOTHING
// that waited for the winning transaction is visible to the binding statement
// that follows, while a CTE of that statement would keep the pre-wait snapshot
// and could falsely report a conflict.
func (r *PostgresRepository) ensureScope(ctx context.Context, scope Scope) error {
	if _, err := r.db.Exec(ctx, `
INSERT INTO artifact_scopes (scope_kind, scope_id)
VALUES ($1, $2)
ON CONFLICT DO NOTHING`, scope.kind, scope.id); err != nil {
		return storeError("create artifact scope", err, nil)
	}
	return nil
}
