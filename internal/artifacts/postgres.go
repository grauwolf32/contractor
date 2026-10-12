package artifacts

import (
	"errors"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/randomid"
	"github.com/jackc/pgx/v5/pgconn"
)

// runRevisionLimitConstraint names the database rejection of LIM-ART-6 raised
// by contractor_count_run_artifact_revisions.
const runRevisionLimitConstraint = "artifact_scopes_run_revision_limit"

// isRunRevisionLimit reports whether err is the database rejection of a
// revision beyond a RunScope's limit: a stored bound the classifier reports,
// named by the revision limit's constraint.
func isRunRevisionLimit(err error) bool {
	var pgError *pgconn.PgError
	return persistencepostgres.Classify(err) == persistencepostgres.ClassLimitExceeded &&
		errors.As(err, &pgError) && pgError.ConstraintName == runRevisionLimitConstraint
}

// PostgresRepository uses only the supplied DBTX. Passing a pgx.Tx makes every
// operation transaction-scoped without this repository starting a nested
// transaction.
type PostgresRepository struct {
	db    persistencepostgres.DBTX
	newID func(string) (string, error)
}

func NewPostgresRepository(db persistencepostgres.DBTX) *PostgresRepository {
	return &PostgresRepository{db: db, newID: randomid.New}
}

var _ Repository = (*PostgresRepository)(nil)
var _ QueryRepository = (*PostgresRepository)(nil)
