package artifacts

import (
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/randomid"
)

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
