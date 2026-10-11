package artifacts

import (
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/randomid"
	"github.com/jackc/pgx/v5"
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

var _ Repository = (*PostgresRepository)(nil)
var _ QueryRepository = (*PostgresRepository)(nil)
