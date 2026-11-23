package postgres

import (
	"context"
	"errors"

	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	"github.com/grauwolf32/contractor/internal/helpers/randomid"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// Storage owns the process dependencies shared by all Artifact repositories.
// Binding a database or transaction preserves backend selection, transfer
// admission and retained-history policy; operation contexts carry no settings.
type Storage struct {
	runtime *blob.Runtime
	history HistoryPolicy
}

func NewStorage(runtime *blob.Runtime, history HistoryPolicy) (*Storage, error) {
	if runtime == nil || runtime.Store == nil {
		return nil, errors.New("Artifact blob runtime is required")
	}
	if history.maxRevisions <= 0 || history.maxBytes <= 0 {
		return nil, ErrInvalidHistoryPolicy
	}
	return &Storage{runtime: runtime, history: history}, nil
}

// Repository binds the shared dependencies to db, without opening a transaction.
func (s *Storage) Repository(db persistencepostgres.DBTX) *Repository {
	return &Repository{Storage: s, db: db, newID: randomid.New}
}

// TxRepository binds the shared dependencies to the caller's transaction.
func (s *Storage) TxRepository(tx pgx.Tx) *TxRepository {
	return &TxRepository{Repository: s.Repository(tx), tx: tx}
}

// AcquireTransfer shares one process budget with the transport and trusted
// operations that must hold their slot across preparation and publication.
func (s *Storage) AcquireTransfer(ctx context.Context) (context.Context, func(), error) {
	return s.runtime.AcquireTransfer(ctx)
}
