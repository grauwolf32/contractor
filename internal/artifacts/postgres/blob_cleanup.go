package postgres

import (
	"context"

	"github.com/grauwolf32/contractor/internal/artifacts/blob/backends/filesystem"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// CleanupFilesystemBlobs supplies the complete registry reference inventory to
// the offline physical cleanup. Apply still requires all writers to be stopped.
func CleanupFilesystemBlobs(ctx context.Context, db postgres.DBTX, path string, apply bool) (filesystem.CleanupReport, error) {
	return filesystem.Cleanup(ctx, path, apply, cleanupReferences{db})
}

type cleanupReferences struct{ db postgres.DBTX }

func (r cleanupReferences) Page(ctx context.Context, after string, limit int) ([]string, error) {
	rows, err := r.db.Query(ctx, `SELECT object_key FROM artifact_blobs WHERE backend='filesystem' AND object_key > $1 ORDER BY object_key LIMIT $2`, after, limit)
	if err != nil {
		return nil, err
	}
	return pgx.CollectRows(rows, pgx.RowTo[string])
}

func (r cleanupReferences) Referenced(ctx context.Context, keys []string) (map[string]bool, error) {
	rows, err := r.db.Query(ctx, `SELECT object_key FROM artifact_blobs WHERE object_key = ANY($1::text[])`, keys)
	if err != nil {
		return nil, err
	}
	found, err := pgx.CollectRows(rows, pgx.RowTo[string])
	if err != nil {
		return nil, err
	}
	result := make(map[string]bool, len(found))
	for _, key := range found {
		result[key] = true
	}
	return result, nil
}
