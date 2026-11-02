package artifacts

import (
	"context"
	"errors"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// replaceFork uses the same history admission as a payload write, but inserts
// only a binding revision of the existing immutable version and its lineage.
func (r *PostgresRepository) replaceFork(ctx context.Context, fork Fork) (ForkResult, error) {
	var result ForkResult
	err := persistencepostgres.Within(ctx, r.db, pgx.TxOptions{}, func(tx pgx.Tx) error {
		result = ForkResult{}
		bound := NewPostgresRepository(tx)
		if err := bound.ensureScope(ctx, fork.Target); err != nil {
			return err
		}
		// Lock the exact source while ordinary history admission advances the target.
		var version, sourceRevision, media string
		var size int64
		err := tx.QueryRow(ctx, replaceForkSourceSQL,
			fork.Source.kind, fork.Source.id, fork.SourceRef.Namespace, fork.SourceRef.Name,
			fork.SourceRef.Revision, fork.Target.kind, fork.Target.id, fork.SourceFrozen,
		).Scan(&version, &sourceRevision, &media, &size)
		if errors.Is(err, pgx.ErrNoRows) {
			return ErrArtifactNotFound
		}
		if err != nil {
			return err
		}
		revision, err := r.newID("rev_")
		if err != nil {
			return err
		}
		policy := historyPolicy(ctx)
		write := writeStatement{size: size, policy: policy}
		write.reuse = func(ctx context.Context, db persistencepostgres.DBTX, removed, bytes int64) (writeRow, error) {
			var row writeRow
			err := db.QueryRow(ctx, replaceForkSQL,
				fork.Target.kind, fork.Target.id, fork.TargetRef.Namespace, fork.TargetRef.Name,
				*fork.ExpectedRevision, revision, version, size, policy.MaxRevisions(), policy.MaxBytes(), removed, bytes,
				fork.Source.kind, fork.Source.id, fork.SourceRef.Namespace, fork.SourceRef.Name, sourceRevision, fork.Kind,
			).Scan(&row.revision, &row.bindingCreatedAt, &row.revisionCreatedAt)
			row.mediaType, row.size = media, size
			return row, err
		}
		stored, err := write.exec(ctx, tx, 0, 0)
		if errors.Is(err, pgx.ErrNoRows) {
			stored, err = bound.advanceWithHistory(ctx, fork.Target, fork.TargetRef, *fork.ExpectedRevision, write)
		}
		if err != nil {
			return err
		}
		result = ForkResult{SourceRef: artifactExactRef(fork.SourceRef.Namespace, fork.SourceRef.Name, sourceRevision),
			TargetRef: artifactExactRef(fork.TargetRef.Namespace, fork.TargetRef.Name, stored.revision), MediaType: media, Size: size}
		return nil
	})
	return result, err
}
