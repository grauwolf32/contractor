package artifacts

import (
	"bytes"
	"context"
	"fmt"
	"slices"

	"github.com/grauwolf32/contractor/internal/artifactpolicy"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// purgeBatchSize bounds the binding revisions one purge batch removes and the
// candidate versions or blobs one purge statement locks or deletes
// (LIM-DEL-2). A Run or Audit purge stays one transaction whose total work the
// Run revision limit (LIM-ART-6) bounds its work; ProjectScope and Audit
// namespace purges commit one batch per transaction.
var purgeBatchSize = persistencepostgres.CleanupBatchRows

// PostgresPurger performs the exceptional physical cleanup used by lifecycle
// deletion. It only accepts an existing transaction so ArtifactStore cleanup
// and removal of the owning resource cannot commit independently.
type PostgresPurger struct {
	tx pgx.Tx
}

// PurgeAuditNamespace removes one Server-reserved Audit namespace without
// touching owner-authored Project bindings. Each call removes one revision
// batch and reports whether the namespace is empty. The owning Audit stays
// deleting across committed batches and is removed last.
func (p *PostgresPurger) PurgeAuditNamespace(
	ctx context.Context, projectID, namespace string,
) (bool, error) {
	scope, err := ProjectScope(projectID)
	if err != nil {
		return false, err
	}
	if !artifactpolicy.IsAuditManagedProjectNamespace(namespace) || validateComponent(namespace) != nil {
		return false, ErrInvalidName
	}
	ctx, cancel := persistencepostgres.WithCleanupBudget(ctx)
	defer cancel()
	if err := p.lockScope(ctx, scope); err != nil {
		return false, err
	}
	if _, err := p.tx.Exec(
		ctx, `SELECT set_config('contractor.lifecycle_purge', 'audit', true)`,
	); err != nil {
		return false, fmt.Errorf("enable Audit lifecycle purge: %w", err)
	}
	batch, err := p.purgeRevisionBatch(ctx, scope, namespace, revisionKey{namespace: namespace})
	if err != nil {
		return false, fmt.Errorf("purge Audit Artifact namespace: %w", err)
	}
	if err := p.collectUnreferencedVersions(ctx, batch.versionIDs); err != nil {
		return false, err
	}
	return len(batch.versionIDs) < purgeBatchSize, nil
}

func NewPostgresPurger(tx pgx.Tx) (*PostgresPurger, error) {
	if tx == nil {
		return nil, fmt.Errorf("PostgreSQL transaction is required")
	}
	return &PostgresPurger{tx: tx}, nil
}

// PurgeRun removes every RunScope binding and every pin owned by runID. Exact
// ProjectScope publications remain available, while their lineage edge back
// to the deleted Run is removed; the caller deletes the Run's publication
// receipts, which name the Run revisions, before this purge (runstore owns
// them). Versions and blobs are collected only when no surviving binding
// revision references them. Run deletion is one transaction (DEL-R22); the
// Run revision limit (LIM-ART-6) bounds its work, and no statement handles
// more than one purge batch (LIM-DEL-2).
func (p *PostgresPurger) PurgeRun(ctx context.Context, runID string) error {
	scope, err := RunScope(runID)
	if err != nil {
		return err
	}
	ctx, cancel := persistencepostgres.WithCleanupBudget(ctx)
	defer cancel()
	if err := p.lockScope(ctx, scope); err != nil {
		return err
	}
	if _, err := p.tx.Exec(
		ctx, `SELECT set_config('contractor.lifecycle_purge', 'run', true)`,
	); err != nil {
		return fmt.Errorf("enable Run lifecycle purge: %w", err)
	}
	versionIDs, err := p.purgeAllRevisions(ctx, scope, "")
	if err != nil {
		return fmt.Errorf("purge WorkflowRun Artifacts: %w", err)
	}
	if _, err := p.tx.Exec(ctx, `
DELETE FROM artifact_scopes
WHERE scope_kind = $1 AND scope_id = $2`, scope.kind, scope.id); err != nil {
		return fmt.Errorf("delete WorkflowRun Artifact scope: %w", err)
	}
	return p.collectUnreferencedVersions(ctx, versionIDs)
}

// PurgeProject removes the ProjectScope of a deleting Project in bounded
// steps, after every Project Run and Audit is gone; Run deletion removed the
// Runs' output publication receipts with them. Each call deletes at most one
// purge batch of binding revisions (LIM-DEL-2), with the lineage edges into
// or out of them and the bindings whose current revision they include, and
// collects the versions and blobs the batch leaves unreferenced. The call
// that finds no revision left removes the scope itself and reports true.
// Project deletion commits every call in its own transaction (DEL-R44); no
// write can enter the scope of a deleting Project without Runs and Audits, so
// a committed batch stays purged. UserScope data and physical content still
// referenced by another scope remain intact.
func (p *PostgresPurger) PurgeProject(ctx context.Context, projectID string) (bool, error) {
	scope, err := ProjectScope(projectID)
	if err != nil {
		return false, err
	}
	ctx, cancel := persistencepostgres.WithCleanupBudget(ctx)
	defer cancel()
	if err := p.lockScope(ctx, scope); err != nil {
		return false, err
	}
	if _, err := p.tx.Exec(
		ctx, `SELECT set_config('contractor.lifecycle_purge', 'project', true)`,
	); err != nil {
		return false, fmt.Errorf("enable Project lifecycle purge: %w", err)
	}
	batch, err := p.purgeRevisionBatch(ctx, scope, "", revisionKey{})
	if err != nil {
		return false, fmt.Errorf("purge Project Artifact batch: %w", err)
	}
	if len(batch.versionIDs) > 0 {
		return false, p.collectUnreferencedVersions(ctx, batch.versionIDs)
	}
	if _, err := p.tx.Exec(ctx, `
DELETE FROM artifact_scopes
WHERE scope_kind = $1 AND scope_id = $2`, scope.kind, scope.id); err != nil {
		return false, fmt.Errorf("delete Project Artifact scope: %w", err)
	}
	return true, nil
}

// Lifecycle callers hold resource authority before entering here. Lock order:
// owning scope, candidate versions (byte order of their IDs), candidate blobs
// (binary hash).
// Reference checks MUST be separate statements after the locks: READ COMMITTED
// then sees references removed/inserted by transactions we waited for. Foreign
// key locks coordinate retaining forks and blob upserts coordinate new writes.
// Cross-resource transactions can still deadlock; the owning database-only
// transaction may replay a definite 40P01/40001, never just this cleanup suffix.
func (p *PostgresPurger) lockScope(ctx context.Context, scope Scope) error {
	var isolation string
	if err := p.tx.QueryRow(ctx, `SHOW transaction_isolation`).Scan(&isolation); err != nil {
		return fmt.Errorf("inspect Artifact purge isolation: %w", err)
	}
	if isolation != "read committed" && isolation != "read uncommitted" {
		return fmt.Errorf("Artifact purge requires READ COMMITTED isolation")
	}
	if err := persistencepostgres.ApplyTransactionBudget(ctx, p.tx); err != nil {
		return err
	}
	_, err := p.tx.Exec(ctx, `SELECT 1 FROM artifact_scopes
WHERE scope_kind = $1 AND scope_id = $2 FOR UPDATE`, scope.kind, scope.id)
	return err
}

// revisionKey is the primary key of a binding revision within its scope. The
// zero key, or a key with only a namespace, sorts before every stored key of
// that range, because stored names and revisions are never empty.
type revisionKey struct {
	namespace, name, revision string
}

// revisionBatch is one purged batch: the version IDs of its revisions in key
// order and the key of its last revision.
type revisionBatch struct {
	versionIDs []string
	last       revisionKey
}

// purgeAllRevisions deletes every binding revision of scope, or of one
// namespace in it, batch by batch, and returns the versions they referenced
// for the caller to collect.
func (p *PostgresPurger) purgeAllRevisions(
	ctx context.Context, scope Scope, namespace string,
) ([]string, error) {
	versionIDs := make([]string, 0)
	after := revisionKey{namespace: namespace}
	for {
		batch, err := p.purgeRevisionBatch(ctx, scope, namespace, after)
		if err != nil {
			return nil, err
		}
		versionIDs = append(versionIDs, batch.versionIDs...)
		if len(batch.versionIDs) < purgeBatchSize {
			return versionIDs, nil
		}
		after = batch.last
	}
}

// purgeRevisionBatch deletes the next batch of at most purgeBatchSize binding
// revisions of scope after the key after, restricted to namespace when it is
// not empty: first the lineage edges into or out of them, because those
// foreign keys are immediate, then the bindings whose current revision is in
// the batch, then the revisions. An empty batch means nothing is left.
func (p *PostgresPurger) purgeRevisionBatch(
	ctx context.Context, scope Scope, namespace string, after revisionKey,
) (revisionBatch, error) {
	var rows pgx.Rows
	var err error
	if namespace == "" {
		rows, err = p.tx.Query(ctx, purgeScopeRevisionBatchSQL,
			scope.kind, scope.id, after.namespace, after.name, after.revision, purgeBatchSize)
	} else {
		after.namespace = namespace
		rows, err = p.tx.Query(ctx, purgeNamespaceRevisionBatchSQL,
			scope.kind, scope.id, namespace, after.name, after.revision, purgeBatchSize)
	}
	if err != nil {
		return revisionBatch{}, fmt.Errorf("list lifecycle purge Artifact revisions: %w", err)
	}
	var batch revisionBatch
	for rows.Next() {
		var versionID string
		if err := rows.Scan(
			&batch.last.namespace, &batch.last.name, &batch.last.revision, &versionID,
		); err != nil {
			rows.Close()
			return revisionBatch{}, fmt.Errorf("scan lifecycle purge Artifact revision: %w", err)
		}
		batch.versionIDs = append(batch.versionIDs, versionID)
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return revisionBatch{}, fmt.Errorf("iterate lifecycle purge Artifact revisions: %w", err)
	}
	if len(batch.versionIDs) == 0 {
		return batch, nil
	}
	arguments := []any{
		scope.kind, scope.id, after.namespace, after.name, after.revision,
		batch.last.namespace, batch.last.name, batch.last.revision,
	}
	for _, statement := range []struct{ sql, what string }{
		{purgeTargetLineageSQL, "Artifact lineage into"},
		{purgeSourceLineageSQL, "Artifact lineage out of"},
		{purgeBindingSQL, "Artifact bindings of"},
	} {
		if _, err := p.tx.Exec(ctx, statement.sql, arguments...); err != nil {
			return revisionBatch{}, fmt.Errorf("delete %s the purge batch: %w", statement.what, err)
		}
	}
	tag, err := p.tx.Exec(ctx, purgeRevisionSQL, arguments...)
	if err != nil {
		return revisionBatch{}, fmt.Errorf("delete Artifact revisions of the purge batch: %w", err)
	}
	// The scope lock keeps writers out, so the key range holds exactly the
	// listed revisions; any other count would leave versions uncollected.
	if tag.RowsAffected() != int64(len(batch.versionIDs)) {
		return revisionBatch{}, fmt.Errorf(
			"Artifact purge batch deleted %d revisions, listed %d",
			tag.RowsAffected(), len(batch.versionIDs),
		)
	}
	return batch, nil
}

// collectUnreferencedVersions locks every candidate version and then every
// candidate blob in one global order, at most purgeBatchSize per statement,
// before it deletes the ones nothing references any more.
func (p *PostgresPurger) collectUnreferencedVersions(
	ctx context.Context,
	versionIDs []string,
) error {
	if len(versionIDs) == 0 {
		return nil
	}
	// Distinct IDs in byte order, the order purgeLockVersionsSQL locks in, so
	// the lock order holds across chunk boundaries.
	versionIDs = slices.Clone(versionIDs)
	slices.Sort(versionIDs)
	versionIDs = slices.Compact(versionIDs)
	// Lock even still-referenced candidates: two purgers may delete different
	// versions of the same blob, or different references to the same version.
	candidateDigests := make([][]byte, 0, len(versionIDs))
	for chunk := range slices.Chunk(versionIDs, purgeBatchSize) {
		if _, err := p.tx.Exec(ctx, purgeLockVersionsSQL, chunk); err != nil {
			return fmt.Errorf("lock Artifact purge versions: %w", err)
		}
		digests, err := p.queryDigests(ctx, `SELECT DISTINCT blob_sha256 FROM artifact_versions
WHERE version_id = ANY($1::text[])`, chunk)
		if err != nil {
			return fmt.Errorf("list Artifact purge blobs: %w", err)
		}
		candidateDigests = append(candidateDigests, digests...)
	}
	for chunk := range slices.Chunk(sortedDigests(candidateDigests), purgeBatchSize) {
		if _, err := p.tx.Exec(ctx, `SELECT sha256 FROM artifact_blobs
WHERE sha256 = ANY($1::bytea[]) ORDER BY sha256 FOR UPDATE`, chunk); err != nil {
			return fmt.Errorf("lock Artifact purge blobs: %w", err)
		}
	}
	blobDigests := make([][]byte, 0, len(candidateDigests))
	for chunk := range slices.Chunk(versionIDs, purgeBatchSize) {
		digests, err := p.queryDigests(ctx, purgeUnreferencedVersionsSQL, chunk)
		if err != nil {
			return fmt.Errorf("collect unreferenced Artifact versions: %w", err)
		}
		blobDigests = append(blobDigests, digests...)
	}
	for chunk := range slices.Chunk(sortedDigests(blobDigests), purgeBatchSize) {
		if err := p.collectUnreferencedBlobs(ctx, chunk); err != nil {
			return err
		}
	}
	return nil
}

func (p *PostgresPurger) collectUnreferencedBlobs(ctx context.Context, digests [][]byte) error {
	deleted, err := p.tx.Query(ctx, purgeUnreferencedBlobsSQL, digests)
	if err != nil {
		return fmt.Errorf("collect unreferenced Artifact blobs: %w", err)
	}
	defer deleted.Close()
	for deleted.Next() {
		var kind BlobBackend
		var key *string
		if err := deleted.Scan(&kind, &key); err != nil {
			return err
		}
		if kind == BlobFilesystem && key != nil {
			if err := deletePhysicalAfterCommit(ctx, p.tx, BlobObject{Backend: kind, Key: *key}); err != nil {
				return err
			}
		}
	}
	return deleted.Err()
}

func (p *PostgresPurger) queryDigests(ctx context.Context, sql string, versionIDs []string) ([][]byte, error) {
	rows, err := p.tx.Query(ctx, sql, versionIDs)
	if err != nil {
		return nil, err
	}
	return pgx.CollectRows(rows, pgx.RowTo[[]byte])
}

// sortedDigests returns the distinct digests in PostgreSQL bytea order.
func sortedDigests(digests [][]byte) [][]byte {
	sorted := slices.Clone(digests)
	slices.SortFunc(sorted, bytes.Compare)
	return slices.CompactFunc(sorted, bytes.Equal)
}
