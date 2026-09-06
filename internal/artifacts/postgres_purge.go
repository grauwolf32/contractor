package artifacts

import (
	"context"
	"fmt"
	"strings"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// PostgresPurger performs the exceptional physical cleanup used by lifecycle
// deletion. It only accepts an existing transaction so ArtifactStore cleanup
// and removal of the owning resource cannot commit independently.
type PostgresPurger struct {
	tx pgx.Tx
}

// PurgeAuditNamespace removes one Server-reserved Audit namespace without
// touching owner-authored Project bindings. The caller must delete the owning
// Audit domain rows in the same transaction.
func (p *PostgresPurger) PurgeAuditNamespace(
	ctx context.Context, projectID, namespace string,
) error {
	scope, err := ProjectScope(projectID)
	if err != nil {
		return err
	}
	if !strings.HasPrefix(namespace, "audit-") || validateComponent(namespace) != nil {
		return ErrInvalidName
	}
	ctx, cancel := persistencepostgres.WithCleanupBudget(ctx)
	defer cancel()
	if err := p.lockScope(ctx, scope); err != nil {
		return err
	}
	if _, err := p.tx.Exec(
		ctx, `SELECT set_config('contractor.lifecycle_purge', 'audit', true)`,
	); err != nil {
		return fmt.Errorf("enable Audit lifecycle purge: %w", err)
	}
	rows, err := p.tx.Query(ctx, `
SELECT DISTINCT version_id
FROM artifact_binding_revisions
WHERE scope_kind = $1 AND scope_id = $2 AND namespace = $3
ORDER BY version_id`, scope.kind, scope.id, namespace)
	if err != nil {
		return fmt.Errorf("list Audit lifecycle purge versions: %w", err)
	}
	versionIDs := make([]string, 0)
	for rows.Next() {
		var versionID string
		if err := rows.Scan(&versionID); err != nil {
			rows.Close()
			return fmt.Errorf("scan Audit lifecycle purge version: %w", err)
		}
		versionIDs = append(versionIDs, versionID)
	}
	if err := rows.Err(); err != nil {
		rows.Close()
		return fmt.Errorf("iterate Audit lifecycle purge versions: %w", err)
	}
	rows.Close()
	if _, err := p.tx.Exec(ctx, `
DELETE FROM artifact_pins
WHERE scope_kind = $1 AND scope_id = $2 AND namespace = $3`, scope.kind, scope.id, namespace); err != nil {
		return fmt.Errorf("delete Audit Artifact pins: %w", err)
	}
	if _, err := p.tx.Exec(ctx, `
DELETE FROM artifact_lineage
WHERE (source_scope_kind = $1 AND source_scope_id = $2 AND source_namespace = $3)
   OR (target_scope_kind = $1 AND target_scope_id = $2 AND target_namespace = $3)`,
		scope.kind, scope.id, namespace); err != nil {
		return fmt.Errorf("delete Audit Artifact lineage: %w", err)
	}
	if _, err := p.tx.Exec(ctx, `
DELETE FROM artifact_bindings
WHERE scope_kind = $1 AND scope_id = $2 AND namespace = $3`, scope.kind, scope.id, namespace); err != nil {
		return fmt.Errorf("delete Audit Artifact bindings: %w", err)
	}
	if _, err := p.tx.Exec(ctx, `
DELETE FROM artifact_binding_revisions
WHERE scope_kind = $1 AND scope_id = $2 AND namespace = $3`, scope.kind, scope.id, namespace); err != nil {
		return fmt.Errorf("delete Audit Artifact revisions: %w", err)
	}
	if err := p.collectUnreferencedVersions(ctx, versionIDs); err != nil {
		return err
	}
	return nil
}

func NewPostgresPurger(tx pgx.Tx) (*PostgresPurger, error) {
	if tx == nil {
		return nil, fmt.Errorf("PostgreSQL transaction is required")
	}
	return &PostgresPurger{tx: tx}, nil
}

// PurgeRun removes every RunScope binding and every pin owned by runID. Exact
// ProjectScope publications remain available, while their receipt and lineage
// edge back to the deleted Run are removed. Versions and blobs are collected
// only when no surviving binding revision references them.
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

	versionIDs, err := p.scopeVersionIDs(ctx, scope)
	if err != nil {
		return err
	}
	if _, err := p.tx.Exec(ctx, `
DELETE FROM workflow_run_output_publications
WHERE run_id = $1`, runID); err != nil {
		return fmt.Errorf("delete WorkflowRun output publications: %w", err)
	}
	if _, err := p.tx.Exec(ctx, `
DELETE FROM artifact_pins
WHERE run_id = $1`, runID); err != nil {
		return fmt.Errorf("delete WorkflowRun Artifact pins: %w", err)
	}
	if _, err := p.tx.Exec(ctx, `
DELETE FROM artifact_lineage
WHERE (source_scope_kind = $1 AND source_scope_id = $2)
   OR (target_scope_kind = $1 AND target_scope_id = $2)`, scope.kind, scope.id); err != nil {
		return fmt.Errorf("delete WorkflowRun Artifact lineage: %w", err)
	}
	if _, err := p.tx.Exec(ctx, `
DELETE FROM artifact_bindings
WHERE scope_kind = $1 AND scope_id = $2`, scope.kind, scope.id); err != nil {
		return fmt.Errorf("delete WorkflowRun Artifact bindings: %w", err)
	}
	if _, err := p.tx.Exec(ctx, `
DELETE FROM artifact_binding_revisions
WHERE scope_kind = $1 AND scope_id = $2`, scope.kind, scope.id); err != nil {
		return fmt.Errorf("delete WorkflowRun Artifact revisions: %w", err)
	}
	if _, err := p.tx.Exec(ctx, `
DELETE FROM artifact_scopes
WHERE scope_kind = $1 AND scope_id = $2`, scope.kind, scope.id); err != nil {
		return fmt.Errorf("delete WorkflowRun Artifact scope: %w", err)
	}
	if err := p.collectUnreferencedVersions(ctx, versionIDs); err != nil {
		return err
	}
	return nil
}

// PurgeProject removes the complete ProjectScope after every Project Run has
// already crossed the terminal/released deletion boundary. UserScope data and
// physical content still referenced by another scope remain intact.
func (p *PostgresPurger) PurgeProject(ctx context.Context, projectID string) error {
	scope, err := ProjectScope(projectID)
	if err != nil {
		return err
	}
	ctx, cancel := persistencepostgres.WithCleanupBudget(ctx)
	defer cancel()
	if err := p.lockScope(ctx, scope); err != nil {
		return err
	}
	if _, err := p.tx.Exec(
		ctx, `SELECT set_config('contractor.lifecycle_purge', 'project', true)`,
	); err != nil {
		return fmt.Errorf("enable Project lifecycle purge: %w", err)
	}

	versionIDs, err := p.scopeVersionIDs(ctx, scope)
	if err != nil {
		return err
	}
	if _, err := p.tx.Exec(ctx, `
DELETE FROM workflow_run_output_publications
WHERE project_id = $1`, projectID); err != nil {
		return fmt.Errorf("delete Project output publications: %w", err)
	}
	if _, err := p.tx.Exec(ctx, `
DELETE FROM artifact_pins
WHERE scope_kind = $1 AND scope_id = $2`, scope.kind, scope.id); err != nil {
		return fmt.Errorf("delete Project Artifact pins: %w", err)
	}
	if _, err := p.tx.Exec(ctx, `
DELETE FROM artifact_lineage
WHERE (source_scope_kind = $1 AND source_scope_id = $2)
   OR (target_scope_kind = $1 AND target_scope_id = $2)`, scope.kind, scope.id); err != nil {
		return fmt.Errorf("delete Project Artifact lineage: %w", err)
	}
	if _, err := p.tx.Exec(ctx, `
DELETE FROM artifact_bindings
WHERE scope_kind = $1 AND scope_id = $2`, scope.kind, scope.id); err != nil {
		return fmt.Errorf("delete Project Artifact bindings: %w", err)
	}
	if _, err := p.tx.Exec(ctx, `
DELETE FROM artifact_binding_revisions
WHERE scope_kind = $1 AND scope_id = $2`, scope.kind, scope.id); err != nil {
		return fmt.Errorf("delete Project Artifact revisions: %w", err)
	}
	if _, err := p.tx.Exec(ctx, `
DELETE FROM artifact_scopes
WHERE scope_kind = $1 AND scope_id = $2`, scope.kind, scope.id); err != nil {
		return fmt.Errorf("delete Project Artifact scope: %w", err)
	}
	if err := p.collectUnreferencedVersions(ctx, versionIDs); err != nil {
		return err
	}
	return nil
}

// Lifecycle callers hold resource authority before entering here. Lock order:
// owning scope, candidate versions (lexical ID), candidate blobs (binary hash).
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

func (p *PostgresPurger) scopeVersionIDs(ctx context.Context, scope Scope) ([]string, error) {
	rows, err := p.tx.Query(ctx, `
SELECT DISTINCT version_id
FROM artifact_binding_revisions
WHERE scope_kind = $1 AND scope_id = $2
ORDER BY version_id`, scope.kind, scope.id)
	if err != nil {
		return nil, fmt.Errorf("list lifecycle purge Artifact versions: %w", err)
	}
	defer rows.Close()
	result := make([]string, 0)
	for rows.Next() {
		var versionID string
		if err := rows.Scan(&versionID); err != nil {
			return nil, fmt.Errorf("scan lifecycle purge Artifact version: %w", err)
		}
		result = append(result, versionID)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate lifecycle purge Artifact versions: %w", err)
	}
	return result, nil
}

func (p *PostgresPurger) collectUnreferencedVersions(
	ctx context.Context,
	versionIDs []string,
) error {
	if len(versionIDs) == 0 {
		return nil
	}
	// Lock even still-referenced candidates: two purgers may delete different
	// versions of the same blob, or different references to the same version.
	if _, err := p.tx.Exec(ctx, `SELECT version_id FROM artifact_versions
WHERE version_id = ANY($1::text[]) ORDER BY version_id FOR UPDATE`, versionIDs); err != nil {
		return fmt.Errorf("lock Artifact purge versions: %w", err)
	}
	if _, err := p.tx.Exec(ctx, `SELECT sha256 FROM artifact_blobs
WHERE sha256 IN (SELECT blob_sha256 FROM artifact_versions WHERE version_id = ANY($1::text[]))
ORDER BY sha256 FOR UPDATE`, versionIDs); err != nil {
		return fmt.Errorf("lock Artifact purge blobs: %w", err)
	}
	rows, err := p.tx.Query(ctx, `
DELETE FROM artifact_versions AS version
WHERE version.version_id = ANY($1::text[])
  AND NOT EXISTS (
      SELECT 1
      FROM artifact_binding_revisions AS revision
      WHERE revision.version_id = version.version_id
  )
RETURNING version.blob_sha256`, versionIDs)
	if err != nil {
		return fmt.Errorf("collect unreferenced Artifact versions: %w", err)
	}
	blobDigests := make([][]byte, 0, len(versionIDs))
	for rows.Next() {
		var digest []byte
		if err := rows.Scan(&digest); err != nil {
			rows.Close()
			return fmt.Errorf("scan unreferenced Artifact blob digest: %w", err)
		}
		blobDigests = append(blobDigests, digest)
	}
	if err := rows.Err(); err != nil {
		rows.Close()
		return fmt.Errorf("iterate unreferenced Artifact blob digests: %w", err)
	}
	rows.Close()
	if len(blobDigests) == 0 {
		return nil
	}
	if _, err := p.tx.Exec(ctx, `
DELETE FROM artifact_blobs AS blob
WHERE blob.sha256 = ANY($1::bytea[])
  AND NOT EXISTS (
      SELECT 1
      FROM artifact_versions AS version
      WHERE version.blob_sha256 = blob.sha256
  )`, blobDigests); err != nil {
		return fmt.Errorf("collect unreferenced Artifact blobs: %w", err)
	}
	return nil
}
