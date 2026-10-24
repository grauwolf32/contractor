package artifacts

import (
	"context"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/artifactpolicy"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/jackc/pgx/v5"
)

// RetainSweepInput forks one exact owner input into a frozen, Server-owned
// binding in the same UserScope. No generic cross-UserScope fork is permitted.
// The sweep creation transaction owns authorization and the lifetime of this
// binding. Its bytes are shared with the source and all later Run input forks.
func (s *Service) RetainSweepInput(ctx context.Context, ownerID, sweepID string, source contracts.ArtifactRef) (ForkResult, error) {
	scope, err := UserScope(ownerID)
	if err != nil {
		return ForkResult{}, err
	}
	target, err := sweepInputRef(sweepID)
	if err != nil {
		return ForkResult{}, err
	}
	if source.ValidateExact() != nil || source.Namespace == artifactpolicy.RunSystemNamespace {
		return ForkResult{}, ErrInvalidName
	}
	return s.repository.Fork(ctx, Fork{
		Kind: LineageInputFork, Source: scope, SourceRef: source,
		Target: scope, TargetRef: target, Frozen: true, Replay: true,
	})
}

func sweepInputRef(sweepID string) (contracts.ArtifactRef, error) {
	ref := contracts.ArtifactRef{Namespace: artifactpolicy.RunSystemNamespace, Name: "scan-sweep." + sweepID + ".rulepack"}
	if sweepID == "" || ref.Validate() != nil {
		return contracts.ArtifactRef{}, ErrInvalidName
	}
	return ref, nil
}

// PurgeSweepInput removes exactly the one frozen snapshot the named sweep
// owns. The caller locks its terminal sweep first and uses READ COMMITTED.
// Child Run forks and the owner's original binding retain their own content.
func (p *PostgresPurger) PurgeSweepInput(ctx context.Context, ownerID, sweepID string) error {
	scope, err := UserScope(ownerID)
	if err != nil {
		return err
	}
	ref, err := sweepInputRef(sweepID)
	if err != nil {
		return err
	}
	if err := p.lockScope(ctx, scope); err != nil {
		return err
	}
	var versionID, revision string
	var frozen bool
	err = p.tx.QueryRow(ctx, `SELECT r.version_id, r.revision, b.frozen
FROM artifact_bindings b JOIN artifact_binding_revisions r
ON (r.scope_kind,r.scope_id,r.namespace,r.name,r.revision) = (b.scope_kind,b.scope_id,b.namespace,b.name,b.current_revision)
WHERE b.scope_kind = 'user' AND b.scope_id = $1 AND b.namespace = $2 AND b.name = $3`, ownerID, ref.Namespace, ref.Name).Scan(&versionID, &revision, &frozen)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("read sweep input for purge: %w", err)
	}
	if !frozen {
		return ErrArtifactIntegrity
	}
	if _, err := p.tx.Exec(ctx, `SELECT set_config('contractor.lifecycle_purge', 'scan-sweep', true)`); err != nil {
		return err
	}
	args := []any{scope.kind, scope.id, ref.Namespace, ref.Name, "", ref.Namespace, ref.Name, revision}
	for _, statement := range []string{purgeTargetLineageSQL, purgeSourceLineageSQL, purgeBindingSQL, purgeRevisionSQL} {
		if _, err := p.tx.Exec(ctx, statement, args...); err != nil {
			return fmt.Errorf("purge sweep input: %w", err)
		}
	}
	return collectUnreferencedVersions(ctx, p.tx, []string{versionID})
}
