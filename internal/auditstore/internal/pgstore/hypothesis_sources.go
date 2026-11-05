package pgstore

import (
	"context"
	"maps"
	"slices"

	"github.com/grauwolf32/contractor/internal/artifacts"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/jackc/pgx/v5"
)

// Accepted exact grounds use the existing Audit link/byte-accounting owner.
// This runs in the hypothesis/Task receipt transaction, so staged forks alone
// never become publishable materials. Existing revisions are charged once.
func persistHypothesisSources(ctx context.Context, tx pgx.Tx, audit auditstore.Audit, writes []auditstore.HypothesisWrite, storage *artifactpostgres.Storage) error {
	materials := map[string]contracts.MaterialDescriptor{}
	for _, write := range writes {
		for _, m := range write.Document.Sources {
			if m.ScopeKind != "audit" || m.ScopeID != audit.AuditID || m.Validate() != nil {
				return auditstore.ErrInvalid
			}
			materials[contracts.MaterialKey(m)] = m
		}
	}
	if len(materials) == 0 {
		return nil
	}
	if len(materials) > auditstore.MaxArtifactLinksPerCall {
		return auditstore.ErrInvalid
	}
	service := artifacts.NewService(storage.TxRepository(tx))
	scope, _ := artifacts.AuditScope(audit.AuditID)
	owned, _ := service.Scoped(scope)
	links := []auditstore.ArtifactLink{}
	for _, key := range slices.Sorted(maps.Keys(materials)) {
		m := materials[key]
		metadata, err := owned.Metadata(ctx, m.Ref)
		if err != nil {
			return err
		}
		if !metadata.Frozen || m.Exact().Verify(metadata.Exact()) != nil {
			return auditstore.ErrInvalid
		}
		links = append(links, auditstore.ArtifactLink{Scope: artifacts.ScopeAudit, LogicalKey: "hypothesis-source/" + contracts.MaterialHandle(m), Artifact: m.Exact(), SourceProvenance: []byte(`{"kind":"hypothesis-source"}`)})
	}
	if _, err := validateArtifactLinks(links); err != nil {
		return err
	}
	encoded, _ := prepareRetainedLinks(auditstore.CollectParams{Retained: links})
	var valid bool
	var bytes int64
	if err := tx.QueryRow(ctx, readCollectionLinkStateSQL, audit.AuditID, encoded).Scan(&valid, &bytes); err != nil {
		return err
	}
	if !valid || bytes > audit.Limits.MaxEvidenceBytes-audit.RetainedEvidenceBytes {
		return auditstore.ErrInvalid
	}
	_, err := tx.Exec(ctx, insertHypothesisSourcesSQL, audit.AuditID, encoded, bytes)
	return err
}
