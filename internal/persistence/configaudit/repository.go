// Package configaudit persists non-authoritative, secret-free managed
// configuration publication audit metadata.
package configaudit

import (
	"context"
	"fmt"

	"github.com/grauwolf32/contractor/internal/config"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
)

type Repository struct {
	db persistencepostgres.DBTX
}

func New(db persistencepostgres.DBTX) *Repository { return &Repository{db: db} }

func (r *Repository) RecordConfigurationPublication(
	ctx context.Context, audit config.PublicationAudit,
) error {
	if r == nil || r.db == nil {
		return fmt.Errorf("configuration publication audit repository is not configured")
	}
	if _, err := config.ParseConfigurationKind(string(audit.Kind)); err != nil {
		return err
	}
	if audit.Kind != config.ConfigurationModelPolicies && audit.Kind != config.ConfigurationLLMGateways {
		return fmt.Errorf("configuration kind %q is not publishable", audit.Kind)
	}
	if _, err := config.ParseSelector(audit.Name + "@" + audit.Version); err != nil {
		return err
	}
	if audit.ActorID == "" || audit.PublishedAt.IsZero() {
		return fmt.Errorf("configuration publication audit actor and timestamp are required")
	}
	_, err := r.db.Exec(ctx, `
INSERT INTO configuration_publications (
    kind, name, version, digest, request_digest,
    idempotency_key_digest, actor_id, published_at
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
ON CONFLICT DO NOTHING`,
		audit.Kind, audit.Name, audit.Version, audit.Digest, audit.RequestDigest,
		audit.IdempotencyKeyDigest, audit.ActorID, audit.PublishedAt,
	)
	if err != nil {
		return fmt.Errorf("record configuration publication audit: %w", err)
	}
	return nil
}
