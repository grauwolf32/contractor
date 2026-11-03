package app

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/grauwolf32/contractor/internal/agentskills"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/jackc/pgx/v5/pgxpool"
)

type catalogServices struct {
	artifacts *artifacts.Service
	findings  *findingintake.Service
}

// configureCatalogs seeds the bundles the startup configuration load
// discovered and validated; it never walks the operator root itself.
func configureCatalogs(
	ctx context.Context,
	pool *pgxpool.Pool,
	configurationManager *workflowconfig.Manager,
	bundles configload.Bundles,
	ownerID string,
	logger *slog.Logger,
	storage *artifactpostgres.Storage,
) (catalogServices, error) {
	artifactService := artifacts.NewService(storage.Repository(pool))
	findingService, err := findingintake.New(pool, storage)
	if err != nil {
		return catalogServices{}, fmt.Errorf("configure finding intake: %w", err)
	}
	skillCatalog, err := agentskills.NewCatalog(artifactService)
	if err != nil {
		return catalogServices{}, fmt.Errorf("configure SkillCatalog: %w", err)
	}
	skillSeedOutcomes, err := skillCatalog.Initialize(ctx, ownerID, bundles.Skills)
	if err != nil {
		return catalogServices{}, fmt.Errorf("initialize bundled skills: %w", err)
	}
	for _, outcome := range skillSeedOutcomes {
		logger.Info(
			"bundled skill initialization",
			"skill", outcome.Name,
			"outcome", outcome.Status,
			"bundled_digest", outcome.BundledDigest,
			"current_digest", outcome.CurrentDigest,
		)
	}
	standardCatalog, err := auditstandards.NewCatalog(artifactService)
	if err != nil {
		return catalogServices{}, fmt.Errorf("configure Audit standard catalog: %w", err)
	}
	standardSeedOutcomes, err := standardCatalog.Initialize(
		ctx, ownerID, bundles.Standards,
	)
	if err != nil {
		return catalogServices{}, fmt.Errorf("initialize bundled Audit standards: %w", err)
	}
	for _, outcome := range standardSeedOutcomes {
		logger.Info(
			"bundled Audit standard initialization",
			"scheme", outcome.Reference.Scheme,
			"version", outcome.Reference.Version,
			"outcome", outcome.Status,
			"digest", outcome.Digest,
		)
	}
	return catalogServices{artifacts: artifactService, findings: findingService}, nil
}
