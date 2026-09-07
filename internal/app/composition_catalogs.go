package app

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/grauwolf32/contractor/internal/agentskills"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/jackc/pgx/v5/pgxpool"
)

type catalogServices struct {
	artifacts *artifacts.Service
	findings  *findingintake.Service
}

func configureCatalogs(
	ctx context.Context,
	pool *pgxpool.Pool,
	configurationManager *workflowconfig.Manager,
	cfg Config,
	ownerID string,
	logger *slog.Logger,
) (catalogServices, error) {
	artifactService := artifacts.NewService(artifacts.NewPostgresRepository(pool))
	findingService, err := findingintake.New(pool)
	if err != nil {
		return catalogServices{}, fmt.Errorf("configure finding intake: %w", err)
	}
	skillCatalog, err := agentskills.NewCatalog(artifactService)
	if err != nil {
		return catalogServices{}, fmt.Errorf("configure SkillCatalog: %w", err)
	}
	skillSeedPlan, err := agentskills.DiscoverBundled(cfg.OperatorConfigRoot)
	if err != nil {
		return catalogServices{}, fmt.Errorf("discover bundled skills: %w", err)
	}
	skillSeedOutcomes, err := skillCatalog.Initialize(ctx, ownerID, skillSeedPlan)
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
	standardSeedPlan, err := auditstandards.DiscoverBundled(cfg.OperatorConfigRoot)
	if err != nil {
		return catalogServices{}, fmt.Errorf("discover bundled Audit standards: %w", err)
	}
	standardSeedOutcomes, err := standardCatalog.Initialize(
		ctx, ownerID, standardSeedPlan,
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
	if err := validateCatalogStandards(ctx, configurationManager, standardCatalog, ownerID); err != nil {
		return catalogServices{}, err
	}
	return catalogServices{artifacts: artifactService, findings: findingService}, nil
}

func validateCatalogStandards(ctx context.Context, configurationManager *workflowconfig.Manager, standardCatalog *auditstandards.Catalog, ownerID string) error {
	for _, profile := range configurationManager.AuditProfiles() {
		for _, standard := range profile.Standards {
			resolved, err := standardCatalog.Resolve(ctx, ownerID, auditstandards.Reference{
				Scheme: standard.Scheme, Version: standard.Version,
			})
			if err != nil {
				return fmt.Errorf(
					"resolve AuditProfile %s@%s standard %s@%s: %w",
					profile.Ref.Name, profile.Ref.Version, standard.Scheme, standard.Version, err,
				)
			}
			if profile.Inventory.Implementation == "standard-mappings@1" {
				var selection *auditdomain.StandardSelection
				if profile.Inventory.StandardSelection != nil {
					selection = &auditdomain.StandardSelection{
						Scope:    profile.Inventory.StandardSelection.Scope,
						Levels:   append([]string{}, profile.Inventory.StandardSelection.Levels...),
						EntryIDs: append([]string{}, profile.Inventory.StandardSelection.EntryIDs...),
					}
				}
				if _, err := auditdomain.BuildStandardMappingInventory(
					resolved.Package,
					auditdomain.InventoryOptions{
						Round: 1, WorkflowRole: profile.Inventory.ItemWorkflowRole,
						SourceInputName: "standard", SourceRef: resolved.Source.Artifact,
						ApprovalRequirement: auditdomain.ApprovalNone,
						StandardSelection:   selection,
					},
				); err != nil {
					return fmt.Errorf(
						"validate AuditProfile %s@%s standard selection: %w",
						profile.Ref.Name, profile.Ref.Version, err,
					)
				}
			}
		}
	}
	return nil
}
