package app

import (
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func validateBundledCatalogStandards(operatorRoot string, snapshot *config.Snapshot) error {
	plan, err := auditstandards.DiscoverBundled(operatorRoot)
	if err != nil {
		return fmt.Errorf("discover bundled Audit standards: %w", err)
	}
	for _, profile := range snapshot.AuditProfiles() {
		for _, standard := range profile.Standards {
			ref := auditstandards.Reference{Scheme: standard.Scheme, Version: standard.Version}
			pkg, err := plan.ResolveBundled(ref)
			if errors.Is(err, auditstandards.ErrNotFound) {
				return fmt.Errorf("AuditProfile %s@%s standard %s@%s is not bundled; offline validation cannot verify database-only standards",
					profile.Ref.Name, profile.Ref.Version, ref.Scheme, ref.Version)
			}
			if err != nil {
				return fmt.Errorf("resolve AuditProfile %s@%s bundled standard %s@%s: %w",
					profile.Ref.Name, profile.Ref.Version, ref.Scheme, ref.Version, err)
			}
			revision := "offline-validation"
			if err := validateProfileStandardSelection(profile, pkg, contracts.ArtifactRef{
				Namespace: auditstandards.CatalogNamespace,
				Name:      auditstandards.ArtifactName(ref), Revision: &revision,
			}); err != nil {
				return err
			}
		}
	}
	return nil
}

func validateProfileStandardSelection(profile config.ResolvedAuditProfile, pkg auditstandards.Package, source contracts.ArtifactRef) error {
	if profile.Inventory.Implementation != "standard-mappings@1" {
		return nil
	}
	var selection *auditdomain.StandardSelection
	if profile.Inventory.StandardSelection != nil {
		selection = &auditdomain.StandardSelection{
			Scope:    profile.Inventory.StandardSelection.Scope,
			Levels:   append([]string{}, profile.Inventory.StandardSelection.Levels...),
			EntryIDs: append([]string{}, profile.Inventory.StandardSelection.EntryIDs...),
		}
	}
	if _, err := auditdomain.BuildStandardMappingInventory(pkg, auditdomain.InventoryOptions{
		Round: 1, WorkflowRole: profile.Inventory.ItemWorkflowRole,
		SourceInputName: "standard", SourceRef: source,
		ApprovalRequirement: auditdomain.ApprovalNone,
		StandardSelection:   selection,
	}); err != nil {
		return fmt.Errorf("validate AuditProfile %s@%s standard selection: %w",
			profile.Ref.Name, profile.Ref.Version, err)
	}
	return nil
}
