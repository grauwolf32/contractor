package auditservice

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditbaseline"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
)

var (
	errEmptyInitialInventory          = errors.New("prepared inventory contains no usable items")
	errInitialInventoryEvidenceBudget = errors.New("prepared inventory exceeds the remaining evidence budget")
)

func preparationInputKey(mapping config.AuditWorkflowInputMapping) string {
	return mapping.Role + "/" + mapping.Name
}

func acceptedPreparationInputs(ctx context.Context, store *auditstore.PostgresStore, auditID string, profile config.ResolvedAuditProfile) (map[string]auditstore.ExactArtifact, error) {
	result := map[string]auditstore.ExactArtifact{}
	if !profile.HasPreparation() {
		return result, nil
	}
	executions, err := store.ListPreparationExecutions(ctx, auditID)
	if err != nil {
		return nil, err
	}
	for _, execution := range executions {
		// Non-accepted attempts have no preparation_outputs by database contract.
		for _, output := range execution.PreparationOutputs {
			result[execution.WorkflowRole+"/"+output.LogicalName] = output.Retained
		}
	}
	for role, binding := range profile.Workflows {
		if binding.Kind != config.AuditWorkflowPrepare {
			continue
		}
		for name := range binding.Outputs {
			if _, accepted := result[role+"/"+name]; !accepted {
				return nil, fmt.Errorf("%w: preparation output %s/%s is not accepted", auditstore.ErrNotFound, role, name)
			}
		}
	}
	return result, nil
}

// PrepareInitialRound publishes deterministic, non-authoritative packages.
// AcceptInitialRound is the sole fenced transaction that makes them executable.
func (s *Service) PrepareInitialRound(ctx context.Context, claim auditstore.ControllerClaim, snapshot auditstore.ReconcileSnapshot) (auditstore.AcceptInitialRoundParams, *auditstore.StopReason, error) {
	audit := snapshot.Audit
	if claim.AuditID != audit.AuditID || audit.Phase != auditdomain.AuditPhaseInventory || audit.CurrentRoundID != nil || snapshot.Round != nil {
		return auditstore.AcceptInitialRoundParams{}, nil, inconsistentRound("initial inventory is outside its preparation boundary", nil)
	}
	params, err := s.prepareInitialInventory(ctx, claim, snapshot)
	if err == nil {
		return params, nil, nil
	}
	if errors.Is(err, errEmptyInitialInventory) {
		return auditstore.AcceptInitialRoundParams{}, &auditstore.StopReason{Code: "empty_inventory", Message: err.Error()}, nil
	}
	if errors.Is(err, errInitialInventoryEvidenceBudget) {
		return auditstore.AcceptInitialRoundParams{}, &auditstore.StopReason{Code: "evidence_budget_exhausted", Message: err.Error()}, nil
	}
	var validation *auditdomain.ValidationError
	var inconsistent *RoundPreparationError
	if errors.Is(err, ErrInvalid) || errors.Is(err, auditstore.ErrInvalid) || errors.As(err, &validation) || errors.As(err, &inconsistent) || errors.Is(err, artifacts.ErrArtifactIntegrity) || errors.Is(err, artifacts.ErrArtifactNotFound) || errors.Is(err, auditstore.ErrNotFound) {
		return auditstore.AcceptInitialRoundParams{}, &auditstore.StopReason{
			Code: "invalid_inventory", Message: "The prepared Audit inventory failed validation: " + err.Error(),
		}, nil
	}
	return auditstore.AcceptInitialRoundParams{}, nil, err
}

func (s *Service) prepareInitialInventory(ctx context.Context, claim auditstore.ControllerClaim, snapshot auditstore.ReconcileSnapshot) (auditstore.AcceptInitialRoundParams, error) {
	audit := snapshot.Audit
	profile, err := config.DecodeResolvedAuditProfileSnapshot(audit.ProfileSnapshot)
	if err != nil || !profile.HasPreparation() || profile.Ref.Name != audit.Profile.Name || profile.Ref.Version != audit.Profile.Version || profile.Ref.Digest != audit.Profile.Digest {
		return auditstore.AcceptInitialRoundParams{}, inconsistentRound("the pinned preparation profile is invalid", err)
	}
	baseline, err := DecodeBaseline(audit.BaselineSnapshot)
	if err != nil || baseline.Inventory != nil {
		return auditstore.AcceptInitialRoundParams{}, inconsistentRound("the original preparation baseline is invalid", err)
	}
	selection := DraftSelection{Inputs: baseline.Inputs, Scope: baseline.Scope, RuntimeLabels: baseline.RuntimeLabels}
	service := artifacts.NewService(artifacts.NewPostgresRepository(s.pool))
	original, err := readAndVerifyInputs(ctx, service, audit.ProjectID, profile, selection)
	if err != nil {
		return auditstore.AcceptInitialRoundParams{}, err
	}
	store := auditstore.NewPostgresStore(s.pool)
	prepared, err := acceptedPreparationInputs(ctx, store, audit.AuditID, profile)
	if err != nil {
		return auditstore.AcceptInitialRoundParams{}, err
	}
	project, err := service.Project(audit.ProjectID)
	if err != nil {
		return auditstore.AcceptInitialRoundParams{}, err
	}
	sources := map[string]auditstore.ExactArtifact{}
	resolve := func(name string, mapping *config.AuditWorkflowInputMapping) (artifacts.ReadResult, error) {
		if mapping == nil {
			return artifacts.ReadResult{}, nil
		}
		if mapping.Source == config.AuditInputFromAudit {
			read, exists := original[mapping.Name]
			if !exists {
				return artifacts.ReadResult{}, fmt.Errorf("%w: inventory %s is missing", ErrInvalid, name)
			}
			sources[name] = baseline.Inputs[mapping.Name]
			return read, nil
		}
		if mapping.Source != config.AuditInputFromPreparation {
			return artifacts.ReadResult{}, fmt.Errorf("%w: inventory %s source is invalid", ErrInvalid, name)
		}
		exact, exists := prepared[preparationInputKey(*mapping)]
		if !exists {
			return artifacts.ReadResult{}, fmt.Errorf("%w: inventory %s output is not accepted", ErrInvalid, name)
		}
		read, err := project.Read(ctx, exact.Ref)
		if err != nil {
			return artifacts.ReadResult{}, err
		}
		if !read.Ref.SameExact(exact.Ref) || read.Payload.MediaType != exact.MediaType || int64(len(read.Payload.Data)) != exact.SizeBytes || auditdomain.DigestBytes(read.Payload.Data) != exact.Digest {
			return artifacts.ReadResult{}, artifacts.ErrArtifactIntegrity
		}
		sources[name] = exact
		return read, nil
	}
	source, err := resolve("source", profile.Inventory.Source)
	if err != nil {
		return auditstore.AcceptInitialRoundParams{}, err
	}
	settings, err := resolve("settings", profile.Inventory.Settings)
	if err != nil {
		return auditstore.AcceptInitialRoundParams{}, err
	}
	var inventory auditdomain.Inventory
	if profile.Inventory.Implementation == "standard-mappings@1" {
		catalog, err := auditstandards.NewCatalog(service)
		if err != nil {
			return auditstore.AcceptInitialRoundParams{}, err
		}
		standards := make([]auditstandards.ResolvedPackage, len(baseline.Standards))
		for i, pinned := range baseline.Standards {
			standards[i], err = catalog.ResolvePinned(ctx, audit.ProjectID, pinned)
			if err != nil {
				return auditstore.AcceptInitialRoundParams{}, err
			}
		}
		inventory, err = buildInventory(profile, selection, original, standards)
	} else {
		approval := auditdomain.ApprovalNone
		if profile.Interaction.ActiveChecks == config.AuditActiveChecksApprovalRequired && workflowRoleSelectsClassifiedTool(profile, profile.Inventory.ItemWorkflowRole, true) {
			approval = auditdomain.ApprovalActiveCheck
		}
		inventory, err = buildInventoryFromSources(profile, selection, source, settings, approval)
	}
	if err != nil {
		return auditstore.AcceptInitialRoundParams{}, err
	}
	if len(inventory.Worklist.Items) == 0 {
		return auditstore.AcceptInitialRoundParams{}, errEmptyInitialInventory
	}
	if len(inventory.Worklist.Items) > roundItemCapacity(audit.Limits.MaxItemsPerRound, audit.Limits.MaxItemsTotal, 0) {
		return auditstore.AcceptInitialRoundParams{}, fmt.Errorf("%w: prepared inventory exceeds the Audit item limit", ErrInvalid)
	}
	if err := validateInventoryTaskExecution(profile, inventory); err != nil {
		return auditstore.AcceptInitialRoundParams{}, err
	}
	namespace := auditdomain.ArtifactNamespace(audit.AuditID)
	tasks, manifest, err := writeTaskPackages(ctx, project, namespace, profile, selection, inventory, prepared)
	if err != nil {
		return auditstore.AcceptInitialRoundParams{}, err
	}
	worklist, err := writeRoundPackage(ctx, project, namespace, 1, inventory, manifest)
	if err != nil {
		return auditstore.AcceptInitialRoundParams{}, err
	}
	derived := auditbaseline.DerivedInventory{Schema: auditbaseline.DerivedInventorySchema, Sources: sources,
		Inventory: BaselineInventory{SourceContentDigest: inventory.SourceContentDigest, CanonicalInventoryDigest: inventory.CanonicalInventoryDigest,
			StandardSelection: cloneAuditStandardSelection(profile.Inventory.StandardSelection), Gaps: append([]string{}, inventory.Gaps...), Worklist: worklist, ExecutionManifest: manifest}}
	encoded, err := json.Marshal(derived)
	if err != nil {
		return auditstore.AcceptInitialRoundParams{}, err
	}
	if len(encoded) > auditstore.MaxSnapshotBytes {
		return auditstore.AcceptInitialRoundParams{}, fmt.Errorf("%w: derived initial inventory snapshot exceeds its byte limit", ErrInvalid)
	}
	if int64(len(encoded)) > audit.Limits.MaxEvidenceBytes-audit.RetainedEvidenceBytes {
		return auditstore.AcceptInitialRoundParams{}, errInitialInventoryEvidenceBudget
	}
	metadata, err := writeImmutableArtifact(ctx, project, contracts.ArtifactRef{Namespace: namespace, Name: auditdomain.DeterministicID("initial-inventory", auditdomain.DigestBytes(encoded))}, artifacts.Payload{MediaType: "application/json", Data: encoded})
	if err != nil {
		return auditstore.AcceptInitialRoundParams{}, err
	}
	items, err := materializeInventoryItems(audit.AuditID, profile, inventory, tasks)
	if err != nil {
		return auditstore.AcceptInitialRoundParams{}, err
	}
	provenance, _ := json.Marshal(map[string]any{"schema": auditbaseline.DerivedInventorySchema, "sources": sources})
	return auditstore.AcceptInitialRoundParams{Claim: claim, ExpectedAuditRevision: audit.Revision,
		RoundID: auditdomain.DeterministicID("round", audit.AuditID, "1"), Manifest: worklist, Items: items,
		Inventory: auditstore.ArtifactLink{LogicalKey: auditstore.InitialInventoryLogicalKey, Artifact: metadata, SourceProvenance: provenance}}, nil
}
