package auditimport

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/auditstore"
)

type retainedStandardLoader func() (retainedStandardIndex, error)

type retainedStandard struct {
	pinned    auditstandards.PinnedPackage
	entries   map[string]auditstandards.Entry
	mappings  map[string]auditstandards.Mapping
	contracts map[auditstandards.EvidenceContractRef]auditstandards.EvidenceContract
}

func indexRetainedStandard(pinned auditstandards.PinnedPackage, pkg auditstandards.Package) retainedStandard {
	result := retainedStandard{
		pinned:    pinned,
		entries:   make(map[string]auditstandards.Entry, len(pkg.Document.Entries)),
		mappings:  make(map[string]auditstandards.Mapping, len(pkg.Document.Mappings)),
		contracts: make(map[auditstandards.EvidenceContractRef]auditstandards.EvidenceContract, len(pkg.Document.EvidenceContracts)),
	}
	for _, entry := range pkg.Document.Entries {
		result.entries[entry.ID] = entry
	}
	for _, mapping := range pkg.Document.Mappings {
		result.mappings[mapping.Key] = mapping
	}
	for _, contract := range pkg.Document.EvidenceContracts {
		result.contracts[contract.Reference()] = contract
	}
	return result
}

type retainedStandardIndex map[string]retainedStandard

func (i *Importer) loadPinnedStandards(
	ctx context.Context, snapshot auditstore.ReconcileSnapshot,
) (retainedStandardIndex, error) {
	var baseline struct {
		Schema    string                         `json:"schema"`
		Standards []auditstandards.PinnedPackage `json:"standards"`
	}
	if json.Unmarshal(snapshot.Audit.BaselineSnapshot, &baseline) != nil ||
		baseline.Schema != "contractor.audit.baseline.v1" || baseline.Standards == nil {
		return nil, fmt.Errorf("%w: Audit standard baseline is invalid", ErrPermanent)
	}
	result := make(retainedStandardIndex, len(baseline.Standards))
	for _, pinned := range baseline.Standards {
		key := retainedStandardKey(pinned.Reference.Scheme, pinned.Reference.Version)
		if _, duplicate := result[key]; duplicate || auditstandards.ValidatePinnedPackage(pinned) != nil {
			return nil, fmt.Errorf("%w: Audit standard baseline is invalid", ErrPermanent)
		}
		payload, err := i.artifacts.ReadProjectExact(
			ctx, snapshot.Audit.ProjectID,
			auditstore.ExactArtifact{
				Ref: pinned.Retained.Artifact, Digest: pinned.Retained.Digest,
				MediaType: pinned.Retained.MediaType, SizeBytes: pinned.Retained.SizeBytes,
			},
		)
		if err != nil {
			return nil, fmt.Errorf("read exact retained Audit standard: %w", err)
		}
		pkg, err := auditstandards.ValidateRetainedPayload(payload, pinned)
		if err != nil {
			return nil, fmt.Errorf("%w: retained Audit standard is invalid", ErrPermanent)
		}
		result[key] = indexRetainedStandard(pinned, *pkg)
	}
	return result, nil
}

func validateMappedProposalStandards(
	task auditdomain.ItemTask,
	document auditdomain.FindingProposal,
	loadStandards retainedStandardLoader,
) error {
	if task.Standard == nil && len(document.StandardRefs) == 0 {
		return nil
	}
	standards, err := loadStandards()
	if err != nil {
		return err
	}
	if err := validateProposalStandardRefs(document, standards); err != nil {
		return err
	}
	if task.Standard == nil {
		return nil
	}
	standard, exists := standards[retainedStandardKey(task.Standard.Scheme, task.Standard.Version)]
	if !exists || !standardTaskMatchesPackage(task, standard) {
		return errors.New("task standard mapping does not match its retained package")
	}
	references := make(map[string]struct{}, len(document.StandardRefs))
	for _, reference := range document.StandardRefs {
		if reference.Scheme == task.Standard.Scheme && reference.Version == task.Standard.Version {
			references[reference.RequirementID] = struct{}{}
		}
	}
	for _, entryID := range task.Standard.EntryIDs {
		if _, exists := references[entryID]; !exists {
			return errors.New("proposal omits its assigned standard entry")
		}
	}
	return nil
}

func validateProposalStandardRefs(
	document auditdomain.FindingProposal, standards retainedStandardIndex,
) error {
	seen := make(map[string]struct{}, len(document.StandardRefs))
	for _, reference := range document.StandardRefs {
		key := retainedStandardKey(reference.Scheme, reference.Version)
		standard, exists := standards[key]
		if !exists {
			return errors.New("proposal names an unpinned standard")
		}
		identity := key + "\x00" + reference.RequirementID
		if _, duplicate := seen[identity]; duplicate {
			return errors.New("proposal repeats a standard reference")
		}
		seen[identity] = struct{}{}
		if _, exists := standard.entries[reference.RequirementID]; !exists {
			return errors.New("proposal names an unknown standard entry")
		}
	}
	return nil
}

func standardTaskMatchesPackage(task auditdomain.ItemTask, standard retainedStandard) bool {
	if task.Standard == nil || task.Checklist == nil ||
		task.SourceContentDigest != standard.pinned.Retained.Digest ||
		!sameRef(task.SourceRef, standard.pinned.Retained.Artifact) {
		return false
	}
	mapping, exists := standard.mappings[task.Standard.MappingKey]
	if !exists || mapping.Key != task.ItemKey || mapping.WorkflowRole != task.WorkflowRole ||
		!standardTaskStatementMatches(task.Checklist.Statement, mapping, standard) ||
		task.Checklist.Version != standard.pinned.Reference.Version ||
		len(task.Checklist.AllowedMethods) != 1 || task.Checklist.AllowedMethods[0] != mapping.Method ||
		!equalStrings(mapping.EntryIDs, task.Standard.EntryIDs) ||
		mapping.EvidenceContract.ID != task.Standard.EvidenceContract.ID ||
		mapping.EvidenceContract.Version != task.Standard.EvidenceContract.Version {
		return false
	}
	contract, exists := standard.contracts[mapping.EvidenceContract]
	if !exists {
		return false
	}
	selected := task.Standard.EvidenceContract
	expectedApplicability, expectedReview := "always", "automatic"
	for _, entryID := range mapping.EntryIDs {
		if entry, exists := standard.entries[entryID]; exists && entry.Applicability.Mode == "human-review" {
			expectedApplicability, expectedReview = "human-review", "manual"
		}
	}
	if contract.HumanReview == "required" {
		expectedReview = "manual"
	}
	requiredEvidence := []string{}
	if contract.MinimumEvidence > 0 {
		requiredEvidence = contract.EvidenceKinds
	}
	return task.Checklist.Applicability == expectedApplicability &&
		task.Checklist.ReviewPolicy == expectedReview &&
		equalStrings(task.Checklist.RequiredEvidence, requiredEvidence) &&
		equalStrings(contract.Assessments, selected.Assessments) &&
		equalStrings(contract.EvidenceKinds, selected.EvidenceKinds) &&
		contract.MinimumEvidence == selected.MinimumEvidence &&
		contract.MaximumEvidence == selected.MaximumEvidence &&
		contract.HumanReview == selected.HumanReview &&
		contract.RationaleRequired == selected.RationaleRequired
}

func standardTaskStatementMatches(
	statement string, mapping auditstandards.Mapping, standard retainedStandard,
) bool {
	if statement == mapping.Objective {
		return true
	}
	// Exact one-entry selections deliberately carry the authoritative standard
	// statement instead of the broader Contractor-authored mapping objective.
	// Accept only that other exact, retained-package value; model-authored text
	// cannot satisfy this boundary.
	if len(mapping.EntryIDs) != 1 {
		return false
	}
	entry, exists := standard.entries[mapping.EntryIDs[0]]
	return exists && statement == entry.Statement
}

func retainedStandardKey(scheme, version string) string { return scheme + "\x00" + version }
