package auditdomain

import (
	"sort"

	"github.com/grauwolf32/contractor/internal/auditstandards"
)

// BuildStandardMappingInventory deterministically expands mappings from one
// exact, already-retained standard package. Catalog packages may describe
// future execution modes; this first executable mapping contract intentionally
// accepts only rules the current Server can enforce without model authority.
func BuildStandardMappingInventory(
	source auditstandards.Package,
	options InventoryOptions,
) (Inventory, error) {
	validated, err := auditstandards.Validate(source.Payload(), source.Reference())
	if err != nil || validated.Digest != source.Digest {
		return Inventory{}, invalid(CodeInventoryInvalid, "standard_package")
	}
	document := validated.Document
	contracts := make(map[string]auditstandards.EvidenceContract, len(document.EvidenceContracts))
	for _, contract := range document.EvidenceContracts {
		contracts[standardContractKey(contract.ID, contract.Version)] = contract
	}
	entries := make(map[string]auditstandards.Entry, len(document.Entries))
	for _, entry := range document.Entries {
		entries[entry.ID] = entry
	}

	basisSubjects := make([]map[string]any, 0, len(document.Mappings))
	subjects := make([]inventorySubject, 0, len(document.Mappings))
	for _, mapping := range document.Mappings {
		if mapping.WorkflowRole != options.WorkflowRole {
			return Inventory{}, invalid(CodeInventoryInvalid, "standard_mapping.workflow_role")
		}
		contract, exists := contracts[standardContractKey(
			mapping.EvidenceContract.ID, mapping.EvidenceContract.Version,
		)]
		if !exists || contract.HumanReview == "on-inconclusive" {
			return Inventory{}, invalid(CodeInventoryInvalid, "standard_mapping.evidence_contract")
		}
		entryIDs := append([]string{}, mapping.EntryIDs...)
		sort.Strings(entryIDs)
		reviewPolicy, applicability := "automatic", "always"
		for _, entryID := range entryIDs {
			entry, ok := entries[entryID]
			if !ok || entry.Applicability.Mode == "profile-rule" {
				return Inventory{}, invalid(CodeInventoryInvalid, "standard_mapping.applicability")
			}
			if entry.Applicability.Mode == "human-review" {
				reviewPolicy, applicability = "manual", "human-review"
			}
		}
		if contract.HumanReview == "required" {
			reviewPolicy = "manual"
		}
		requested := []string{}
		if contract.MinimumEvidence > 0 {
			requested = append(requested, contract.EvidenceKinds...)
		}
		standard := &StandardMappingTask{
			Scheme: document.Standard.Scheme, Version: document.Standard.Version,
			MappingKey: mapping.Key, EntryIDs: entryIDs,
			EvidenceContract: StandardEvidenceContract{
				ID: mapping.EvidenceContract.ID, Version: mapping.EvidenceContract.Version,
				Assessments:     append([]string{}, contract.Assessments...),
				EvidenceKinds:   append([]string{}, contract.EvidenceKinds...),
				MinimumEvidence: contract.MinimumEvidence, MaximumEvidence: contract.MaximumEvidence,
				HumanReview: contract.HumanReview, RationaleRequired: contract.RationaleRequired,
			},
		}
		checklist := &ChecklistTask{
			Version: document.Standard.Version, Statement: mapping.Objective,
			Applicability: applicability, AllowedMethods: []string{mapping.Method},
			RequiredEvidence: append([]string{}, requested...), ReviewPolicy: reviewPolicy,
		}
		basisSubject := map[string]any{
			"item_key": mapping.Key, "subject_key": mapping.Key,
			"checklist": checklist, "standard": standard,
		}
		basisSubjects = append(basisSubjects, basisSubject)
		approval := ApprovalNone
		if reviewPolicy == "manual" {
			approval = ApprovalHumanReview
		}
		subjects = append(subjects, inventorySubject{
			itemKey: mapping.Key, kind: "standard-mapping", subjectKey: mapping.Key,
			approval: approval, checklist: checklist, standard: standard,
			requested: requested,
		})
	}
	basis := inventoryBasis{
		Schema: InventoryBasisSchema, Kind: "standard-mappings",
		Subjects: basisSubjects, Gaps: []string{},
	}
	return finishInventory(
		validated.Payload(), auditstandards.MediaType, basis, subjects, options,
	)
}

func standardContractKey(id, version string) string { return id + "\x00" + version }
