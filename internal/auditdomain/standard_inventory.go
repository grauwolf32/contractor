package auditdomain

import (
	"sort"
	"strings"

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
	mappings, selection, err := selectedStandardMappings(
		document, entries, options.StandardSelection,
	)
	if err != nil {
		return Inventory{}, err
	}

	basisSubjects := make([]map[string]any, 0, len(mappings))
	subjects := make([]inventorySubject, 0, len(mappings))
	for _, mapping := range mappings {
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
		statement := mapping.Objective
		// An exact selected requirements denominator carries the normative
		// requirement statement into each task. Legacy all-mapping profiles
		// retain their existing scenario-objective behavior (Top 10).
		if selection != nil && len(entryIDs) == 1 {
			statement = entries[entryIDs[0]].Statement
		}
		checklist := &ChecklistTask{
			Version: document.Standard.Version, Statement: statement,
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
		Schema: InventoryBasisSchema, Kind: "standard-mappings", Selection: selection,
		Subjects: basisSubjects, Gaps: []string{},
	}
	return finishInventory(
		validated.Payload(), auditstandards.MediaType, basis, subjects, options,
	)
}

func standardContractKey(id, version string) string { return id + "\x00" + version }

func selectedStandardMappings(
	document auditstandards.Document,
	entries map[string]auditstandards.Entry,
	requested *StandardSelection,
) ([]auditstandards.Mapping, *StandardSelection, error) {
	if requested == nil {
		return append([]auditstandards.Mapping{}, document.Mappings...), nil, nil
	}
	selection := &StandardSelection{
		Scope:    requested.Scope,
		Levels:   append([]string{}, requested.Levels...),
		EntryIDs: append([]string{}, requested.EntryIDs...),
	}
	sort.Strings(selection.Levels)
	sort.Strings(selection.EntryIDs)
	if selection.Scope == "" || selection.Scope != strings.TrimSpace(selection.Scope) ||
		len([]byte(selection.Scope)) > 512 || len(selection.Levels) == 0 || len(selection.EntryIDs) == 0 ||
		len(selection.Levels) > 16 || len(selection.EntryIDs) > MaximumItems ||
		!strictlySortedNonEmpty(selection.Levels) || !strictlySortedNonEmpty(selection.EntryIDs) {
		return nil, nil, invalid(CodeInventoryInvalid, "standard_selection")
	}
	selected := make(map[string]struct{}, len(selection.EntryIDs))
	seenLevels := make(map[string]struct{}, len(selection.Levels))
	for _, entryID := range selection.EntryIDs {
		entry, exists := entries[entryID]
		if !exists || !containsString(selection.Levels, entry.Level) {
			return nil, nil, invalid(CodeInventoryInvalid, "standard_selection.entry_ids")
		}
		selected[entryID] = struct{}{}
		seenLevels[entry.Level] = struct{}{}
	}
	if len(seenLevels) != len(selection.Levels) {
		return nil, nil, invalid(CodeInventoryInvalid, "standard_selection.levels")
	}
	result := make([]auditstandards.Mapping, 0, len(selection.EntryIDs))
	seenEntries := make(map[string]struct{}, len(selection.EntryIDs))
	for _, mapping := range document.Mappings {
		matches := false
		for _, entryID := range mapping.EntryIDs {
			if _, exists := selected[entryID]; exists {
				matches = true
				break
			}
		}
		if !matches {
			continue
		}
		// One selected requirement is one denominator row. Multi-entry or
		// aliased mappings are rejected instead of silently collapsing it.
		if len(mapping.EntryIDs) != 1 || mapping.Key != mapping.EntryIDs[0] {
			return nil, nil, invalid(CodeInventoryInvalid, "standard_selection.mapping")
		}
		entryID := mapping.EntryIDs[0]
		if _, duplicate := seenEntries[entryID]; duplicate {
			return nil, nil, invalid(CodeInventoryInvalid, "standard_selection.mapping")
		}
		seenEntries[entryID] = struct{}{}
		result = append(result, mapping)
	}
	if len(result) != len(selection.EntryIDs) {
		return nil, nil, invalid(CodeInventoryInvalid, "standard_selection.mapping")
	}
	sort.Slice(result, func(i, j int) bool { return result[i].Key < result[j].Key })
	return result, selection, nil
}

func strictlySortedNonEmpty(values []string) bool {
	previous := ""
	for _, value := range values {
		if value == "" || value <= previous {
			return false
		}
		previous = value
	}
	return true
}

func containsString(values []string, candidate string) bool {
	index := sort.SearchStrings(values, candidate)
	return index < len(values) && values[index] == candidate
}
