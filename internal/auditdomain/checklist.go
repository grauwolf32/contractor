package auditdomain

import (
	"encoding/json"
	"sort"
)

// BuildChecklistInventory imports a complete checklist atomically. It accepts
// strict JSON or YAML and performs no model or network call.
func BuildChecklistInventory(source []byte, mediaType string, options InventoryOptions) (Inventory, error) {
	root, err := parseJSONOrYAML(source, mediaType)
	if err != nil {
		return Inventory{}, err
	}
	normalized, err := canonicalJSON(root)
	if err != nil {
		return Inventory{}, invalid(CodeInventoryInvalid, "checklist")
	}
	var document ChecklistDocument
	if _, err := decodeStrictJSON(normalized, &document); err != nil {
		return Inventory{}, invalid(CodeInventoryInvalid, "checklist")
	}
	entries, err := normalizeChecklist(document)
	if err != nil {
		return Inventory{}, err
	}

	basisSubjects := make([]map[string]any, 0, len(entries))
	subjects := make([]inventorySubject, 0, len(entries))
	for _, entry := range entries {
		task := ChecklistTask{
			Version: entry.Version, Statement: entry.Statement, Applicability: entry.Applicability,
			AllowedMethods:   copyStrings(entry.AllowedMethods),
			RequiredEvidence: copyStrings(entry.RequiredEvidence), ReviewPolicy: entry.ReviewPolicy,
		}
		basisSubjects = append(basisSubjects, map[string]any{
			"item_key": entry.Key, "version": entry.Version, "statement": entry.Statement,
			"applicability": entry.Applicability, "allowed_methods": entry.AllowedMethods,
			"required_evidence": entry.RequiredEvidence, "review_policy": entry.ReviewPolicy,
		})
		approval := ApprovalRequirement("")
		if entry.ReviewPolicy == "manual" {
			approval = ApprovalHumanReview
		}
		subjects = append(subjects, inventorySubject{
			itemKey: entry.Key, kind: "checklist", subjectKey: entry.Key,
			approval: approval, checklist: &task, requested: copyStrings(entry.RequiredEvidence),
		})
	}
	basis := inventoryBasis{Schema: InventoryBasisSchema, Kind: "checklist", Subjects: basisSubjects, Gaps: []string{}}
	return finishInventory(source, mediaType, basis, subjects, options)
}

func normalizeChecklist(document ChecklistDocument) ([]ChecklistEntry, error) {
	if document.Schema != ChecklistSchema {
		return nil, invalid(CodeSchemaUnsupported, "schema")
	}
	if document.Items == nil || len(document.Items) > MaximumItems {
		return nil, invalid(CodeLimitExceeded, "items")
	}
	result := make([]ChecklistEntry, len(document.Items))
	seen := make(map[string]struct{}, len(document.Items))
	for index, entry := range document.Items {
		if validateIdentifier(entry.Key, "items.key") != nil || validateText(entry.Version, "items.version", true) != nil ||
			validateText(entry.Statement, "items.statement", true) != nil || validateText(entry.Applicability, "items.applicability", true) != nil ||
			entry.ReviewPolicy != "automatic" && entry.ReviewPolicy != "manual" {
			return nil, invalid(CodeInventoryInvalid, "items")
		}
		if _, duplicate := seen[entry.Key]; duplicate {
			return nil, invalid(CodeInventoryInvalid, "items.key")
		}
		seen[entry.Key] = struct{}{}
		if entry.AllowedMethods == nil || entry.RequiredEvidence == nil {
			return nil, invalid(CodeInventoryInvalid, "items")
		}
		entry.AllowedMethods = copyStrings(entry.AllowedMethods)
		entry.RequiredEvidence = copyStrings(entry.RequiredEvidence)
		sort.Strings(entry.AllowedMethods)
		sort.Strings(entry.RequiredEvidence)
		if validateSortedStrings(entry.AllowedMethods, MaximumCoverageValues, "items.allowed_methods", true) != nil ||
			validateSortedStrings(entry.RequiredEvidence, MaximumEvidencePerItem, "items.required_evidence", true) != nil {
			return nil, invalid(CodeInventoryInvalid, "items")
		}
		result[index] = entry
	}
	sort.Slice(result, func(i, j int) bool { return result[i].Key < result[j].Key })
	return result, nil
}

// DecodeChecklist is exposed for profile input validation without inventory
// construction. It returns a detached, normalized checklist.
func DecodeChecklist(source []byte, mediaType string) (ChecklistDocument, error) {
	root, err := parseJSONOrYAML(source, mediaType)
	if err != nil {
		return ChecklistDocument{}, err
	}
	encoded, err := json.Marshal(root)
	if err != nil {
		return ChecklistDocument{}, invalid(CodeInventoryInvalid, "checklist")
	}
	var document ChecklistDocument
	if _, err := decodeStrictJSON(encoded, &document); err != nil {
		return ChecklistDocument{}, invalid(CodeInventoryInvalid, "checklist")
	}
	document.Items, err = normalizeChecklist(document)
	if err != nil {
		return ChecklistDocument{}, err
	}
	return document, nil
}
