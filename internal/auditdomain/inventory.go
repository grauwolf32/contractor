package auditdomain

import (
	"bytes"
	"sort"
	"strings"

	"github.com/grauwolf32/contractor/internal/contracts"
)

type inventorySubject struct {
	itemKey    string
	kind       string
	subjectKey string
	approval   ApprovalRequirement
	checklist  *ChecklistTask
	standard   *StandardMappingTask
	operation  *OperationTask
	finding    *FindingTask
	requested  []string
	gaps       []string
}

func finishInventory(
	source []byte,
	sourceMediaType string,
	basis inventoryBasis,
	subjects []inventorySubject,
	options InventoryOptions,
) (Inventory, error) {
	if err := validateInventoryOptions(options, sourceMediaType); err != nil {
		return Inventory{}, invalid(CodeInventoryInvalid, "options")
	}
	if basis.Schema != InventoryBasisSchema || len(subjects) != len(basis.Subjects) {
		return Inventory{}, invalid(CodeInventoryInvalid, "items")
	}
	if len(subjects) > MaximumItems {
		return Inventory{}, invalid(CodeLimitExceeded, "items")
	}
	canonical, err := canonicalJSON(basis)
	if err != nil {
		return Inventory{}, invalid(CodeInventoryInvalid, "inventory")
	}
	canonicalDigest := digestBytes(canonical)
	sourceDigest := digestBytes(source)
	worklist := WorklistManifest{Schema: WorklistSchema, Round: options.Round, Items: make([]WorklistItem, 0, len(subjects))}
	execution := ExecutionManifest{Schema: ExecutionManifestSchema, Items: make([]ExecutionItem, 0, len(subjects))}
	coverage := CoverageEnvelope{Schema: CoverageSchema, Rows: make([]CoverageRow, 0, len(subjects))}
	tasks := make([]GeneratedTask, 0, len(subjects))

	seen := make(map[string]struct{}, len(subjects))
	var generatedBytes int64
	for ordinal, subject := range subjects {
		if _, duplicate := seen[subject.itemKey]; duplicate {
			return Inventory{}, invalid(CodeInventoryInvalid, "items.item_key")
		}
		seen[subject.itemKey] = struct{}{}
		document := ItemTask{
			Schema: TaskSchema, ItemKey: subject.itemKey, Kind: subject.kind,
			SubjectKey: subject.subjectKey, WorkflowRole: options.WorkflowRole,
			SourceContentDigest: sourceDigest, SourceMediaType: normalizedMediaType(sourceMediaType), SourceRef: copyArtifactRef(options.SourceRef),
			CanonicalInventoryDigest: canonicalDigest,
			Scope:                    copyStringMap(options.Scope), Checklist: subject.checklist,
			Standard: subject.standard, Operation: subject.operation, Finding: subject.finding,
		}
		documentBytes, encodeErr := EncodeItemTask(document)
		if encodeErr != nil {
			return Inventory{}, encodeErr
		}
		packageID := deterministicPackageID(documentBytes)
		packageBytes, validatedPackage, packageErr := BuildPackage(packageID, PackageKindTask, "", []PackageInput{{
			ID: "task-document", Path: "task.json", MediaType: "application/json", Data: documentBytes,
		}})
		if packageErr != nil {
			return Inventory{}, packageErr
		}
		generatedBytes += int64(len(packageBytes))
		if generatedBytes > MaximumGeneratedBytes {
			return Inventory{}, invalid(CodeLimitExceeded, "generated_packages")
		}
		approval := subject.approval
		// A trusted profile-level active-check gate is stronger than an
		// inventory-authored/manual gate. One exact owner decision authorizes
		// the resulting action, so untrusted inventory data can request more
		// review but can never downgrade the profile policy.
		if options.ApprovalRequirement == ApprovalActiveCheck || approval == "" {
			approval = options.ApprovalRequirement
		}
		item := WorklistItem{
			ItemKey: subject.itemKey, Ordinal: ordinal, Kind: subject.kind,
			SubjectKey: subject.subjectKey, WorkflowRole: options.WorkflowRole,
			TaskPackageID: packageID, ApprovalRequirement: approval,
		}
		worklist.Items = append(worklist.Items, item)
		execution.Items = append(execution.Items, ExecutionItem{
			ItemKey: subject.itemKey, Ordinal: ordinal, SubjectKey: subject.subjectKey,
			TaskPackageID: packageID, TaskPackageDigest: validatedPackage.Digest,
			Inputs: []ExactInput{{Name: options.SourceInputName, Ref: copyArtifactRef(options.SourceRef), Digest: sourceDigest}},
		})
		coverage.Rows = append(coverage.Rows, CoverageRow{
			ItemKey: subject.itemKey, SubjectKey: subject.subjectKey, Status: "not-tested",
			Requested: arrayStrings(subject.requested), Completed: []string{}, Gaps: arrayStrings(subject.gaps),
		})
		tasks = append(tasks, GeneratedTask{
			Item: item, Document: document, Package: packageBytes, PackageDigest: validatedPackage.Digest,
		})
	}
	if err := validateWorklist(worklist); err != nil {
		return Inventory{}, err
	}
	if err := validateExecutionManifest(execution); err != nil {
		return Inventory{}, err
	}
	if err := validateCoverage(coverage); err != nil {
		return Inventory{}, err
	}
	result := Inventory{
		SourceContentDigest: sourceDigest, CanonicalInventoryDigest: canonicalDigest,
		CanonicalInventory: canonical, Worklist: worklist, ExecutionManifest: execution,
		Coverage: coverage, Tasks: tasks, Gaps: copyStrings(basis.Gaps),
	}
	if err := ValidateInventory(result); err != nil {
		return Inventory{}, err
	}
	return result, nil
}

// ValidateInventory rechecks every cross-document identity and package before
// a caller persists or dispatches any part of the all-or-nothing result.
func ValidateInventory(value Inventory) error {
	if !validDigest(value.SourceContentDigest) || !validDigest(value.CanonicalInventoryDigest) ||
		digestBytes(value.CanonicalInventory) != value.CanonicalInventoryDigest {
		return invalid(CodeInventoryInvalid, "inventory.digest")
	}
	var basis inventoryBasis
	parsed, err := decodeStrictJSON(value.CanonicalInventory, &basis)
	if err != nil || basis.Schema != InventoryBasisSchema {
		return invalid(CodeInventoryInvalid, "inventory.canonical")
	}
	canonical, err := canonicalJSON(parsed)
	if err != nil || !bytes.Equal(canonical, value.CanonicalInventory) || !equalStringSlices(basis.Gaps, value.Gaps) ||
		validateSortedStrings(basis.Gaps, MaximumCoverageValues, "inventory.gaps", false) != nil ||
		basis.Kind != "checklist" && basis.Kind != "standard-mappings" &&
			basis.Kind != "openapi-operations" && basis.Kind != "finding-candidates" {
		return invalid(CodeInventoryInvalid, "inventory.canonical")
	}
	if basis.Selection != nil {
		selection := basis.Selection
		if basis.Kind != "standard-mappings" || selection.Scope == "" ||
			selection.Scope != strings.TrimSpace(selection.Scope) || len([]byte(selection.Scope)) > 512 ||
			len(selection.Levels) == 0 || len(selection.Levels) > 16 ||
			len(selection.EntryIDs) == 0 || len(selection.EntryIDs) > MaximumItems ||
			!strictlySortedNonEmpty(selection.Levels) || !strictlySortedNonEmpty(selection.EntryIDs) {
			return invalid(CodeInventoryInvalid, "inventory.standard_selection")
		}
	}
	if err := validateWorklist(value.Worklist); err != nil {
		return err
	}
	if err := validateExecutionManifest(value.ExecutionManifest); err != nil {
		return err
	}
	if err := validateCoverage(value.Coverage); err != nil {
		return err
	}
	count := len(value.Worklist.Items)
	if len(basis.Subjects) != count || len(value.ExecutionManifest.Items) != count || len(value.Coverage.Rows) != count || len(value.Tasks) != count {
		return invalid(CodeInventoryInvalid, "inventory.membership")
	}
	var generatedBytes int64
	for index := 0; index < count; index++ {
		workItem := value.Worklist.Items[index]
		executionItem := value.ExecutionManifest.Items[index]
		coverage := value.Coverage.Rows[index]
		generated := value.Tasks[index]
		if generated.Item != workItem || executionItem.Ordinal != workItem.Ordinal || executionItem.ItemKey != workItem.ItemKey ||
			executionItem.SubjectKey != workItem.SubjectKey || executionItem.TaskPackageID != workItem.TaskPackageID ||
			coverage.ItemKey != workItem.ItemKey || coverage.SubjectKey != workItem.SubjectKey || generated.Document.ItemKey != workItem.ItemKey ||
			generated.Document.SubjectKey != workItem.SubjectKey || generated.Document.WorkflowRole != workItem.WorkflowRole ||
			generated.Document.SourceContentDigest != value.SourceContentDigest || generated.Document.CanonicalInventoryDigest != value.CanonicalInventoryDigest {
			return invalid(CodeInventoryInvalid, "inventory.membership")
		}
		if coverage.Status != "not-tested" || len(coverage.Completed) != 0 {
			return invalid(CodeInventoryInvalid, "inventory.coverage")
		}
		if err := validateBasisItem(basis, value.CanonicalInventoryDigest, index, workItem, generated.Document, coverage); err != nil {
			return err
		}
		if len(executionItem.Inputs) != 1 || executionItem.Inputs[0].Digest != value.SourceContentDigest ||
			!sameArtifactRef(executionItem.Inputs[0].Ref, generated.Document.SourceRef) {
			return invalid(CodeInventoryInvalid, "inventory.inputs")
		}
		validated, err := ValidatePackage(generated.Package)
		if err != nil || validated.Manifest.Kind != PackageKindTask || validated.Manifest.PackageID != workItem.TaskPackageID ||
			validated.Digest != generated.PackageDigest || executionItem.TaskPackageDigest != generated.PackageDigest || len(validated.Members()) != 1 {
			return invalid(CodeInventoryInvalid, "inventory.task_package")
		}
		member, exists := validated.MemberByID("task-document")
		metadata := member.Metadata()
		if !exists || metadata.Path != "task.json" || metadata.MediaType != "application/json" {
			return invalid(CodeInventoryInvalid, "inventory.task_package")
		}
		decoded, err := DecodeItemTask(member.Data())
		if err != nil {
			return invalid(CodeInventoryInvalid, "inventory.task_package")
		}
		stored, err := EncodeItemTask(generated.Document)
		if err != nil || !bytes.Equal(stored, member.Data()) || deterministicPackageID(stored) != workItem.TaskPackageID {
			return invalid(CodeInventoryInvalid, "inventory.task_package")
		}
		decodedBytes, err := EncodeItemTask(decoded)
		if err != nil || !bytes.Equal(decodedBytes, stored) {
			return invalid(CodeInventoryInvalid, "inventory.task_package")
		}
		generatedBytes += int64(len(generated.Package))
		if generatedBytes > MaximumGeneratedBytes {
			return invalid(CodeLimitExceeded, "generated_packages")
		}
	}
	return nil
}

func validateBasisItem(
	basis inventoryBasis,
	canonicalDigest string,
	index int,
	item WorklistItem,
	task ItemTask,
	coverage CoverageRow,
) error {
	subject := basis.Subjects[index]
	switch basis.Kind {
	case "checklist":
		itemKey, ok := subject["item_key"].(string)
		if !ok || itemKey != item.ItemKey || item.Kind != "checklist" || task.Checklist == nil ||
			!equalStringSlices(coverage.Requested, task.Checklist.RequiredEvidence) || len(coverage.Gaps) != 0 ||
			task.Checklist.ReviewPolicy == "manual" &&
				item.ApprovalRequirement != ApprovalHumanReview &&
				item.ApprovalRequirement != ApprovalActiveCheck ||
			!sameCanonicalValue(subject, map[string]any{
				"item_key": item.ItemKey, "version": task.Checklist.Version, "statement": task.Checklist.Statement,
				"applicability": task.Checklist.Applicability, "allowed_methods": task.Checklist.AllowedMethods,
				"required_evidence": task.Checklist.RequiredEvidence, "review_policy": task.Checklist.ReviewPolicy,
			}) {
			return invalid(CodeInventoryInvalid, "inventory.checklist")
		}
	case "standard-mappings":
		if task.Checklist == nil || task.Standard == nil || item.Kind != "standard-mapping" ||
			item.ItemKey != task.Standard.MappingKey || item.SubjectKey != task.Standard.MappingKey ||
			!equalStringSlices(coverage.Requested, task.Checklist.RequiredEvidence) || len(coverage.Gaps) != 0 ||
			task.Checklist.ReviewPolicy == "manual" &&
				item.ApprovalRequirement != ApprovalHumanReview &&
				item.ApprovalRequirement != ApprovalActiveCheck ||
			!sameCanonicalValue(subject, map[string]any{
				"item_key": item.ItemKey, "subject_key": item.SubjectKey,
				"checklist": task.Checklist, "standard": task.Standard,
			}) {
			return invalid(CodeInventoryInvalid, "inventory.standard_mappings")
		}
		if basis.Selection != nil && (len(task.Standard.EntryIDs) != 1 ||
			!containsString(basis.Selection.EntryIDs, task.Standard.EntryIDs[0])) {
			return invalid(CodeInventoryInvalid, "inventory.standard_selection")
		}
	case "openapi-operations":
		pathTemplate, pathOK := subject["path"].(string)
		method, methodOK := subject["method"].(string)
		expectedKey, keyErr := openAPIOperationKey(canonicalDigest, pathTemplate, method)
		if !pathOK || !methodOK || task.Operation == nil || item.Kind != "operation-trace" ||
			task.Operation.Path != pathTemplate || task.Operation.Method != method ||
			keyErr != nil || item.ItemKey != expectedKey ||
			!equalStringSlices(coverage.Requested, []string{"operation-resolution"}) ||
			!equalStringSlices(coverage.Gaps, task.Operation.Gaps) ||
			!sameCanonicalValue(subject, map[string]any{
				"path": task.Operation.Path, "method": task.Operation.Method,
				"resolved": task.Operation.Resolved, "gaps": task.Operation.Gaps,
			}) {
			return invalid(CodeInventoryInvalid, "inventory.operation")
		}
	case "finding-candidates":
		if task.Finding == nil || item.Kind != "finding-verification" ||
			!equalStringSlices(coverage.Requested, []string{task.Finding.Method}) ||
			!equalStringSlices(coverage.Gaps, task.Finding.Limitations) ||
			!sameCanonicalValue(subject, map[string]any{
				"receipt_id":             task.Finding.ReceiptID,
				"proposal_ref":           task.Finding.ProposalRef,
				"proposal_digest":        task.Finding.ProposalDigest,
				"proposed_check_ordinal": task.Finding.ProposedCheckOrdinal,
				"objective":              task.Finding.Objective,
				"method":                 task.Finding.Method,
				"subject_key":            task.SubjectKey,
				"limitations":            task.Finding.Limitations,
			}) {
			return invalid(CodeInventoryInvalid, "inventory.finding")
		}
	}
	return nil
}

func sameCanonicalValue(left, right any) bool {
	leftBytes, leftErr := canonicalJSON(left)
	rightBytes, rightErr := canonicalJSON(right)
	return leftErr == nil && rightErr == nil && bytes.Equal(leftBytes, rightBytes)
}

func sameArtifactRef(left, right contracts.ArtifactRef) bool {
	if left.Namespace != right.Namespace || left.Name != right.Name || (left.Revision == nil) != (right.Revision == nil) {
		return false
	}
	return left.Revision == nil || *left.Revision == *right.Revision
}

func copyArtifactRef(source contracts.ArtifactRef) contracts.ArtifactRef {
	result := contracts.ArtifactRef{Namespace: source.Namespace, Name: source.Name}
	if source.Revision != nil {
		revision := *source.Revision
		result.Revision = &revision
	}
	return result
}

func equalStringSlices(left, right []string) bool {
	if len(left) != len(right) || (left == nil) != (right == nil) {
		return false
	}
	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}
	return true
}

func validateInventoryOptions(options InventoryOptions, sourceMediaType string) error {
	if options.Round <= 0 || validateIdentifier(options.WorkflowRole, "workflow_role") != nil ||
		validateIdentifier(options.SourceInputName, "source_input_name") != nil || !validApprovalRequirement(options.ApprovalRequirement) ||
		!validMediaType(sourceMediaType) || options.SourceRef.ValidateExact() != nil || len(options.Scope) > 64 {
		return invalid(CodeInventoryInvalid, "options")
	}
	for key, value := range options.Scope {
		if validateIdentifier(key, "scope") != nil || validateText(value, "scope", false) != nil {
			return invalid(CodeInventoryInvalid, "options.scope")
		}
	}
	return nil
}

func deterministicPackageID(document []byte) string {
	digest := digestBytes(append([]byte("contractor.audit.task-package.v1\x00"), document...))
	return "task-" + strings.TrimPrefix(digest, "sha256:")
}

func cloneMap(value map[string]any) map[string]any {
	if value == nil {
		return nil
	}
	result := make(map[string]any, len(value))
	for key, child := range value {
		result[key] = cloneJSONValue(child)
	}
	return result
}

func cloneJSONValue(value any) any {
	switch typed := value.(type) {
	case map[string]any:
		return cloneMap(typed)
	case []any:
		result := make([]any, len(typed))
		for index, child := range typed {
			result[index] = cloneJSONValue(child)
		}
		return result
	default:
		return typed
	}
}

func copyStrings(values []string) []string {
	if values == nil {
		return nil
	}
	return append([]string{}, values...)
}

func arrayStrings(values []string) []string {
	return append([]string{}, values...)
}

func sortedMapKeys(value map[string]any) []string {
	result := make([]string, 0, len(value))
	for key := range value {
		result = append(result, key)
	}
	sort.Strings(result)
	return result
}

func openAPIOperationKey(inventoryDigest, pathTemplate, method string) (string, error) {
	tuple, err := canonicalJSON([]any{
		"contractor.audit.openapi-operation.v1", inventoryDigest, pathTemplate, method,
	})
	if err != nil {
		return "", invalid(CodeInventoryInvalid, "operation_key")
	}
	return "op-" + strings.TrimPrefix(digestBytes(tuple), "sha256:"), nil
}

func mapValue(value any, field string) (map[string]any, error) {
	result, ok := value.(map[string]any)
	if !ok {
		return nil, invalid(CodeInventoryInvalid, field)
	}
	return result, nil
}

func stringValue(value any, field string) (string, error) {
	result, ok := value.(string)
	if !ok || validateText(result, field, true) != nil {
		return "", invalid(CodeInventoryInvalid, field)
	}
	return result, nil
}

func formatGap(kind, location string) string {
	return kind + ":" + location
}
