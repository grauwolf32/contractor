package auditdomain

import (
	"sort"
	"strings"
	"time"
)

func EncodeWorklist(value WorklistManifest) ([]byte, error) {
	return encodeDocument(value, validateWorklist)
}

func DecodeWorklist(data []byte) (WorklistManifest, error) {
	return decodeDocument(data, validateWorklist)
}

func EncodeItemTask(value ItemTask) ([]byte, error) {
	return encodeDocument(value, validateItemTask)
}

func DecodeItemTask(data []byte) (ItemTask, error) {
	return decodeDocument(data, validateItemTask)
}

func EncodeExecutionManifest(value ExecutionManifest) ([]byte, error) {
	return encodeDocument(value, validateExecutionManifest)
}

func DecodeExecutionManifest(data []byte) (ExecutionManifest, error) {
	return decodeDocument(data, validateExecutionManifest)
}

func EncodeCheckResultSet(value CheckResultSet) ([]byte, error) {
	return encodeDocument(value, validateCheckResultSet)
}

func DecodeCheckResultSet(data []byte) (CheckResultSet, error) {
	return decodeDocument(data, validateCheckResultSet)
}

func EncodeFindingProposal(value FindingProposal) ([]byte, error) {
	return encodeDocument(value, validateFindingProposal)
}

func DecodeFindingProposal(data []byte) (FindingProposal, error) {
	return decodeDocument(data, validateFindingProposal)
}

func EncodeEvidence(value EvidenceEnvelope) ([]byte, error) {
	return encodeDocument(value, validateEvidence)
}

func DecodeEvidence(data []byte) (EvidenceEnvelope, error) {
	return decodeDocument(data, validateEvidence)
}

func EncodeCoverage(value CoverageEnvelope) ([]byte, error) {
	return encodeDocument(value, validateCoverage)
}

func DecodeCoverage(data []byte) (CoverageEnvelope, error) {
	return decodeDocument(data, validateCoverage)
}

func encodeDocument[T any](value T, validate func(T) error) ([]byte, error) {
	if err := validate(value); err != nil {
		return nil, err
	}
	encoded, err := canonicalJSON(value)
	if err != nil || len(encoded) > MaximumDocumentBytes {
		return nil, invalid(CodeLimitExceeded, "document")
	}
	return encoded, nil
}

func decodeDocument[T any](data []byte, validate func(T) error) (T, error) {
	var result T
	if len(data) > MaximumDocumentBytes {
		return result, invalid(CodeLimitExceeded, "document")
	}
	if _, err := decodeStrictJSON(data, &result); err != nil {
		return result, err
	}
	if err := validate(result); err != nil {
		return result, err
	}
	return result, nil
}

func DigestExecutionManifest(value ExecutionManifest) (string, error) {
	encoded, err := EncodeExecutionManifest(value)
	if err != nil {
		return "", err
	}
	return digestBytes(encoded), nil
}

// ValidateResultSet verifies complete, exact membership. It returns no partial
// result projection on failure.
func ValidateResultSet(value CheckResultSet, manifest ExecutionManifest) error {
	if err := validateCheckResultSet(value); err != nil {
		return err
	}
	if err := validateExecutionManifest(manifest); err != nil {
		return err
	}
	digest, err := DigestExecutionManifest(manifest)
	if err != nil || value.ExecutionManifestDigest != digest || len(value.Results) != len(manifest.Items) {
		return invalid(CodeResultSetInvalid, "execution_manifest")
	}
	results := make(map[string]CheckResult, len(value.Results))
	for _, result := range value.Results {
		if _, duplicate := results[result.ItemKey]; duplicate {
			return invalid(CodeResultSetInvalid, "results.item_key")
		}
		results[result.ItemKey] = result
	}
	for _, item := range manifest.Items {
		result, exists := results[item.ItemKey]
		if !exists || result.SubjectKey != item.SubjectKey {
			return invalid(CodeResultSetInvalid, "results.membership")
		}
	}
	return nil
}

func validateWorklist(value WorklistManifest) error {
	if value.Schema != WorklistSchema {
		return invalid(CodeSchemaUnsupported, "schema")
	}
	if value.Round <= 0 || len(value.Items) > MaximumItems || value.Items == nil {
		return invalid(CodeLimitExceeded, "items")
	}
	seenKeys := make(map[string]struct{}, len(value.Items))
	seenPackages := make(map[string]struct{}, len(value.Items))
	for index, item := range value.Items {
		if item.Ordinal != index || validateIdentifier(item.ItemKey, "items.item_key") != nil ||
			validateIdentifier(item.Kind, "items.kind") != nil || validateIdentifier(item.SubjectKey, "items.subject_key") != nil ||
			validateIdentifier(item.WorkflowRole, "items.workflow_role") != nil || validateIdentifier(item.TaskPackageID, "items.task_package_id") != nil ||
			!validApprovalRequirement(item.ApprovalRequirement) {
			return invalid(CodeInventoryInvalid, "items")
		}
		if _, duplicate := seenKeys[item.ItemKey]; duplicate {
			return invalid(CodeInventoryInvalid, "items.item_key")
		}
		if _, duplicate := seenPackages[item.TaskPackageID]; duplicate {
			return invalid(CodeInventoryInvalid, "items.task_package_id")
		}
		seenKeys[item.ItemKey] = struct{}{}
		seenPackages[item.TaskPackageID] = struct{}{}
	}
	return nil
}

func validateItemTask(value ItemTask) error {
	if value.Schema != TaskSchema {
		return invalid(CodeSchemaUnsupported, "schema")
	}
	for field, candidate := range map[string]string{
		"item_key": value.ItemKey, "kind": value.Kind, "subject_key": value.SubjectKey,
		"workflow_role": value.WorkflowRole,
	} {
		if err := validateIdentifier(candidate, field); err != nil {
			return err
		}
	}
	if !validDigest(value.SourceContentDigest) || !validDigest(value.CanonicalInventoryDigest) || !validMediaType(value.SourceMediaType) || value.SourceMediaType != normalizedMediaType(value.SourceMediaType) {
		return invalid(CodeInvalid, "digest")
	}
	if err := value.SourceRef.ValidateExact(); err != nil {
		return invalid(CodeReferenceInvalid, "source_ref")
	}
	if len(value.Scope) > 64 {
		return invalid(CodeLimitExceeded, "scope")
	}
	for key, candidate := range value.Scope {
		if validateIdentifier(key, "scope") != nil || validateText(candidate, "scope", false) != nil {
			return invalid(CodeInvalid, "scope")
		}
	}
	if (value.Checklist == nil) == (value.Operation == nil) {
		return invalid(CodeInvalid, "task")
	}
	if value.Checklist != nil {
		if value.Kind != "checklist" {
			return invalid(CodeInvalid, "kind")
		}
		return validateChecklistTask(*value.Checklist)
	}
	if value.Kind != "operation-trace" {
		return invalid(CodeInvalid, "kind")
	}
	return validateOperationTask(*value.Operation)
}

func validateChecklistTask(value ChecklistTask) error {
	if validateText(value.Version, "checklist.version", true) != nil ||
		validateText(value.Statement, "checklist.statement", true) != nil ||
		validateText(value.Applicability, "checklist.applicability", true) != nil ||
		value.ReviewPolicy != "automatic" && value.ReviewPolicy != "manual" {
		return invalid(CodeInvalid, "checklist")
	}
	if err := validateSortedStrings(value.AllowedMethods, MaximumCoverageValues, "checklist.allowed_methods", true); err != nil {
		return err
	}
	return validateSortedStrings(value.RequiredEvidence, MaximumEvidencePerItem, "checklist.required_evidence", true)
}

func validateOperationTask(value OperationTask) error {
	if !strings.HasPrefix(value.Path, "/") || validateText(value.Path, "operation.path", true) != nil || !validHTTPMethod(value.Method) ||
		validateText(value.OperationID, "operation.operation_id", false) != nil || value.Resolved == nil {
		return invalid(CodeInvalid, "operation")
	}
	return validateSortedStrings(value.Gaps, MaximumCoverageValues, "operation.gaps", false)
}

func validateExecutionManifest(value ExecutionManifest) error {
	if value.Schema != ExecutionManifestSchema {
		return invalid(CodeSchemaUnsupported, "schema")
	}
	if value.Items == nil || len(value.Items) > MaximumItems {
		return invalid(CodeLimitExceeded, "items")
	}
	seen := make(map[string]struct{}, len(value.Items))
	seenPackages := make(map[string]struct{}, len(value.Items))
	for index, item := range value.Items {
		if item.Ordinal != index || validateIdentifier(item.ItemKey, "items.item_key") != nil ||
			validateIdentifier(item.SubjectKey, "items.subject_key") != nil || validateIdentifier(item.TaskPackageID, "items.task_package_id") != nil ||
			!validDigest(item.TaskPackageDigest) {
			return invalid(CodeInvalid, "items")
		}
		if _, duplicate := seen[item.ItemKey]; duplicate {
			return invalid(CodeInvalid, "items.item_key")
		}
		seen[item.ItemKey] = struct{}{}
		if _, duplicate := seenPackages[item.TaskPackageID]; duplicate {
			return invalid(CodeInvalid, "items.task_package_id")
		}
		seenPackages[item.TaskPackageID] = struct{}{}
		if item.TaskRef != nil {
			if err := item.TaskRef.ValidateExact(); err != nil {
				return invalid(CodeReferenceInvalid, "items.task_ref")
			}
		}
		previous := ""
		for _, input := range item.Inputs {
			if validateIdentifier(input.Name, "items.inputs.name") != nil || input.Name <= previous || !validDigest(input.Digest) {
				return invalid(CodeInvalid, "items.inputs")
			}
			if err := input.Ref.ValidateExact(); err != nil {
				return invalid(CodeReferenceInvalid, "items.inputs.ref")
			}
			previous = input.Name
		}
	}
	return nil
}

// ValidateDispatchExecutionManifest applies the additional exact-reference
// gate required immediately before a trusted child Run is created. Inventory
// construction may use the same codec before task packages have revisions.
func ValidateDispatchExecutionManifest(value ExecutionManifest) error {
	if err := validateExecutionManifest(value); err != nil {
		return err
	}
	for _, item := range value.Items {
		if item.TaskRef == nil {
			return invalid(CodeReferenceInvalid, "items.task_ref")
		}
	}
	return nil
}

func validateCheckResultSet(value CheckResultSet) error {
	if value.Schema != CheckResultsSchema {
		return invalid(CodeSchemaUnsupported, "schema")
	}
	if !validDigest(value.ExecutionManifestDigest) || len(value.Results) == 0 || len(value.Results) > MaximumItems {
		return invalid(CodeResultSetInvalid, "results")
	}
	seen := make(map[string]struct{}, len(value.Results))
	for _, result := range value.Results {
		if validateIdentifier(result.ItemKey, "results.item_key") != nil || validateIdentifier(result.SubjectKey, "results.subject_key") != nil ||
			!validAssessment(result.Assessment) || validateText(result.Summary, "results.summary", true) != nil {
			return invalid(CodeResultSetInvalid, "results")
		}
		if _, duplicate := seen[result.ItemKey]; duplicate {
			return invalid(CodeResultSetInvalid, "results.item_key")
		}
		seen[result.ItemKey] = struct{}{}
		if err := validateUniqueStrings(result.EvidenceIDs, MaximumEvidencePerItem, "results.evidence_ids", true); err != nil {
			return invalid(CodeResultSetInvalid, "results.evidence_ids")
		}
		if err := validateResultCoverage(result.Coverage); err != nil {
			return err
		}
		if err := validateUniqueStrings(result.Proposals, MaximumProposalsPerItem, "results.proposals", true); err != nil {
			return invalid(CodeResultSetInvalid, "results.proposals")
		}
	}
	return nil
}

func validateResultCoverage(value ResultCoverage) error {
	if validateUniqueStrings(value.Requested, MaximumCoverageValues, "coverage.requested", true) != nil ||
		validateUniqueStrings(value.Completed, MaximumCoverageValues, "coverage.completed", true) != nil ||
		validateUniqueStrings(value.Gaps, MaximumCoverageValues, "coverage.gaps", false) != nil {
		return invalid(CodeResultSetInvalid, "coverage")
	}
	requested := make(map[string]struct{}, len(value.Requested))
	for _, item := range value.Requested {
		requested[item] = struct{}{}
	}
	for _, item := range value.Completed {
		if _, exists := requested[item]; !exists {
			return invalid(CodeResultSetInvalid, "coverage.completed")
		}
	}
	return nil
}

func validateFindingProposal(proposal FindingProposal) error {
	if proposal.Schema != FindingProposalSchema {
		return invalid(CodeSchemaUnsupported, "schema")
	}
	if validateIdentifier(proposal.ClientKey, "client_key") != nil || validateText(proposal.Title, "title", true) != nil ||
		validateText(proposal.Description, "description", true) != nil || validateIdentifier(proposal.Subject.Kind, "subject.kind") != nil ||
		validateIdentifier(proposal.Subject.Key, "subject.key") != nil || validateText(proposal.Hypothesis, "hypothesis", false) != nil ||
		!validSeverity(proposal.SeveritySuggestion) {
		return invalid(CodeInvalid, "proposal")
	}
	if validateStringList(proposal.Preconditions, MaximumCoverageValues, "preconditions") != nil ||
		validateUniqueStrings(proposal.EvidenceIDs, MaximumEvidencePerItem, "evidence_ids", true) != nil ||
		validateStringList(proposal.Limitations, MaximumCoverageValues, "limitations") != nil ||
		len(proposal.StandardRefs) > MaximumCoverageValues || len(proposal.ProposedChecks) > MaximumCoverageValues {
		return invalid(CodeLimitExceeded, "proposal")
	}
	for _, reference := range proposal.StandardRefs {
		if validateIdentifier(reference.Scheme, "standard_refs") != nil || validateText(reference.Version, "standard_refs", true) != nil || validateIdentifier(reference.RequirementID, "standard_refs") != nil {
			return invalid(CodeInvalid, "standard_refs")
		}
	}
	for _, check := range proposal.ProposedChecks {
		if validateText(check.Objective, "proposed_checks", true) != nil || validateIdentifier(check.Method, "proposed_checks") != nil {
			return invalid(CodeInvalid, "proposed_checks")
		}
	}
	return nil
}

func validateEvidence(value EvidenceEnvelope) error {
	if value.Schema != EvidenceSchema {
		return invalid(CodeSchemaUnsupported, "schema")
	}
	if value.Evidence == nil || len(value.Evidence) > MaximumItems {
		return invalid(CodeLimitExceeded, "evidence")
	}
	seen := make(map[string]struct{}, len(value.Evidence))
	for _, evidence := range value.Evidence {
		if validateIdentifier(evidence.ID, "evidence.id") != nil || validateIdentifier(evidence.Kind, "evidence.kind") != nil || validateText(evidence.Summary, "evidence.summary", true) != nil ||
			(evidence.Artifact == nil) == (evidence.ContentMemberID == "") {
			return invalid(CodeInvalid, "evidence")
		}
		if _, duplicate := seen[evidence.ID]; duplicate {
			return invalid(CodeInvalid, "evidence.id")
		}
		seen[evidence.ID] = struct{}{}
		if evidence.Artifact != nil {
			if err := evidence.Artifact.ValidateExact(); err != nil {
				return invalid(CodeReferenceInvalid, "evidence.artifact")
			}
		}
		if evidence.ContentMemberID != "" && validateIdentifier(evidence.ContentMemberID, "evidence.content_member_id") != nil {
			return invalid(CodeReferenceInvalid, "evidence.content_member_id")
		}
		if evidence.ObservedAt != "" {
			parsed, err := time.Parse(time.RFC3339Nano, evidence.ObservedAt)
			if err != nil || parsed.Format(time.RFC3339Nano) != evidence.ObservedAt {
				return invalid(CodeInvalid, "evidence.observed_at")
			}
		}
		if validateText(evidence.DeploymentMarker, "evidence.deployment_marker", false) != nil {
			return invalid(CodeInvalid, "evidence.deployment_marker")
		}
	}
	return nil
}

func validateCoverage(value CoverageEnvelope) error {
	if value.Schema != CoverageSchema {
		return invalid(CodeSchemaUnsupported, "schema")
	}
	if value.Rows == nil || len(value.Rows) > MaximumItems {
		return invalid(CodeLimitExceeded, "rows")
	}
	seen := make(map[string]struct{}, len(value.Rows))
	for _, row := range value.Rows {
		if validateIdentifier(row.ItemKey, "rows.item_key") != nil || validateIdentifier(row.SubjectKey, "rows.subject_key") != nil || !validCoverageStatus(row.Status) || validateText(row.Rationale, "rows.rationale", false) != nil {
			return invalid(CodeInvalid, "rows")
		}
		if _, duplicate := seen[row.ItemKey]; duplicate {
			return invalid(CodeInvalid, "rows.item_key")
		}
		seen[row.ItemKey] = struct{}{}
		if validateSortedStrings(row.Requested, MaximumCoverageValues, "rows.requested", true) != nil ||
			validateSortedStrings(row.Completed, MaximumCoverageValues, "rows.completed", true) != nil ||
			validateSortedStrings(row.Gaps, MaximumCoverageValues, "rows.gaps", false) != nil {
			return invalid(CodeInvalid, "rows")
		}
	}
	return nil
}

func validApprovalRequirement(value ApprovalRequirement) bool {
	return value == ApprovalNone || value == ApprovalActiveCheck || value == ApprovalHumanReview
}

func validAssessment(value string) bool {
	switch value {
	case "supported", "refuted", "inconclusive", "blocked", "satisfied", "violated", "not-tested", "not-applicable":
		return true
	default:
		return false
	}
}

func validCoverageStatus(value string) bool {
	switch value {
	case "not-tested", "inconclusive", "satisfied", "violated", "not-applicable", "blocked", "excluded",
		"traced-complete", "traced-partial", "unmapped":
		return true
	default:
		return false
	}
}

func validSeverity(value string) bool {
	switch value {
	case "", "informational", "low", "medium", "high", "critical":
		return true
	default:
		return false
	}
}

func validHTTPMethod(value string) bool {
	switch value {
	case "get", "put", "post", "delete", "options", "head", "patch", "trace":
		return true
	default:
		return false
	}
}

func validateSortedStrings(values []string, maximum int, field string, identifiers bool) error {
	if values == nil || len(values) > maximum {
		return invalid(CodeLimitExceeded, field)
	}
	previous := ""
	for _, value := range values {
		if value <= previous || identifiers && validateIdentifier(value, field) != nil || !identifiers && validateText(value, field, true) != nil {
			return invalid(CodeInvalid, field)
		}
		previous = value
	}
	return nil
}

func validateUniqueStrings(values []string, maximum int, field string, identifiers bool) error {
	if values == nil || len(values) > maximum {
		return invalid(CodeLimitExceeded, field)
	}
	seen := make(map[string]struct{}, len(values))
	for _, value := range values {
		if identifiers && validateIdentifier(value, field) != nil || !identifiers && validateText(value, field, true) != nil {
			return invalid(CodeInvalid, field)
		}
		if _, duplicate := seen[value]; duplicate {
			return invalid(CodeInvalid, field)
		}
		seen[value] = struct{}{}
	}
	return nil
}

func validateStringList(values []string, maximum int, field string) error {
	if values == nil || len(values) > maximum {
		return invalid(CodeLimitExceeded, field)
	}
	for _, value := range values {
		if validateText(value, field, true) != nil {
			return invalid(CodeInvalid, field)
		}
	}
	return nil
}

func sortedUnique(values []string) []string {
	seen := make(map[string]struct{}, len(values))
	for _, value := range values {
		seen[value] = struct{}{}
	}
	result := make([]string, 0, len(seen))
	for value := range seen {
		result = append(result, value)
	}
	sort.Strings(result)
	return result
}
