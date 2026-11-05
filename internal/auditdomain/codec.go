package auditdomain

import (
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
)

func DecodeCheckResultSet(data []byte) (CheckResultSet, error) {
	return decodeDocument(data, validateCheckResultSet)
}

func DecodeDirectVerificationSet(data []byte) (DirectVerificationSet, error) {
	return decodeDocument(data, validateDirectVerificationSet)
}

func EncodeFindingProposal(value FindingProposal) ([]byte, error) {
	return encodeDocument(value, validateFindingProposal)
}

func DecodeFindingProposal(data []byte) (FindingProposal, error) {
	return decodeDocument(data, validateFindingProposal)
}

func DecodeEvidence(data []byte) (EvidenceEnvelope, error) {
	return decodeDocument(data, validateEvidence)
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
	raw, err := decodeStrictJSON(data, &result)
	if err != nil {
		return result, err
	}
	switch any(result).(type) {
	case ExecutionManifest:
		if object, ok := raw.(map[string]any); ok {
			if items, ok := object["items"].([]any); ok {
				for _, item := range items {
					if err = rejectNullItemAuthority(item); err != nil {
						return result, err
					}
				}
			}
		}
	}
	if err := validate(result); err != nil {
		return result, err
	}
	return result, nil
}

func rejectNullItemAuthority(raw any) error {
	if object, ok := raw.(map[string]any); ok {
		for _, name := range []string{"accepted_plan", "dependencies"} {
			if value, present := object[name]; present && value == nil {
				return invalid(CodeInvalid, "item.authority_null")
			}
		}
	}
	return nil
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
			!contentdigest.Valid(item.TaskPackageDigest) {
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
			if validateIdentifier(input.Name, "items.inputs.name") != nil || input.Name <= previous || !contentdigest.Valid(input.Digest) {
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
	if !contentdigest.Valid(value.ExecutionManifestDigest) || len(value.Results) == 0 || len(value.Results) > MaximumItems {
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
		previousProposal := ""
		for _, proposal := range result.Proposals {
			if validateIdentifier(proposal.InvocationID, "results.proposals.invocation_id") != nil ||
				validateIdentifier(proposal.ClientKey, "results.proposals.client_key") != nil {
				return invalid(CodeResultSetInvalid, "results.proposals")
			}
			key := proposal.InvocationID + "\x00" + proposal.ClientKey
			if key <= previousProposal {
				return invalid(CodeResultSetInvalid, "results.proposals")
			}
			previousProposal = key
		}
		if len(result.Proposals) > MaximumProposalsPerItem {
			return invalid(CodeResultSetInvalid, "results.proposals")
		}
	}
	return nil
}

func validateDirectVerificationSet(value DirectVerificationSet) error {
	if value.Schema != DirectVerificationsSchema {
		return invalid(CodeSchemaUnsupported, "schema")
	}
	if len(value.Verifications) == 0 || len(value.Verifications) > MaximumItems {
		return invalid(CodeLimitExceeded, "verifications")
	}
	previous := ""
	for _, result := range value.Verifications {
		if validateIdentifier(result.InvocationID, "verifications.invocation_id") != nil ||
			validateIdentifier(result.ClientKey, "verifications.client_key") != nil ||
			!validDirectAssessment(result.Assessment) ||
			validateText(result.Summary, "verifications.summary", true) != nil ||
			validateSortedStrings(
				result.EvidenceIDs, MaximumEvidencePerItem,
				"verifications.evidence_ids", true,
			) != nil {
			return invalid(CodeResultSetInvalid, "verifications")
		}
		identity := result.InvocationID + "\x00" + result.ClientKey
		if identity <= previous {
			return invalid(CodeResultSetInvalid, "verifications.identity")
		}
		previous = identity
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
	if proposal.EvidenceIDs == nil || proposal.ProposedChecks == nil {
		return invalid(CodeInvalid, "proposal.arrays")
	}
	if validateIdentifier(proposal.ClientKey, "client_key") != nil {
		return invalid(CodeInvalid, "proposal")
	}
	if err := validateFindingReportProjection(ProjectFindingReport(proposal)); err != nil {
		return err
	}
	if err := validateFindingEvidenceReferences(proposal); err != nil {
		return err
	}
	if validateUniqueStrings(proposal.EvidenceIDs, MaximumEvidencePerItem, "evidence_ids", true) != nil ||
		len(proposal.ProposedChecks) > MaximumCoverageValues {
		return invalid(CodeLimitExceeded, "proposal")
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
		if evidence.HTTPRequestID != "" || evidence.HTTPRequestTag != "" {
			id, err := strconv.ParseUint(evidence.HTTPRequestID, 10, 64)
			if err != nil || id == 0 || id > contracts.MaxExactInteger || strconv.FormatUint(id, 10) != evidence.HTTPRequestID ||
				validateIdentifier(evidence.HTTPRequestTag, "evidence.http_request_tag") != nil || evidence.ObservedAt == "" || !strings.HasSuffix(evidence.ObservedAt, "Z") || evidence.ContentMemberID == "" ||
				len(evidence.DeploymentMarker) > 256 || strings.IndexFunc(evidence.DeploymentMarker, func(r rune) bool { return r < 32 }) >= 0 {
				return invalid(CodeInvalid, "evidence.http_observation")
			}
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

func validAssessment(value string) bool {
	switch value {
	case "supported", "refuted", "inconclusive", "blocked", "satisfied", "violated", "not-tested", "not-applicable":
		return true
	default:
		return false
	}
}

func validDirectAssessment(value string) bool {
	switch value {
	case "supported", "refuted", "inconclusive", "blocked":
		return true
	default:
		return false
	}
}

func validCoverageStatus(value string) bool {
	_, ok := CoverageStatusBucket(value)
	return ok
}

func validSeverity(value string) bool {
	switch value {
	case "", "informational", "low", "medium", "high", "critical":
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
