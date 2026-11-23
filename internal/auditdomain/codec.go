package auditdomain

import (
	"sort"
)

func DecodeDirectVerificationSet(data []byte) (DirectVerificationSet, error) {
	return decodeDocument(data, validateDirectVerificationSet)
}

func EncodeFindingProposal(value FindingProposal) ([]byte, error) {
	return encodeDocument(value, validateFindingProposal)
}

func DecodeFindingProposal(data []byte) (FindingProposal, error) {
	return decodeDocument(data, validateFindingProposal)
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
	_, err := decodeStrictJSON(data, &result)
	if err != nil {
		return result, err
	}
	if err := validate(result); err != nil {
		return result, err
	}
	return result, nil
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

func validDirectAssessment(value string) bool {
	switch value {
	case "supported", "refuted", "inconclusive", "blocked":
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
