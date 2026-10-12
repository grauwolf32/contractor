package auditdomain

import "slices"

// FindingReportProjection is the part of a finding proposal that an Audit
// report shows. Finding intake derives it once from the validated proposal
// and stores it with the receipt, so report finalization reads bounded rows
// instead of one proposal artifact per finding. The proposal validator and
// the projection share one definition of these fields.
type FindingReportProjection struct {
	Title              string               `json:"title"`
	Description        string               `json:"description"`
	Subject            *FindingSubject      `json:"subject"`
	Hypothesis         string               `json:"hypothesis,omitempty"`
	StandardRefs       []StandardReference  `json:"standard_refs"`
	SeveritySuggestion string               `json:"severity_suggestion"`
	Limitations        []string             `json:"limitations"`
	Locations          []FindingLocation    `json:"locations,omitempty"`
	HTTPExchange       *FindingHTTPExchange `json:"http_exchange,omitempty"`
}

// ProjectFindingReport returns the report projection of a proposal.
func ProjectFindingReport(proposal FindingProposal) FindingReportProjection {
	return FindingReportProjection{
		Title: proposal.Title, Description: proposal.Description, Subject: proposal.Subject,
		Hypothesis: proposal.Hypothesis, StandardRefs: proposal.StandardRefs,
		SeveritySuggestion: proposal.SeveritySuggestion, Limitations: proposal.Limitations,
		Locations: proposal.Locations, HTTPExchange: proposal.HTTPExchange,
	}
}

// EncodeFindingReportProjection returns the canonical bytes of a valid
// projection, at most MaximumDocumentBytes.
func EncodeFindingReportProjection(value FindingReportProjection) ([]byte, error) {
	return encodeDocument(value, validateFindingReportProjection)
}

// DecodeFindingReportProjection strictly decodes and validates stored
// projection bytes.
func DecodeFindingReportProjection(data []byte) (FindingReportProjection, error) {
	return decodeDocument(data, validateFindingReportProjection)
}

func (projection *FindingReportProjection) UnmarshalJSON(data []byte) error {
	type plain FindingReportProjection
	var decoded plain
	raw, err := decodeStrictJSON(data, &decoded)
	if err != nil {
		return err
	}
	err = validateFindingJSONFields(raw,
		"title description subject standard_refs severity_suggestion limitations",
		"hypothesis locations http_exchange", "subject")
	if err != nil {
		return err
	}
	if err := validateFindingReportJSON(raw.(map[string]any), decoded.Subject != nil); err != nil {
		return err
	}
	*projection = FindingReportProjection(decoded)
	return nil
}

// validateFindingReportJSON checks the nested report fields of a decoded
// proposal or projection object: an explicit subject and a closed HTTP
// exchange shape.
func validateFindingReportJSON(fields map[string]any, decodedSubject bool) error {
	subject, present := fields["subject"]
	if !present || (subject != nil && !decodedSubject) {
		return invalid(CodeInvalid, "subject")
	}
	if subject != nil {
		if err := validateFindingJSONFields(subject, "kind key", "", ""); err != nil {
			return err
		}
	}
	if exchange, present := fields["http_exchange"]; present {
		if err := validateHTTPExchangeJSON(exchange); err != nil {
			return err
		}
	}
	return nil
}

// validateFindingReportProjection is the one definition of the report fields
// of a finding proposal; validateFindingProposal applies it too.
func validateFindingReportProjection(value FindingReportProjection) error {
	if value.StandardRefs == nil || value.Limitations == nil {
		return invalid(CodeInvalid, "proposal.arrays")
	}
	if validateText(value.Title, "title", true) != nil || validateText(value.Description, "description", true) != nil ||
		validateText(value.Hypothesis, "hypothesis", false) != nil || !validSeverity(value.SeveritySuggestion) {
		return invalid(CodeInvalid, "proposal")
	}
	if value.Subject != nil &&
		(validateIdentifier(value.Subject.Kind, "subject.kind") != nil || validateIdentifier(value.Subject.Key, "subject.key") != nil) {
		return invalid(CodeInvalid, "subject")
	}
	if len(value.Locations) > MaximumFindingLocations {
		return invalid(CodeLimitExceeded, "locations")
	}
	for _, location := range value.Locations {
		if err := validateFindingLocation(location); err != nil {
			return err
		}
	}
	if value.HTTPExchange != nil {
		if err := validateFindingHTTPExchange(*value.HTTPExchange); err != nil {
			return err
		}
	}
	if validateStringList(value.Limitations, MaximumCoverageValues, "limitations") != nil ||
		len(value.StandardRefs) > MaximumCoverageValues {
		return invalid(CodeLimitExceeded, "proposal")
	}
	for _, reference := range value.StandardRefs {
		if validateIdentifier(reference.Scheme, "standard_refs") != nil || validateText(reference.Version, "standard_refs", true) != nil || validateIdentifier(reference.RequirementID, "standard_refs") != nil {
			return invalid(CodeInvalid, "standard_refs")
		}
	}
	return nil
}

// validateFindingEvidenceReferences checks the proposal-only reference from an
// HTTP exchange to one of the proposal's evidence IDs.
func validateFindingEvidenceReferences(proposal FindingProposal) error {
	exchange := proposal.HTTPExchange
	if exchange != nil && exchange.ResponseBodyEvidenceID != "" &&
		!slices.Contains(proposal.EvidenceIDs, exchange.ResponseBodyEvidenceID) {
		return invalid(CodeReferenceInvalid, "http_exchange.response_body_evidence_id")
	}
	return nil
}
