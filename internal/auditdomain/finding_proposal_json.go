package auditdomain

import "strings"

func (proposal *FindingProposal) UnmarshalJSON(data []byte) error {
	type plain FindingProposal
	var decoded plain
	raw, err := decodeStrictJSON(data, &decoded)
	if err != nil {
		return err
	}
	fields, ok := raw.(map[string]any)
	if !ok {
		return invalid(CodeInvalid, "proposal")
	}
	err = validateFindingJSONFields(fields,
		"schema client_key title description subject preconditions standard_refs evidence_ids proposed_checks severity_suggestion limitations",
		"hypothesis locations http_exchange", "subject")
	if err != nil {
		return err
	}
	subject, present := fields["subject"]
	if !present || (subject != nil && decoded.Subject == nil) {
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
	*proposal = FindingProposal(decoded)
	return nil
}

func (location *FindingLocation) UnmarshalJSON(data []byte) error {
	type plain FindingLocation
	var decoded plain
	raw, err := decodeStrictJSON(data, &decoded)
	if err != nil {
		return err
	}
	fields, ok := raw.(map[string]any)
	if !ok {
		return invalid(CodeInvalid, "locations")
	}
	if err := validateFindingJSONFields(fields, "", "file line range url method", ""); err != nil {
		return err
	}
	for key, value := range fields {
		if value == nil || value == "" {
			return invalid(CodeInvalid, "locations."+key)
		}
	}
	if region, present := fields["range"]; present {
		if err := validateFindingJSONFields(region, "start_line end_line", "", ""); err != nil {
			return err
		}
	}
	*location = FindingLocation(decoded)
	return validateFindingLocation(*location)
}

func validateFindingJSONFields(value any, required, optional, nullable string) error {
	fields, ok := value.(map[string]any)
	if !ok {
		return invalid(CodeInvalid, "finding.object")
	}
	allowed := make(map[string]bool)
	for _, key := range strings.Fields(required + " " + optional) {
		allowed[key] = true
	}
	for _, key := range strings.Fields(required) {
		if _, present := fields[key]; !present {
			return invalid(CodeInvalid, "finding.object")
		}
	}
	for key, value := range fields {
		if !allowed[key] || (value == nil && key != nullable) {
			return invalid(CodeInvalid, "finding.object")
		}
	}
	return nil
}

func validateHTTPExchangeJSON(value any) error {
	if err := validateFindingJSONFields(value, "request_id request_tag attempts", "response_body_evidence_id", ""); err != nil {
		return err
	}
	fields := value.(map[string]any)
	attempts, ok := fields["attempts"].([]any)
	if !ok {
		return invalid(CodeInvalid, "http_exchange.attempts")
	}
	for _, raw := range attempts {
		if err := validateFindingJSONFields(raw, "method url headers body_base64", "status response_headers error", ""); err != nil {
			return err
		}
		attempt := raw.(map[string]any)
		for _, key := range []string{"headers", "response_headers"} {
			values, present := attempt[key]
			if !present {
				continue
			}
			headers, ok := values.([]any)
			if !ok {
				return invalid(CodeInvalid, "http_exchange.headers")
			}
			for _, header := range headers {
				if err := validateFindingJSONFields(header, "name value", "", ""); err != nil {
					return err
				}
			}
		}
	}
	return nil
}
