package auditdomain

func (evidence *Evidence) UnmarshalJSON(data []byte) error {
	type plain Evidence
	var value plain
	raw, err := decodeStrictJSON(data, &value)
	if err != nil {
		return err
	}
	if err := validateFindingJSONFields(raw, "id kind summary", "artifact content_member_id observed_at deployment_marker http_request_id http_request_tag", ""); err != nil {
		return err
	}
	for key, field := range raw.(map[string]any) {
		if field == "" {
			return invalid(CodeInvalid, "evidence."+key)
		}
	}
	*evidence = Evidence(value)
	return nil
}
