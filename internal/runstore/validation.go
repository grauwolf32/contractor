package runstore

import (
	"encoding/json"
	"strings"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func validateReason(reason Reason) error {
	if strings.TrimSpace(reason.Code) == "" {
		return invalidf("state reason code is required")
	}
	return nil
}

func validateOpaque(field, value string) error {
	if strings.TrimSpace(value) == "" {
		return invalidf("%s is required", field)
	}
	return nil
}

// validateResourceID requires the public ResourceId grammar of
// contracts.ValidResourceID for an identity the public API shows, so no
// stored identity can fail a later strict read (API-R70).
func validateResourceID(field, value string) error {
	if !contracts.ValidResourceID(value) {
		return invalidf("%s must be a resource identifier of at most %d characters", field, contracts.MaxResourceIDLength)
	}
	return nil
}

// validateName requires the shared name grammar of contracts.ValidName for a
// Stage or logical Agent name (WFL-R84).
func validateName(field, value string) error {
	if !contracts.ValidName(value) {
		return invalidf("%s must be a name of at most %d characters", field, contracts.MaxNameLength)
	}
	return nil
}

func validateJSONObject(field string, value json.RawMessage) error {
	var object map[string]json.RawMessage
	if len(value) == 0 || json.Unmarshal(value, &object) != nil || object == nil {
		return invalidf("%s must be one JSON object", field)
	}
	return nil
}
