package auditservice

import (
	"fmt"

	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts/scan"
)

// auditRulepackInputs follows the resolved execution bindings rather than a
// profile, role or input name. Audit sources remain ProjectScope even when an
// ordinary owner-rulepack Workflow uses UserScope at Run admission.
func auditRulepackInputs(profile auditprofile.ResolvedAuditProfile) map[string]bool {
	names := make(map[string]bool)
	for _, role := range profile.Workflows {
		for _, slot := range resolved.NucleiRulepackInputs(role.Workflow) {
			mapping, ok := role.Inputs[slot]
			if ok && mapping.Source == auditprofile.AuditInputFromAudit {
				names[mapping.Name] = true
			}
		}
	}
	return names
}

func validateAuditRulepackMetadata(name, mediaType string, size int64) error {
	if !scan.ValidNucleiRulepackMetadata(mediaType, size) {
		return fmt.Errorf("%w: Audit rulepack input %q has unsupported media or exceeds its byte limit", ErrInvalid, name)
	}
	return nil
}
