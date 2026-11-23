package evalservice

import (
	"slices"

	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/evaldomain"
)

func NativePolicySHA256() string {
	return nativePolicyV2SHA256
}
func RegisteredChecks() []string {
	return []string{"human-review@1", "required-artifact@1", "media-type@1", "json-schema@1"}
}
func RegisteredSchemas() []string {
	return []string{"json@1", "contractor.audit.result.v1", "contractor.audit.conclusions.v1", auditdomain.FindingProposalSchema, auditdomain.FindingCollectionSchema}
}

func validateNativeCheck(c evaldomain.Check) error {
	if !slices.Contains(RegisteredChecks(), c.Evaluator) {
		return evaldomain.Failure("eval_not_ready")
	}
	allowed := map[string]bool{}
	switch c.Evaluator {
	case "human-review@1":
		return c.ValidateHumanReviewParameters()
	case "required-artifact@1":
		allowed["output"] = true
	case "media-type@1":
		allowed["output"], allowed["mediaType"] = true, true
		if c.Parameters["output"] == "" || c.Parameters["mediaType"] == "" {
			return evaldomain.Failure("eval_invalid")
		}
	case "json-schema@1":
		allowed["output"], allowed["schema"] = true, true
		if c.Parameters["output"] == "" || !slices.Contains(RegisteredSchemas(), c.Parameters["schema"]) {
			return evaldomain.Failure("eval_not_ready")
		}
	}
	for key := range c.Parameters {
		if !allowed[key] {
			return evaldomain.Failure("eval_invalid")
		}
	}
	if c.ImplementationSHA256 != "" && c.ImplementationSHA256 != NativePolicySHA256() {
		return evaldomain.Failure("eval_pin_mismatch")
	}
	return nil
}
func validateRegisteredSchema(schema string, raw []byte) error {
	// These are compiled-in existing bounded decoders. Parameters cannot supply
	// a module, URL, filesystem path or executable uploaded schema extension.
	switch schema {
	case "json@1":
		_, err := evaldomain.StrictOutputJSON(raw)
		return err
	case "contractor.audit.result.v1", "contractor.audit.conclusions.v1":
		return auditcontract.ValidateResultShape(schema, raw)
	case auditdomain.FindingProposalSchema:
		_, err := auditdomain.DecodeFindingProposal(raw)
		return err
	case auditdomain.FindingCollectionSchema:
		_, err := auditdomain.DecodeFindingCollection(raw)
		return err
	}
	return evaldomain.Failure("eval_not_ready")
}
