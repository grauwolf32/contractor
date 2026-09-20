package evalservice

import (
	_ "embed"
	"slices"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/evaldomain"
)

//go:embed native_checks.go
var nativeCheckSource []byte

//go:embed native_check_execution.go
var nativeCheckExecutionSource []byte

func NativePolicySHA256() string {
	source := append(append([]byte{}, nativeCheckSource...), nativeCheckExecutionSource...)
	return evaldomain.Digest(source)
}
func RegisteredChecks() []string {
	return []string{"human-review@1", "required-artifact@1", "media-type@1", "json-schema@1"}
}
func RegisteredSchemas() []string {
	return []string{"json@1", auditdomain.EvidenceSchema, auditdomain.CoverageSchema, auditdomain.CheckResultsSchema, auditdomain.FindingProposalSchema, auditdomain.FindingCollectionSchema}
}

func validateNativeCheck(c evaldomain.Check) error {
	if !slices.Contains(RegisteredChecks(), c.Evaluator) {
		return evaldomain.Failure("eval_not_ready")
	}
	allowed := map[string]bool{}
	switch c.Evaluator {
	case "human-review@1":
		return nil
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
		_, err := evaldomain.StrictJSON(raw)
		return err
	case auditdomain.EvidenceSchema:
		_, err := auditdomain.DecodeEvidence(raw)
		return err
	case auditdomain.CoverageSchema:
		_, err := auditdomain.DecodeCoverage(raw)
		return err
	case auditdomain.CheckResultsSchema:
		_, err := auditdomain.DecodeCheckResultSet(raw)
		return err
	case auditdomain.FindingProposalSchema:
		_, err := auditdomain.DecodeFindingProposal(raw)
		return err
	case auditdomain.FindingCollectionSchema:
		_, err := auditdomain.DecodeFindingCollection(raw)
		return err
	}
	return evaldomain.Failure("eval_not_ready")
}
