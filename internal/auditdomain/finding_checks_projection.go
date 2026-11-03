package auditdomain

import (
	"bytes"
	"encoding/json"
	"sort"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

const (
	FindingChecksProjectionSchema       = "contractor.audit.finding-checks-projection.v1"
	MaximumFindingChecksProjectionBytes = 256 << 10
)

type FindingFragment struct {
	Offset int64  `json:"offset" strictjson:"required"`
	Length int64  `json:"length" strictjson:"required"`
	Digest string `json:"digest" strictjson:"required"`
}

type FindingCheckProjection struct {
	Ordinal      int             `json:"ordinal"`
	Fragment     FindingFragment `json:"fragment"`
	SourceMethod string          `json:"source_method"`
	Status       string          `json:"status"`
	PlanSchema   string          `json:"plan_schema,omitempty"`
	Phase        string          `json:"phase,omitempty"`
	PlanDigest   string          `json:"plan_digest,omitempty"`
	PlanBytes    int64           `json:"plan_bytes,omitempty"`
}

func (entry *FindingCheckProjection) UnmarshalJSON(data []byte) error {
	type plain FindingCheckProjection
	var value plain
	if err := strictjson.Decode(data, &value); err != nil {
		return err
	}
	var members map[string]json.RawMessage
	if err := strictjson.Decode(data, &members); err != nil {
		return err
	}
	required := []string{"ordinal", "fragment", "source_method", "status"}
	if value.Status != "unsupported" {
		required = append(required, "plan_schema", "phase", "plan_digest", "plan_bytes")
	}
	if len(members) != len(required) {
		return invalid(CodeInvalid, "checks_projection.entry")
	}
	for _, name := range required {
		if _, ok := members[name]; !ok {
			return invalid(CodeInvalid, "checks_projection.entry")
		}
	}
	*entry = FindingCheckProjection(value)
	return nil
}

// FindingChecksProjection indexes the final immutable proposal bytes. It
// contains no objective or limitations prose and establishes no source grant.
type FindingChecksProjection struct {
	Schema         string                   `json:"schema" strictjson:"required"`
	ProposalDigest string                   `json:"proposal_digest" strictjson:"required"`
	SizeBytes      int64                    `json:"size_bytes" strictjson:"required"`
	Discarded      bool                     `json:"discarded" strictjson:"required"`
	Subject        *FindingFragment         `json:"subject" strictjson:"required,nullable"`
	Limitations    *FindingFragment         `json:"limitations" strictjson:"required,nullable"`
	Checks         []FindingCheckProjection `json:"checks" strictjson:"required"`
}

func BuildFindingChecksProjection(finalBytes []byte, validatedDoc FindingProposal) (FindingChecksProjection, error) {
	decoded, err := DecodeFindingProposal(finalBytes)
	if err != nil || !sameCanonicalValue(decoded, validatedDoc) {
		return FindingChecksProjection{}, invalid(CodeInvalid, "checks_projection.source")
	}
	result := FindingChecksProjection{Schema: FindingChecksProjectionSchema, ProposalDigest: DigestBytes(finalBytes), SizeBytes: int64(len(finalBytes)), Checks: []FindingCheckProjection{}}
	decoder := json.NewDecoder(bytes.NewReader(finalBytes))
	if _, err = decoder.Token(); err != nil {
		return result, err
	}
	for decoder.More() {
		member, readErr := decoder.Token()
		if readErr != nil {
			return result, readErr
		}
		if member == "proposed_checks" {
			if _, err = decoder.Token(); err != nil {
				return result, err
			}
			for decoder.More() {
				var raw json.RawMessage
				if err = decoder.Decode(&raw); err != nil {
					return result, err
				}
				ordinal := len(result.Checks)
				check := validatedDoc.ProposedChecks[ordinal]
				entry := FindingCheckProjection{Ordinal: ordinal, Fragment: projectionFragment(raw, decoder.InputOffset()), SourceMethod: check.Method, Status: "unsupported"}
				if _, known := contracts.LookupCheckPlanMethod(contracts.FindingCheckPlanSchema, check.Method); known {
					plan, planErr := BuildFindingCheckPlan(validatedDoc, ordinal)
					if planErr != nil {
						return result, planErr
					}
					encoded, planErr := EncodeCheckPlan(plan)
					if planErr != nil {
						return result, planErr
					}
					entry.Status = "registered"
					entry.PlanSchema = plan.Schema
					entry.Phase = plan.Phase
					entry.PlanDigest = DigestBytes(encoded)
					entry.PlanBytes = int64(len(encoded))
					if len(encoded) > MaximumAcceptedPlanBytes {
						entry.Status = "plan-too-large"
					}
				}
				result.Checks = append(result.Checks, entry)
			}
			if _, err = decoder.Token(); err != nil {
				return result, err
			}
		} else {
			var raw json.RawMessage
			if err = decoder.Decode(&raw); err != nil {
				return result, err
			}
			fragment := projectionFragment(raw, decoder.InputOffset())
			if member == "subject" {
				result.Subject = &fragment
			}
			if member == "limitations" {
				result.Limitations = &fragment
			}
		}
	}
	return result, ValidateFindingChecksProjection(result)
}

func projectionFragment(raw []byte, end int64) FindingFragment {
	return FindingFragment{Offset: end - int64(len(raw)), Length: int64(len(raw)), Digest: DigestBytes(raw)}
}

func EncodeFindingChecksProjection(value FindingChecksProjection) ([]byte, error) {
	if err := ValidateFindingChecksProjection(value); err != nil {
		return nil, err
	}
	return canonicalJSON(value)
}

func ValidateFindingChecksProjection(value FindingChecksProjection) error {
	if value.Schema != FindingChecksProjectionSchema || !contentdigest.Valid(value.ProposalDigest) || value.SizeBytes < 1 || value.SizeBytes > MaximumDocumentBytes || value.Checks == nil || len(value.Checks) > MaximumCoverageValues {
		return invalid(CodeInvalid, "checks_projection")
	}
	if value.Discarded {
		if value.Subject != nil || value.Limitations != nil || len(value.Checks) != 0 {
			return invalid(CodeInvalid, "checks_projection.discarded")
		}
	} else {
		if value.Subject == nil || value.Limitations == nil {
			return invalid(CodeInvalid, "checks_projection.context")
		}
		for _, fragment := range []*FindingFragment{value.Subject, value.Limitations} {
			if !validFindingFragment(*fragment, value.SizeBytes) {
				return invalid(CodeInvalid, "checks_projection.context")
			}
		}
		fragments := []FindingFragment{*value.Subject, *value.Limitations}
		previousEnd := int64(-1)
		for ordinal, entry := range value.Checks {
			if entry.Ordinal != ordinal || !validFindingFragment(entry.Fragment, value.SizeBytes) || entry.Fragment.Offset < previousEnd || validateIdentifier(entry.SourceMethod, "checks_projection.method") != nil {
				return invalid(CodeInvalid, "checks_projection.entry")
			}
			previousEnd = entry.Fragment.Offset + entry.Fragment.Length
			fragments = append(fragments, entry.Fragment)
			key, known := contracts.LookupCheckPlanMethod(contracts.FindingCheckPlanSchema, entry.SourceMethod)
			if entry.Status == "unsupported" {
				if known || entry.PlanSchema != "" || entry.Phase != "" || entry.PlanDigest != "" || entry.PlanBytes != 0 {
					return invalid(CodeInvalid, "checks_projection.status")
				}
			} else if !known || key.Schema != entry.PlanSchema || key.Phase != entry.Phase || !contentdigest.Valid(entry.PlanDigest) || entry.PlanBytes < 1 || entry.PlanBytes > MaximumDocumentBytes ||
				(entry.Status != "registered" && entry.Status != "plan-too-large") || (entry.Status == "registered") != (entry.PlanBytes <= MaximumAcceptedPlanBytes) {
				return invalid(CodeInvalid, "checks_projection.status")
			}
		}
		sort.Slice(fragments, func(i, j int) bool { return fragments[i].Offset < fragments[j].Offset })
		for i := 1; i < len(fragments); i++ {
			if fragments[i].Offset < fragments[i-1].Offset+fragments[i-1].Length {
				return invalid(CodeInvalid, "checks_projection.overlap")
			}
		}
	}
	encoded, err := canonicalJSON(value)
	if err != nil || len(encoded) > MaximumFindingChecksProjectionBytes {
		return invalid(CodeLimitExceeded, "checks_projection")
	}
	return nil
}

func validFindingFragment(fragment FindingFragment, size int64) bool {
	return fragment.Offset >= 0 && fragment.Length > 0 && fragment.Offset <= size && fragment.Length <= size-fragment.Offset && contentdigest.Valid(fragment.Digest)
}
