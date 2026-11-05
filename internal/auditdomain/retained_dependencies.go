package auditdomain

import (
	"encoding/json"
	"sort"
	"unicode/utf8"

	artifactpolicy "github.com/grauwolf32/contractor/internal/artifacts/policy"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

const (
	RetainedDependencySchema       = "contractor.audit.retained-dependencies.v1"
	MaximumRetainedDependencies    = 32
	MaximumRetainedDependencyBytes = 256 << 10
)

// RetainedDependencyManifest pins the complete Server-authored input closure
// of one accepted item. Decoding establishes its shape; the owning admission
// service must prove every origin, consumer pin and authorization against the
// accepted Audit records before publishing or materializing it.
type RetainedDependencyManifest struct {
	Schema         string               `json:"schema"`
	OwnerID        string               `json:"owner_id"`
	ProjectID      string               `json:"project_id"`
	AuditID        string               `json:"audit_id"`
	ItemKey        string               `json:"item_key"`
	BaselineDigest string               `json:"baseline_digest"`
	ProfileDigest  string               `json:"profile_digest"`
	WorkflowDigest string               `json:"workflow_digest"`
	SkillsDigest   string               `json:"skills_digest"`
	Dependencies   []RetainedDependency `json:"dependencies"`
}

// RetainedDependency names one exact source and its independent Audit copy.
// Requires enumerates other descriptors necessary to use these bytes; JSON
// references omitted from that explicit graph confer no loading authority.
type RetainedDependency struct {
	Name       string                   `json:"name"`
	Kind       string                   `json:"kind"`
	Projection string                   `json:"projection"`
	Source     contracts.ExactArtifact  `json:"source"`
	Retained   contracts.ExactArtifact  `json:"retained"`
	Origin     RetainedDependencyOrigin `json:"origin"`
	Requires   []string                 `json:"requires"`
}

// UnmarshalJSON requires complete exact metadata, including an explicitly
// present size for empty payloads. The shared ExactArtifact type also serves
// digest-only records elsewhere, so this stricter transport owns that check.
func (dependency *RetainedDependency) UnmarshalJSON(data []byte) error {
	type exact struct {
		Ref       contracts.ArtifactRef `json:"ref" strictjson:"required"`
		Digest    string                `json:"digest" strictjson:"required"`
		MediaType string                `json:"mediaType" strictjson:"required"`
		SizeBytes *int64                `json:"sizeBytes" strictjson:"required"`
	}
	var value struct {
		Name       string                   `json:"name"`
		Kind       string                   `json:"kind"`
		Projection string                   `json:"projection"`
		Source     exact                    `json:"source" strictjson:"required"`
		Retained   exact                    `json:"retained" strictjson:"required"`
		Origin     RetainedDependencyOrigin `json:"origin"`
		Requires   []string                 `json:"requires"`
	}
	if err := strictjson.Decode(data, &value); err != nil {
		return err
	}
	if value.Source.SizeBytes == nil || value.Retained.SizeBytes == nil {
		return invalid(CodeInvalid, "dependencies.descriptor")
	}
	*dependency = RetainedDependency{
		Name: value.Name, Kind: value.Kind, Projection: value.Projection,
		Source: contracts.ExactArtifact{Ref: value.Source.Ref, Digest: value.Source.Digest,
			MediaType: value.Source.MediaType, SizeBytes: *value.Source.SizeBytes},
		Retained: contracts.ExactArtifact{Ref: value.Retained.Ref, Digest: value.Retained.Digest,
			MediaType: value.Retained.MediaType, SizeBytes: *value.Retained.SizeBytes},
		Origin: value.Origin, Requires: value.Requires,
	}
	return nil
}

// RetainedDependencyOrigin is a closed set of source records that production
// admission can verify. Only finding-receipt uses a receipt and check ordinal;
// other origins cannot invent those fields to acquire finding authority.
type RetainedDependencyOrigin struct {
	Kind                 string `json:"kind"`
	BaselineInput        string `json:"baseline_input,omitempty"`
	WorkflowRole         string `json:"workflow_role,omitempty"`
	LogicalOutput        string `json:"logical_output,omitempty"`
	ExecutionID          string `json:"execution_id,omitempty"`
	RoundID              string `json:"round_id,omitempty"`
	ItemID               string `json:"item_id,omitempty"`
	Attempt              int    `json:"attempt,omitempty"`
	RunID                string `json:"run_id,omitempty"`
	ReceiptID            string `json:"receipt_id,omitempty"`
	ProposedCheckOrdinal *int   `json:"proposed_check_ordinal,omitempty"`
}

func (origin *RetainedDependencyOrigin) UnmarshalJSON(data []byte) error {
	type plain RetainedDependencyOrigin
	var decoded plain
	if err := strictjson.Decode(data, &decoded); err != nil {
		return err
	}
	var members map[string]json.RawMessage
	if err := json.Unmarshal(data, &members); err != nil {
		return err
	}
	variants := map[string][]string{
		"baseline":             {"kind", "baseline_input"},
		"prepare-output":       {"kind", "workflow_role", "logical_output", "execution_id", "attempt", "run_id"},
		"accepted-item-result": {"kind", "execution_id", "round_id", "item_id", "attempt", "run_id"},
		"finding-receipt":      {"kind", "run_id", "receipt_id", "proposed_check_ordinal"},
	}
	expected, supported := variants[decoded.Kind]
	if !supported || len(members) != len(expected) {
		return invalid(CodeInvalid, "dependencies.origin")
	}
	for _, name := range expected {
		if _, present := members[name]; !present {
			return invalid(CodeInvalid, "dependencies.origin")
		}
	}
	*origin = RetainedDependencyOrigin(decoded)
	return nil
}

func DecodeRetainedDependencies(data []byte) (RetainedDependencyManifest, error) {
	if len(data) > MaximumRetainedDependencyBytes {
		return RetainedDependencyManifest{}, invalid(CodeLimitExceeded, "dependencies")
	}
	return decodeDocument(data, ValidateRetainedDependencies)
}

func ValidateRetainedDependencies(value RetainedDependencyManifest) error {
	if value.Schema != RetainedDependencySchema {
		return invalid(CodeSchemaUnsupported, "dependencies.schema")
	}
	for _, identity := range []string{value.OwnerID, value.ProjectID, value.AuditID} {
		if contracts.ValidateResourceID("dependencies.identity", identity) != nil {
			return invalid(CodeInvalid, "dependencies.identity")
		}
	}
	if validateText(value.ItemKey, "dependencies.item_key", true) != nil || len(value.ItemKey) > contracts.MaxResourceIDLength {
		return invalid(CodeInvalid, "dependencies.item_key")
	}
	for _, digest := range []string{value.BaselineDigest, value.ProfileDigest, value.WorkflowDigest, value.SkillsDigest} {
		if !contentdigest.Valid(digest) {
			return invalid(CodeInvalid, "dependencies.digest")
		}
	}
	if value.Dependencies == nil || len(value.Dependencies) == 0 || len(value.Dependencies) > MaximumRetainedDependencies {
		return invalid(CodeLimitExceeded, "dependencies")
	}
	names := make(map[string]int, len(value.Dependencies))
	previous := ""
	for index, dependency := range value.Dependencies {
		if contracts.ValidateArtifactName(dependency.Name) != nil || dependency.Name <= previous {
			return invalid(CodeInvalid, "dependencies.name")
		}
		if err := validateRetainedDependency(value.AuditID, dependency); err != nil {
			return err
		}
		names[dependency.Name] = index
		previous = dependency.Name
	}
	if err := validateRetainedDependencyClosure(value.Dependencies, names); err != nil {
		return err
	}
	encoded, err := canonicalJSON(value)
	if err != nil || len(encoded) > MaximumRetainedDependencyBytes {
		return invalid(CodeLimitExceeded, "dependencies")
	}
	return nil
}

func validateRetainedDependency(auditID string, dependency RetainedDependency) error {
	if dependency.Projection != "exact" ||
		dependency.Source.Validate() != nil || !dependency.Source.HasMetadata() ||
		dependency.Retained.Validate() != nil || !dependency.Retained.HasMetadata() ||
		dependency.Source.Digest != dependency.Retained.Digest ||
		dependency.Source.MediaType != dependency.Retained.MediaType ||
		dependency.Source.SizeBytes != dependency.Retained.SizeBytes {
		return invalid(CodeInvalid, "dependencies.descriptor")
	}
	namespace := ArtifactNamespace(auditID)
	if dependency.Kind == "finding-receipt" {
		namespace = artifactpolicy.FindingMaterialsNamespace
	}
	if dependency.Retained.Ref.Namespace != namespace && !(dependency.Kind == "accepted-item-result" && dependency.Retained.Ref.Namespace == artifactpolicy.FindingMaterialsNamespace) {
		return invalid(CodeReferenceInvalid, "dependencies.retained")
	}
	if dependency.Kind != dependency.Origin.Kind {
		return invalid(CodeInvalid, "dependencies.origin")
	}
	return validateRetainedDependencyOrigin(dependency.Origin)
}

func validateRetainedDependencyOrigin(origin RetainedDependencyOrigin) error {
	for _, identity := range []string{origin.ExecutionID, origin.RoundID, origin.ItemID, origin.RunID, origin.ReceiptID} {
		if identity != "" && contracts.ValidateResourceID("dependencies.origin", identity) != nil {
			return invalid(CodeInvalid, "dependencies.origin")
		}
	}
	for _, name := range []string{origin.BaselineInput, origin.WorkflowRole, origin.LogicalOutput} {
		if name != "" && (!utf8.ValidString(name) || len(name) > contracts.MaxNameLength || resolved.ValidateMapKey("dependencies.origin", name) != nil) {
			return invalid(CodeInvalid, "dependencies.origin")
		}
	}
	valid := false
	switch origin.Kind {
	case "baseline":
		valid = origin.BaselineInput != "" && origin.WorkflowRole == "" && origin.LogicalOutput == "" &&
			origin.ExecutionID == "" && origin.RoundID == "" && origin.ItemID == "" && origin.Attempt == 0 &&
			origin.RunID == "" && origin.ReceiptID == "" && origin.ProposedCheckOrdinal == nil
	case "prepare-output":
		valid = origin.BaselineInput == "" && origin.WorkflowRole != "" && origin.LogicalOutput != "" &&
			origin.ExecutionID != "" && origin.RoundID == "" && origin.ItemID == "" &&
			origin.Attempt > 0 && origin.Attempt <= 10 && origin.RunID != "" &&
			origin.ReceiptID == "" && origin.ProposedCheckOrdinal == nil
	case "accepted-item-result":
		valid = origin.BaselineInput == "" && origin.WorkflowRole == "" && origin.LogicalOutput == "" &&
			origin.ExecutionID != "" && origin.RoundID != "" && origin.ItemID != "" &&
			origin.Attempt > 0 && origin.Attempt <= 10 && origin.RunID != "" &&
			origin.ReceiptID == "" && origin.ProposedCheckOrdinal == nil
	case "finding-receipt":
		valid = origin.BaselineInput == "" && origin.WorkflowRole == "" && origin.LogicalOutput == "" &&
			origin.ExecutionID == "" && origin.RoundID == "" && origin.ItemID == "" && origin.Attempt == 0 &&
			origin.RunID != "" && origin.ReceiptID != "" && origin.ProposedCheckOrdinal != nil &&
			*origin.ProposedCheckOrdinal >= 0 && *origin.ProposedCheckOrdinal < MaximumCoverageValues
	}
	if !valid {
		return invalid(CodeInvalid, "dependencies.origin")
	}
	return nil
}

func validateRetainedDependencyClosure(dependencies []RetainedDependency, names map[string]int) error {
	state := make([]uint8, len(dependencies))
	var visit func(int) error
	visit = func(index int) error {
		switch state[index] {
		case 1:
			return invalid(CodeReferenceInvalid, "dependencies.cycle")
		case 2:
			return nil
		}
		state[index] = 1
		requires := dependencies[index].Requires
		if requires == nil || len(requires) >= len(dependencies) || !sort.StringsAreSorted(requires) {
			return invalid(CodeReferenceInvalid, "dependencies.closure")
		}
		previous := ""
		for _, required := range requires {
			next, exists := names[required]
			if !exists || required <= previous || contracts.ValidateArtifactName(required) != nil {
				return invalid(CodeReferenceInvalid, "dependencies.closure")
			}
			previous = required
			if err := visit(next); err != nil {
				return err
			}
		}
		state[index] = 2
		return nil
	}
	for index := range dependencies {
		if err := visit(index); err != nil {
			return err
		}
	}
	return nil
}
