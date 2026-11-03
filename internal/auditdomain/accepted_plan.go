package auditdomain

import (
	"sort"

	artifactpolicy "github.com/grauwolf32/contractor/internal/artifacts/policy"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

const AcceptedPlanSchema = "contractor.audit.accepted-plan.v1"

type AcceptedPlanSource struct {
	Kind                 string                  `json:"kind" strictjson:"required"`
	ReceiptID            string                  `json:"receipt_id" strictjson:"required"`
	ProposedCheckOrdinal *int                    `json:"proposed_check_ordinal" strictjson:"required"`
	RunID                string                  `json:"run_id" strictjson:"required"`
	Proposal             contracts.ExactArtifact `json:"proposal" strictjson:"required"`
	RetainedProposal     contracts.ExactArtifact `json:"retained_proposal" strictjson:"required"`
}

// UnmarshalJSON preserves explicit zero ordinal and exact zero-byte metadata.
func (source *AcceptedPlanSource) UnmarshalJSON(data []byte) error {
	type exact struct {
		Ref       contracts.ArtifactRef `json:"ref" strictjson:"required"`
		Digest    string                `json:"digest" strictjson:"required"`
		MediaType string                `json:"mediaType" strictjson:"required"`
		SizeBytes *int64                `json:"sizeBytes" strictjson:"required"`
	}
	var value struct {
		Kind      string `json:"kind" strictjson:"required"`
		ReceiptID string `json:"receipt_id" strictjson:"required"`
		Ordinal   *int   `json:"proposed_check_ordinal" strictjson:"required"`
		RunID     string `json:"run_id" strictjson:"required"`
		Proposal  exact  `json:"proposal" strictjson:"required"`
		Retained  exact  `json:"retained_proposal" strictjson:"required"`
	}
	if err := strictjson.Decode(data, &value); err != nil {
		return err
	}
	if value.Proposal.SizeBytes == nil || value.Retained.SizeBytes == nil {
		return invalid(CodeInvalid, "accepted_plan.source")
	}
	*source = AcceptedPlanSource{Kind: value.Kind, ReceiptID: value.ReceiptID, ProposedCheckOrdinal: value.Ordinal, RunID: value.RunID, Proposal: contracts.ExactArtifact{Ref: value.Proposal.Ref, Digest: value.Proposal.Digest, MediaType: value.Proposal.MediaType, SizeBytes: *value.Proposal.SizeBytes}, RetainedProposal: contracts.ExactArtifact{Ref: value.Retained.Ref, Digest: value.Retained.Digest, MediaType: value.Retained.MediaType, SizeBytes: *value.Retained.SizeBytes}}
	return nil
}

type AcceptedPlanRoute struct {
	MappingID            string                                 `json:"mapping_id"`
	MappingDigest        string                                 `json:"mapping_digest"`
	WorkflowRole         string                                 `json:"workflow_role"`
	BindingDigest        string                                 `json:"binding_digest"`
	WorkflowDigest       string                                 `json:"workflow_digest"`
	SkillsDigest         string                                 `json:"skills_digest"`
	RequiredCapabilities []contracts.CheckCapabilityRequirement `json:"required_capabilities" strictjson:"required"`
}

type AcceptedPlanAuthorization struct {
	ScopeDigest         string                       `json:"scope_digest"`
	ActiveChecks        contracts.ActiveChecksPolicy `json:"active_checks"`
	ApprovalRequirement ApprovalRequirement          `json:"approval_requirement"`
	LimitsDigest        string                       `json:"limits_digest"`
}

// AcceptedPlan is immutable Server-authored provenance. Validation checks the
// full exact shape and dependency binding; admission still proves authority
// against the retained source, profile, baseline and review ledger.
type AcceptedPlan struct {
	Schema             string                    `json:"schema"`
	OwnerID            string                    `json:"owner_id"`
	ProjectID          string                    `json:"project_id"`
	AuditID            string                    `json:"audit_id"`
	ItemKey            string                    `json:"item_key"`
	BaselineDigest     string                    `json:"baseline_digest"`
	ProfileDigest      string                    `json:"profile_digest"`
	Source             AcceptedPlanSource        `json:"source"`
	Plan               CheckPlan                 `json:"plan"`
	PlanDigest         string                    `json:"plan_digest"`
	Route              AcceptedPlanRoute         `json:"route"`
	Authorization      AcceptedPlanAuthorization `json:"authorization"`
	DependenciesDigest *string                   `json:"dependencies_digest" strictjson:"required,nullable"`
}

func ValidateAcceptedPlan(value AcceptedPlan, deps *RetainedDependencyManifest) error {
	if value.Schema != AcceptedPlanSchema {
		return invalid(CodeSchemaUnsupported, "accepted_plan.schema")
	}
	for _, id := range []string{value.OwnerID, value.ProjectID, value.AuditID} {
		if contracts.ValidateResourceID("accepted_plan.identity", id) != nil {
			return invalid(CodeInvalid, "accepted_plan.identity")
		}
	}
	if validateIdentifier(value.ItemKey, "accepted_plan.item_key") != nil || ValidateCheckPlan(value.Plan) != nil || validateAcceptedPlanSource(value.Source, value.AuditID) != nil {
		return invalid(CodeInvalid, "accepted_plan")
	}
	for _, digest := range []string{value.BaselineDigest, value.ProfileDigest, value.PlanDigest, value.Route.MappingDigest, value.Route.BindingDigest, value.Route.WorkflowDigest, value.Route.SkillsDigest, value.Authorization.ScopeDigest, value.Authorization.LimitsDigest} {
		if !contentdigest.Valid(digest) {
			return invalid(CodeInvalid, "accepted_plan.digest")
		}
	}
	planBytes, err := EncodeCheckPlan(value.Plan)
	if err != nil || DigestBytes(planBytes) != value.PlanDigest {
		return invalid(CodeInvalid, "accepted_plan.plan_digest")
	}
	if validateText(value.Route.MappingID, "accepted_plan.mapping", true) != nil || validateText(value.Route.WorkflowRole, "accepted_plan.role", true) != nil || resolved.ValidateMapKey("accepted_plan.mapping", value.Route.MappingID) != nil || resolved.ValidateMapKey("accepted_plan.role", value.Route.WorkflowRole) != nil || len(value.Route.MappingID) > 128 || len(value.Route.WorkflowRole) > 128 || validatePlanCapabilities(value.Route.RequiredCapabilities) != nil || !value.Authorization.ActiveChecks.Valid() || !validApprovalRequirement(value.Authorization.ApprovalRequirement) {
		return invalid(CodeInvalid, "accepted_plan.route")
	}
	if deps == nil {
		if value.DependenciesDigest != nil {
			return invalid(CodeInvalid, "accepted_plan.dependencies")
		}
	} else {
		encoded, err := EncodeRetainedDependencies(*deps)
		if err != nil || value.DependenciesDigest == nil || *value.DependenciesDigest != DigestBytes(encoded) || deps.ItemKey != value.ItemKey || deps.OwnerID != value.OwnerID || deps.ProjectID != value.ProjectID || deps.AuditID != value.AuditID || deps.BaselineDigest != value.BaselineDigest || deps.ProfileDigest != value.ProfileDigest || deps.WorkflowDigest != value.Route.WorkflowDigest || deps.SkillsDigest != value.Route.SkillsDigest {
			return invalid(CodeInvalid, "accepted_plan.dependencies")
		}
		proposalFound := false
		for _, dependency := range deps.Dependencies {
			if dependency.Name == "proposal" && dependency.Kind == "finding-receipt" && dependency.Origin.ReceiptID == value.Source.ReceiptID && dependency.Origin.RunID == value.Source.RunID && dependency.Origin.ProposedCheckOrdinal != nil && *dependency.Origin.ProposedCheckOrdinal == *value.Source.ProposedCheckOrdinal && value.Source.Proposal.Verify(dependency.Source) == nil && value.Source.RetainedProposal.Verify(dependency.Retained) == nil {
				proposalFound = true
				break
			}
		}
		if !proposalFound {
			return invalid(CodeInvalid, "accepted_plan.source_dependencies")
		}
	}
	encoded, err := canonicalJSON(value)
	if err != nil || len(encoded) > MaximumAcceptedPlanBytes {
		return invalid(CodeLimitExceeded, "accepted_plan")
	}
	return nil
}
func validateAcceptedPlanSource(source AcceptedPlanSource, auditID string) error {
	if source.Kind != "finding-receipt" || source.ProposedCheckOrdinal == nil || *source.ProposedCheckOrdinal < 0 || *source.ProposedCheckOrdinal >= MaximumCoverageValues || contracts.ValidateResourceID("source.receipt", source.ReceiptID) != nil || contracts.ValidateResourceID("source.run", source.RunID) != nil || source.Proposal.Validate() != nil || source.RetainedProposal.Validate() != nil || !source.Proposal.HasMetadata() || !source.RetainedProposal.HasMetadata() || source.Proposal.Digest != source.RetainedProposal.Digest || source.Proposal.MediaType != source.RetainedProposal.MediaType || source.Proposal.SizeBytes != source.RetainedProposal.SizeBytes || source.RetainedProposal.Ref.Namespace != artifactpolicy.FindingMaterialsNamespace {
		return invalid(CodeInvalid, "accepted_plan.source")
	}
	return nil
}
func validatePlanCapabilities(values []contracts.CheckCapabilityRequirement) error {
	if values == nil || len(values) > contracts.MaximumCheckCapabilities {
		return invalid(CodeInvalid, "accepted_plan.capabilities")
	}
	previous := ""
	for _, value := range values {
		key := value.Stage + "\x00" + value.Agent + "\x00" + value.Ref.ToolsetID + "@" + value.Ref.Version
		if key <= previous || resolved.ValidateWorkflowName("stage", value.Stage) != nil || resolved.ValidateWorkflowName("agent", value.Agent) != nil || contracts.ValidateSelector("toolset", value.Ref.ToolsetID+"@"+value.Ref.Version) != nil || len(value.Tools) == 0 || len(value.Tools) > contracts.MaximumCheckCapabilityTools || !sort.StringsAreSorted(value.Tools) {
			return invalid(CodeInvalid, "accepted_plan.capabilities")
		}
		previous = key
		last := ""
		for _, tool := range value.Tools {
			if tool <= last || validateText(tool, "tool", true) != nil || contracts.ValidateOpaqueID("tool", tool) != nil {
				return invalid(CodeInvalid, "accepted_plan.capabilities")
			}
			last = tool
		}
	}
	return nil
}
