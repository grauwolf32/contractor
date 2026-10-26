package auditdomain

import (
	"slices"

	"github.com/grauwolf32/contractor/internal/contracts"
)

const FindingCheckSourceSchema = "contractor.audit.finding-check-source.v1"

// FindingCheckSourceDocument is the bounded exact source for a typed candidate.
// Its writer binds these fields to the accepted receipt's verified fragments;
// merely decoding this document does not grant admission authority.
type FindingCheckSourceDocument struct {
	Schema      string             `json:"schema"`
	Source      AcceptedPlanSource `json:"source"`
	Check       ProposedCheck      `json:"check"`
	Subject     *FindingSubject    `json:"subject" strictjson:"required,nullable"`
	Limitations []string           `json:"limitations" strictjson:"required"`
}

type FindingCheckOptions struct {
	WorkflowRole        string
	ApprovalRequirement ApprovalRequirement
	ExecutionInputs     []ExactInput
	AcceptedPlan        *AcceptedPlan
	Dependencies        *RetainedDependencyManifest
}
type FindingCheckInventoryOptions struct {
	Common InventoryOptions
	Check  FindingCheckOptions
}

func EncodeFindingCheckSourceDocument(value FindingCheckSourceDocument) ([]byte, error) {
	return encodeDocument(value, validateFindingCheckSourceDocument)
}
func DecodeFindingCheckSourceDocument(data []byte) (FindingCheckSourceDocument, error) {
	return decodeDocument(data, validateFindingCheckSourceDocument)
}
func validateFindingCheckSourceDocument(value FindingCheckSourceDocument) error {
	if value.Schema != FindingCheckSourceSchema || value.Source.Kind != "finding-receipt" || value.Source.ProposedCheckOrdinal == nil || *value.Source.ProposedCheckOrdinal < 0 || *value.Source.ProposedCheckOrdinal >= MaximumCoverageValues || contracts.ValidateResourceID("receipt", value.Source.ReceiptID) != nil || contracts.ValidateResourceID("run", value.Source.RunID) != nil || value.Source.Proposal.Validate() != nil || value.Source.RetainedProposal.Validate() != nil || !value.Source.Proposal.HasMetadata() || !value.Source.RetainedProposal.HasMetadata() || value.Source.Proposal.Digest != value.Source.RetainedProposal.Digest || value.Source.Proposal.MediaType != value.Source.RetainedProposal.MediaType || value.Source.Proposal.SizeBytes != value.Source.RetainedProposal.SizeBytes || validateText(value.Check.Objective, "check.objective", true) != nil || validateIdentifier(value.Check.Method, "check.method") != nil || ValidateFindingLimitations(value.Limitations) != nil {
		return invalid(CodeInvalid, "finding_check_source")
	}
	if _, known := contracts.LookupCheckPlanMethod(contracts.FindingCheckPlanSchema, value.Check.Method); !known {
		return invalid(CodeInvalid, "finding_check_source.method")
	}
	if value.Subject != nil && (validateIdentifier(value.Subject.Kind, "subject.kind") != nil || validateIdentifier(value.Subject.Key, "subject.key") != nil) {
		return invalid(CodeInvalid, "finding_check_source.subject")
	}
	return nil
}

func BuildFindingCheckInventory(source []byte, options FindingCheckInventoryOptions) (Inventory, error) {
	document, err := DecodeFindingCheckSourceDocument(source)
	if err != nil {
		return Inventory{}, err
	}
	binding := options.Check
	if binding.Dependencies != nil {
		encoded, err := EncodeRetainedDependencies(*binding.Dependencies)
		if err != nil {
			return Inventory{}, err
		}
		cloned, err := DecodeRetainedDependencies(encoded)
		if err != nil {
			return Inventory{}, err
		}
		binding.Dependencies = &cloned
	}
	if binding.AcceptedPlan != nil {
		encoded, err := EncodeAcceptedPlan(*binding.AcceptedPlan, binding.Dependencies)
		if err != nil {
			return Inventory{}, err
		}
		cloned, err := DecodeAcceptedPlan(encoded, binding.Dependencies)
		if err != nil {
			return Inventory{}, err
		}
		binding.AcceptedPlan = &cloned
	}
	if binding.AcceptedPlan == nil || ValidateAcceptedPlan(*binding.AcceptedPlan, binding.Dependencies) != nil || binding.WorkflowRole != binding.AcceptedPlan.Route.WorkflowRole || binding.ApprovalRequirement != binding.AcceptedPlan.Authorization.ApprovalRequirement || !sameCanonicalValue(document.Source, binding.AcceptedPlan.Source) || document.Check.Objective != binding.AcceptedPlan.Plan.Objective || document.Check.Method != binding.AcceptedPlan.Plan.Method {
		return Inventory{}, invalid(CodeInvalid, "finding_check_source.binding")
	}
	ordinal := *document.Source.ProposedCheckOrdinal
	proposal := FindingInventoryProposal{ReceiptID: document.Source.ReceiptID, Proposal: FindingInventoryArtifact{Ref: document.Source.RetainedProposal.Ref.Clone(), Digest: document.Source.RetainedProposal.Digest, MediaType: document.Source.RetainedProposal.MediaType, SizeBytes: document.Source.RetainedProposal.SizeBytes}, Document: FindingProposal{Subject: document.Subject, Limitations: slices.Clone(document.Limitations), ProposedChecks: make([]ProposedCheck, ordinal+1)}}
	proposal.Document.ProposedChecks[ordinal] = document.Check
	basis, subject := newFindingChecks(proposal).subject(ordinal)
	if binding.AcceptedPlan.ItemKey != subject.itemKey {
		return Inventory{}, invalid(CodeInvalid, "finding_check_source.item_key")
	}
	subject.acceptedPlan = binding.AcceptedPlan
	subject.dependencies = binding.Dependencies
	basis["accepted_plan"] = binding.AcceptedPlan
	common := options.Common
	common.WorkflowRole = binding.WorkflowRole
	common.ApprovalRequirement = binding.ApprovalRequirement
	return finishInventory(source, JSONMediaType, newFindingInventoryBasis([]map[string]any{basis}), []inventorySubject{subject}, common)
}
