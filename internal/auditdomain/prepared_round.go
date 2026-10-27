package auditdomain

import (
	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/contracts"
)

const (
	ItemOriginSchema = "contractor.audit.item-origin.v1"

	PreparedRoundRootSchema = "contractor.audit.prepared-round.v1"
	PreparedRoundPageSchema = "contractor.audit.prepared-round-page.v1"

	PreparedRoundRootMediaType = "application/vnd.contractor.audit.prepared-round+json"
	PreparedRoundPageMediaType = "application/vnd.contractor.audit.prepared-round-page+json"

	MaximumPreparedRoundPageItems = 128
	MaximumPreparedRoundRelated   = 512
	MaximumPreparedRoundPageBytes = 1 << 20
	MaximumPreparedRoundRootBytes = 256 << 10
)

// PreparedRoundRoot is a compact immutable manifest. The owning indexed store
// seals the complete published chain before database membership is staged.
// The final bounded staging transaction installs a separate SQL membership seal;
// publication alone grants no membership. Dispatch reads indexed selected entries.
type PreparedRoundRoot struct {
	Schema         string `json:"schema" strictjson:"required"`
	AuditID        string `json:"audit_id" strictjson:"required"`
	RoundID        string `json:"round_id" strictjson:"required"`
	Generation     string `json:"generation" strictjson:"required"`
	ProfileDigest  string `json:"profile_digest" strictjson:"required"`
	BaselineDigest string `json:"baseline_digest" strictjson:"required"`
	PageCount      int    `json:"page_count" strictjson:"required"`
	ItemCount      int    `json:"item_count" strictjson:"required"`
	PagesDigest    string `json:"pages_digest" strictjson:"required"`
}

// PreparedRoundPage contains complete immutable item metadata and a page-local
// execution manifest. PreviousPageDigest is the prior exact page digest, or
// DigestBytes(nil) for the first page. The root pins the final page digest.
type PreparedRoundPage struct {
	Schema             string              `json:"schema" strictjson:"required"`
	AuditID            string              `json:"audit_id" strictjson:"required"`
	RoundID            string              `json:"round_id" strictjson:"required"`
	Generation         string              `json:"generation" strictjson:"required"`
	PageOrdinal        int                 `json:"page_ordinal" strictjson:"required"`
	FirstItemOrdinal   int                 `json:"first_item_ordinal" strictjson:"required"`
	PreviousPageDigest string              `json:"previous_page_digest" strictjson:"required"`
	Items              []PreparedRoundItem `json:"items" strictjson:"required"`
}

type PreparedRoundItem struct {
	ItemID          string                   `json:"item_id" strictjson:"required"`
	ItemKey         string                   `json:"item_key" strictjson:"required"`
	Ordinal         int                      `json:"ordinal" strictjson:"required"`
	Kind            string                   `json:"kind" strictjson:"required"`
	SubjectKey      string                   `json:"subject_key" strictjson:"required"`
	WorkflowRole    string                   `json:"workflow_role" strictjson:"required"`
	Task            contracts.ExactArtifact  `json:"task" strictjson:"required"`
	InitialState    string                   `json:"initial_state" strictjson:"required"`
	ApprovalKind    string                   `json:"approval_kind" strictjson:"required"`
	ApprovalDigest  string                   `json:"approval_digest" strictjson:"required"`
	Origin          PreparedItemOrigin       `json:"origin" strictjson:"required"`
	Coverage        RoundResultCoverage      `json:"coverage" strictjson:"required"`
	ProposalSources []PreparedProposalSource `json:"proposal_sources" strictjson:"required"`
	Execution       PreparedExecutionItem    `json:"execution" strictjson:"required"`
}

// PreparedExecutionItem is private accepted-Round metadata. Its flattened JSON
// preserves the indexed prepared projection; only ExecutionItem is transported
// to an ordinary child Run. The exact task package owns transported authority.
type PreparedExecutionItem struct {
	ExecutionItem
	AcceptedPlan *AcceptedPlan               `json:"accepted_plan,omitempty"`
	Dependencies *RetainedDependencyManifest `json:"dependencies,omitempty"`
}

// ValidatePreparedExecutionItem retains the complete private projection gates
// independently of the compact execution manifest codec. The stored page-local
// ordinal and the selected global ordinal are checked by their owning readers.
func ValidatePreparedExecutionItem(item PreparedExecutionItem) error {
	if item.Ordinal < 0 || item.Ordinal >= MaximumItems {
		return invalid(CodeInvalid, "prepared_execution.ordinal")
	}
	compact := item.ExecutionItem
	compact.Ordinal = 0
	if err := ValidateDispatchExecutionManifest(ExecutionManifest{Schema: ExecutionManifestSchema, Items: []ExecutionItem{compact}}); err != nil {
		return err
	}
	if item.AcceptedPlan == nil || ValidateAcceptedPlan(*item.AcceptedPlan, item.Dependencies) != nil || item.AcceptedPlan.ItemKey != item.ItemKey {
		return invalid(CodeInvalid, "prepared_execution.accepted_plan")
	}
	if item.Dependencies != nil {
		return validateItemDependencies(item.ItemKey, item.Dependencies)
	}
	return nil
}

type PreparedItemOrigin struct {
	Schema                   string                `json:"schema" strictjson:"required"`
	SourceRef                contracts.ArtifactRef `json:"source_ref" strictjson:"required"`
	SourceContentDigest      string                `json:"source_content_digest" strictjson:"required"`
	SourceMediaType          string                `json:"source_media_type" strictjson:"required"`
	CanonicalInventoryDigest string                `json:"canonical_inventory_digest" strictjson:"required"`
	EntryKey                 string                `json:"entry_key" strictjson:"required"`
}

type PreparedProposalSource struct {
	ReceiptID            string                  `json:"receipt_id" strictjson:"required"`
	ProposedCheckOrdinal int                     `json:"proposed_check_ordinal" strictjson:"required"`
	Proposal             contracts.ExactArtifact `json:"proposal" strictjson:"required"`
}

func EncodePreparedRoundRoot(value PreparedRoundRoot) ([]byte, error) {
	return encodePreparedDocument(value, validatePreparedRoundRoot, MaximumPreparedRoundRootBytes)
}

func DecodePreparedRoundRoot(data []byte) (PreparedRoundRoot, error) {
	return decodePreparedDocument(data, validatePreparedRoundRoot, MaximumPreparedRoundRootBytes)
}

func EncodePreparedRoundPage(value PreparedRoundPage) ([]byte, error) {
	return encodePreparedDocument(value, validatePreparedRoundPage, MaximumPreparedRoundPageBytes)
}

func DecodePreparedRoundPage(data []byte) (PreparedRoundPage, error) {
	return decodePreparedDocument(data, validatePreparedRoundPage, MaximumPreparedRoundPageBytes)
}

func validatePreparedRoundRoot(value PreparedRoundRoot) error {
	if value.Schema != PreparedRoundRootSchema {
		return invalid(CodeSchemaUnsupported, "prepared.schema")
	}
	return validatePreparedRootHeader(value.AuditID, value.RoundID, value.Generation, value.ProfileDigest, value.BaselineDigest, value.PageCount, value.ItemCount, value.PagesDigest)
}

func validatePreparedRootHeader(auditID, roundID, generation, profileDigest, baselineDigest string, pages, items int, pagesDigest string) error {
	if !validPreparedIdentity(auditID, roundID, generation) || !contentdigest.Valid(profileDigest) || !contentdigest.Valid(baselineDigest) || !contentdigest.Valid(pagesDigest) {
		return invalid(CodeInvalid, "prepared.header")
	}
	if items < 1 || items > MaximumItems || pages < 1 || pages > items {
		return invalid(CodeLimitExceeded, "prepared.counts")
	}
	return nil
}

func validatePreparedRoundPage(value PreparedRoundPage) error {
	if value.Schema != PreparedRoundPageSchema {
		return invalid(CodeSchemaUnsupported, "prepared_page.schema")
	}
	if !validPreparedIdentity(value.AuditID, value.RoundID, value.Generation) || !contentdigest.Valid(value.PreviousPageDigest) {
		return invalid(CodeInvalid, "prepared_page.header")
	}
	if value.PageOrdinal < 0 || value.PageOrdinal >= MaximumItems || value.FirstItemOrdinal < 0 || value.FirstItemOrdinal > MaximumItems-len(value.Items) || len(value.Items) < 1 || len(value.Items) > MaximumPreparedRoundPageItems {
		return invalid(CodeLimitExceeded, "prepared_page.items")
	}
	if value.PageOrdinal == 0 && (value.FirstItemOrdinal != 0 || value.PreviousPageDigest != DigestBytes(nil)) {
		return invalid(CodeInvalid, "prepared_page.chain")
	}
	manifest := ExecutionManifest{Schema: ExecutionManifestSchema, Items: make([]ExecutionItem, len(value.Items))}
	ids := make(map[string]bool, len(value.Items))
	for index, item := range value.Items {
		if !contracts.ValidResourceID(item.ItemID) || ids[item.ItemID] || item.Ordinal != value.FirstItemOrdinal+index || item.Kind != "finding-verification" ||
			validateIdentifier(item.WorkflowRole, "prepared_page.workflow_role") != nil || item.Execution.Ordinal != index || item.Execution.ItemKey != item.ItemKey || item.Execution.SubjectKey != item.SubjectKey ||
			item.Execution.TaskRef == nil || !item.Execution.TaskRef.SameExact(item.Task.Ref) || item.Execution.TaskPackageDigest != item.Task.Digest || !validPreparedArtifact(item.Task, PackageMediaType) {
			return invalid(CodeInvalid, "prepared_page.item")
		}
		ids[item.ItemID] = true
		if item.ApprovalKind == "none" {
			if item.ApprovalDigest != "" || item.InitialState != "ready" {
				return invalid(CodeInvalid, "prepared_page.approval")
			}
		} else if item.ApprovalKind != "active-check-approval" || item.InitialState != "awaiting_review" || !contentdigest.Valid(item.ApprovalDigest) {
			return invalid(CodeInvalid, "prepared_page.approval")
		}
		origin := item.Origin
		if origin.Schema != ItemOriginSchema || origin.SourceRef.ValidateExact() != nil || origin.EntryKey != item.ItemKey || !contentdigest.Valid(origin.SourceContentDigest) || !contentdigest.Valid(origin.CanonicalInventoryDigest) || !validMediaType(origin.SourceMediaType) {
			return invalid(CodeInvalid, "prepared_page.origin")
		}
		if item.Coverage.Status != CoverageNotTested || item.Coverage.Completed == nil || len(item.Coverage.Completed) != 0 || item.Coverage.Rationale != "" || validateSortedStrings(item.Coverage.Requested, MaximumCoverageValues, "prepared_page.coverage", false) != nil || validateSortedStrings(item.Coverage.Gaps, MaximumCoverageValues, "prepared_page.gaps", false) != nil {
			return invalid(CodeInvalid, "prepared_page.coverage")
		}
		if len(item.ProposalSources) != 1 {
			return invalid(CodeInvalid, "prepared_page.sources")
		}
		source := item.ProposalSources[0]
		if !contracts.ValidResourceID(source.ReceiptID) || source.ProposedCheckOrdinal < 0 || source.ProposedCheckOrdinal >= MaximumCoverageValues || !validPreparedArtifact(source.Proposal, JSONMediaType) {
			return invalid(CodeInvalid, "prepared_page.source")
		}
		if err := ValidatePreparedExecutionItem(item.Execution); err != nil {
			return err
		}
		plan := item.Execution.AcceptedPlan
		if plan == nil || ValidateAcceptedPlan(*plan, item.Execution.Dependencies) != nil || plan.AuditID != value.AuditID || plan.ItemKey != item.ItemKey || plan.ItemKey != FindingCheckItemKey(source.ReceiptID, source.ProposedCheckOrdinal) || plan.Route.WorkflowRole != item.WorkflowRole || plan.Source.ReceiptID != source.ReceiptID || *plan.Source.ProposedCheckOrdinal != source.ProposedCheckOrdinal || plan.Source.RetainedProposal.Verify(source.Proposal) != nil {
			return invalid(CodeInvalid, "prepared_page.accepted_plan")
		}
		if (plan.Authorization.ApprovalRequirement == ApprovalNone) != (item.ApprovalKind == "none") || (plan.Authorization.ApprovalRequirement == ApprovalActiveCheck) != (item.ApprovalKind == "active-check-approval") || len(item.Coverage.Requested) != 1 || item.Coverage.Requested[0] != plan.Plan.Method {
			return invalid(CodeInvalid, "prepared_page.authorization")
		}
		manifest.Items[index] = item.Execution.ExecutionItem
	}
	if PreparedRoundPageDescriptors(value) > MaximumPreparedRoundRelated {
		return invalid(CodeLimitExceeded, "prepared_page.descriptors")
	}
	return ValidateDispatchExecutionManifest(manifest)
}

// PreparedRoundPageDescriptors counts every exact source/task/input occurrence
// and selected capability descriptor represented by one page. Repeated refs
// still consume metadata work even when their retained bytes are charged once.
func PreparedRoundPageDescriptors(value PreparedRoundPage) int {
	count := 0
	for _, item := range value.Items {
		count += 5 + len(item.Execution.Inputs)
		if item.Execution.Dependencies != nil {
			count += 2 * len(item.Execution.Dependencies.Dependencies)
		}
		if item.Execution.AcceptedPlan != nil {
			count += len(item.Execution.AcceptedPlan.Route.RequiredCapabilities)
		}
	}
	return count
}

func validPreparedIdentity(values ...string) bool {
	for _, value := range values {
		if !contracts.ValidResourceID(value) {
			return false
		}
	}
	return true
}

func validPreparedArtifact(value contracts.ExactArtifact, mediaType string) bool {
	return value.Validate() == nil && value.MediaType == mediaType && value.SizeBytes > 0
}

func encodePreparedDocument[T any](value T, validate func(T) error, maximum int) ([]byte, error) {
	data, err := encodeDocument(value, validate)
	if err == nil && len(data) > maximum {
		return nil, invalid(CodeLimitExceeded, "prepared_document")
	}
	return data, err
}

func decodePreparedDocument[T any](data []byte, validate func(T) error, maximum int) (T, error) {
	var zero T
	if len(data) > maximum {
		return zero, invalid(CodeLimitExceeded, "prepared_document")
	}
	return decodeDocument(data, validate)
}
