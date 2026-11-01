package auditdomain

import (
	"encoding/json"
	"fmt"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

const (
	RoundResultsSchema                  = "contractor.audit.round-results.v1"
	RoundResultsPageSchema              = "contractor.audit.round-results-page.v1"
	RoundResultsConsumerSchema          = "contractor.audit.round-results-consumer.v1"
	MaximumRoundResultsPages            = 256
	MaximumRoundResultsDocumentBytes    = 256 << 10
	MaximumRoundResultsBytes            = 64 << 20
	MaximumRoundResultsPageItems        = 128
	MaximumRoundResultsSliceDescriptors = 512
	MaximumRoundResultsSliceBytes       = 1 << 20
)

// RoundResultsManifest describes the complete immutable source snapshot. Page
// references are authoritative only after this complete root is accepted by
// the owning Round. SourceGeneration pins the coherent item/review capture;
// later human decisions never rewrite it.
type RoundResultsManifest struct {
	Schema           string                `json:"schema" strictjson:"required"`
	AuditID          string                `json:"audit_id" strictjson:"required"`
	RoundID          string                `json:"round_id" strictjson:"required"`
	SourceGeneration uint64                `json:"source_generation" strictjson:"required"`
	ItemCount        int                   `json:"item_count" strictjson:"required"`
	MetadataBytes    int64                 `json:"metadata_bytes" strictjson:"required"`
	Pages            []RoundResultsPageRef `json:"pages" strictjson:"required"`
}

type RoundResultsPageRef struct {
	Ordinal   int                     `json:"ordinal" strictjson:"required"`
	ItemCount int                     `json:"item_count" strictjson:"required"`
	Artifact  contracts.ExactArtifact `json:"artifact" strictjson:"required"`
}

// RoundResultsPage contains trusted descriptors, never an inferred semantic
// verdict. Explicit final dispositions preserve absent and technical results.
// Result and attachment refs name retained Audit copies. They grant no access
// until the Audit materializer projects them into the consumer Run.
type RoundResultsPage struct {
	Schema           string                `json:"schema" strictjson:"required"`
	AuditID          string                `json:"audit_id" strictjson:"required"`
	RoundID          string                `json:"round_id" strictjson:"required"`
	SourceGeneration uint64                `json:"source_generation" strictjson:"required"`
	Ordinal          int                   `json:"ordinal" strictjson:"required"`
	Items            []RoundResultItem     `json:"items" strictjson:"required"`
	Mappings         []RoundResultsMapping `json:"mappings,omitempty"`
}

type RoundResultItem struct {
	ItemID           string                   `json:"item_id" strictjson:"required"`
	ItemKey          string                   `json:"item_key" strictjson:"required"`
	Ordinal          int                      `json:"ordinal" strictjson:"required"`
	Kind             string                   `json:"kind" strictjson:"required"`
	SubjectKey       string                   `json:"subject_key" strictjson:"required"`
	FinalDisposition string                   `json:"final_disposition" strictjson:"required"`
	Coverage         RoundResultCoverage      `json:"coverage" strictjson:"required"`
	SelectedAttempt  *RoundResultAttempt      `json:"selected_attempt,omitempty"`
	Result           *contracts.ExactArtifact `json:"result,omitempty"`
	Attachments      []RoundResultAttachment  `json:"attachments" strictjson:"required"`
	Findings         []RoundResultFinding     `json:"findings" strictjson:"required"`
}

type RoundResultCoverage struct {
	Status    string   `json:"status" strictjson:"required"`
	Requested []string `json:"requested" strictjson:"required"`
	Completed []string `json:"completed" strictjson:"required"`
	Gaps      []string `json:"gaps" strictjson:"required"`
	Rationale string   `json:"rationale,omitempty"`
}

type RoundResultAttempt struct {
	ExecutionID           string                   `json:"execution_id" strictjson:"required"`
	ExecutionItemID       string                   `json:"execution_item_id" strictjson:"required"`
	Attempt               int                      `json:"attempt" strictjson:"required"`
	ReceiptID             string                   `json:"receipt_id" strictjson:"required"`
	CollectionDisposition string                   `json:"collection_disposition" strictjson:"required"`
	TerminalOutcome       string                   `json:"terminal_outcome" strictjson:"required"`
	RunID                 *string                  `json:"run_id,omitempty"`
	SourceResult          *contracts.ExactArtifact `json:"source_result,omitempty"`
}

// RoundResultAttachment retains a typed accepted collection descriptor without
// interpreting its schema. Classification is conservative provenance, not a
// capability or a claim that a scanner observation proves a finding.
type RoundResultAttachment struct {
	LogicalKey      string                   `json:"logical_key" strictjson:"required"`
	Classification  string                   `json:"classification" strictjson:"required"`
	Kind            string                   `json:"kind,omitempty"`
	EvidenceID      string                   `json:"evidence_id,omitempty"`
	ContentMemberID string                   `json:"content_member_id,omitempty"`
	DisplayRef      string                   `json:"display_ref,omitempty"`
	Original        *contracts.ExactArtifact `json:"original,omitempty"`
	Artifact        contracts.ExactArtifact  `json:"artifact" strictjson:"required"`
}

type RoundResultFinding struct {
	FindingID string  `json:"finding_id" strictjson:"required"`
	ProjectID string  `json:"project_id" strictjson:"required"`
	Revision  uint64  `json:"revision" strictjson:"required"`
	State     *string `json:"state" strictjson:"required"`
	Verdict   *string `json:"verdict" strictjson:"required"`
}

// RoundResultsConsumer is a derived Run-local descriptor. The accepted source
// root remains unchanged. Every Page.Artifact and required mapping Consumer
// resolves in the child Run; Source refs record provenance only.
type RoundResultsConsumer struct {
	Schema           string                  `json:"schema" strictjson:"required"`
	Source           contracts.ExactArtifact `json:"source" strictjson:"required"`
	AuditID          string                  `json:"audit_id" strictjson:"required"`
	RoundID          string                  `json:"round_id" strictjson:"required"`
	SourceGeneration uint64                  `json:"source_generation" strictjson:"required"`
	ItemCount        int                     `json:"item_count" strictjson:"required"`
	Pages            []RoundResultsPageRef   `json:"pages" strictjson:"required"`
	Mappings         []RoundResultsMapping   `json:"mappings" strictjson:"required"`
}

type RoundResultsMapping struct {
	Source   contracts.ExactArtifact `json:"source" strictjson:"required"`
	Consumer contracts.ExactArtifact `json:"consumer" strictjson:"required"`
}

func EncodeRoundResultsManifest(value RoundResultsManifest) ([]byte, error) {
	if err := validateRoundResultsManifest(value); err != nil {
		return nil, err
	}
	return encodeRoundResults(value)
}

func DecodeRoundResultsManifest(data []byte) (RoundResultsManifest, error) {
	var value RoundResultsManifest
	if len(data) > MaximumRoundResultsDocumentBytes || strictjson.Decode(data, &value) != nil {
		return value, invalid(CodeInvalid, "round_results")
	}
	return value, validateRoundResultsManifest(value)
}

func validateRoundResultsManifest(value RoundResultsManifest) error {
	if value.Schema != RoundResultsSchema || value.AuditID == "" || value.RoundID == "" ||
		value.SourceGeneration == 0 || value.ItemCount < 1 || value.ItemCount > MaximumItems ||
		value.MetadataBytes < 0 || value.MetadataBytes > MaximumRoundResultsBytes ||
		len(value.Pages) < 1 || len(value.Pages) > MaximumRoundResultsPages {
		return invalid(CodeInvalid, "round_results")
	}
	count, bytes := 0, int64(0)
	for index, page := range value.Pages {
		if page.Ordinal != index || page.ItemCount < 1 || page.ItemCount > MaximumRoundResultsPageItems ||
			!validRoundResultsExact(page.Artifact) || page.Artifact.MediaType != "application/json" ||
			page.Artifact.SizeBytes > MaximumRoundResultsDocumentBytes {
			return invalid(CodeInvalid, "round_results.pages")
		}
		count += page.ItemCount
		bytes += page.Artifact.SizeBytes
	}
	if count != value.ItemCount || bytes != value.MetadataBytes {
		return invalid(CodeInvalid, "round_results.pages")
	}
	return nil
}

func EncodeRoundResultsPage(value RoundResultsPage) ([]byte, error) {
	if err := ValidateRoundResultsPage(value); err != nil {
		return nil, err
	}
	return encodeRoundResults(value)
}

func DecodeRoundResultsPage(data []byte) (RoundResultsPage, error) {
	var value RoundResultsPage
	if len(data) > MaximumRoundResultsDocumentBytes || strictjson.Decode(data, &value) != nil {
		return value, invalid(CodeInvalid, "round_results_page")
	}
	return value, ValidateRoundResultsPage(value)
}

func ValidateRoundResultsPage(value RoundResultsPage) error {
	if value.Schema != RoundResultsPageSchema || value.AuditID == "" || value.RoundID == "" ||
		value.SourceGeneration == 0 || value.Ordinal < 0 || value.Ordinal >= MaximumRoundResultsPages ||
		len(value.Items) < 1 || len(value.Items) > MaximumRoundResultsPageItems {
		return invalid(CodeInvalid, "round_results_page")
	}
	previous, descriptors := -1, 0
	seen := make(map[string]bool, len(value.Items))
	for _, item := range value.Items {
		if item.ItemID == "" || item.ItemKey == "" || item.Kind == "" || item.SubjectKey == "" ||
			item.Ordinal <= previous || seen[item.ItemID] || item.Attachments == nil || item.Findings == nil ||
			item.Coverage.Requested == nil || item.Coverage.Completed == nil || item.Coverage.Gaps == nil {
			return invalid(CodeInvalid, "round_results_page.items")
		}
		if _, ok := CoverageStatusBucket(item.Coverage.Status); !ok {
			return invalid(CodeInvalid, "round_results_page.coverage")
		}
		switch item.FinalDisposition {
		case "accepted-result", "missing-output", "invalid-result", "execution-failed", "execution-cancelled", "excluded", "not-applicable":
		default:
			return invalid(CodeInvalid, "round_results_page.disposition")
		}
		if (item.FinalDisposition == "accepted-result") != (item.Result != nil) || item.Result != nil && item.SelectedAttempt == nil {
			return invalid(CodeInvalid, "round_results_page.result")
		}
		if item.Result != nil && !validRoundResultsExact(*item.Result) {
			return invalid(CodeInvalid, "round_results_page.result")
		}
		if attempt := item.SelectedAttempt; attempt != nil {
			if attempt.ExecutionID == "" || attempt.ExecutionItemID == "" || attempt.Attempt < 1 || attempt.ReceiptID == "" ||
				attempt.CollectionDisposition == "" || attempt.TerminalOutcome == "" || attempt.SourceResult != nil && attempt.SourceResult.Validate() != nil {
				return invalid(CodeInvalid, "round_results_page.attempt")
			}
		}
		for _, attachment := range item.Attachments {
			if attachment.LogicalKey == "" || attachment.Classification != "accepted-attachment" || !validRoundResultsExact(attachment.Artifact) || attachment.Original != nil && !validRoundResultsExact(*attachment.Original) {
				return invalid(CodeInvalid, "round_results_page.attachments")
			}
		}
		for _, finding := range item.Findings {
			if finding.FindingID == "" || finding.ProjectID == "" || finding.Revision == 0 {
				return invalid(CodeInvalid, "round_results_page.findings")
			}
		}
		descriptors += 1 + len(item.Attachments) + len(item.Findings)
		if item.SelectedAttempt != nil {
			descriptors++
		}
		if item.Result != nil {
			descriptors++
		}
		previous, seen[item.ItemID] = item.Ordinal, true
	}
	for _, mapping := range value.Mappings {
		if !validRoundResultsExact(mapping.Source) || !validRoundResultsExact(mapping.Consumer) {
			return invalid(CodeInvalid, "round_results_page.mappings")
		}
	}
	if descriptors > MaximumRoundResultsSliceDescriptors {
		return invalid(CodeInvalid, "round_results_page.descriptors")
	}
	return nil
}

func EncodeRoundResultsConsumer(value RoundResultsConsumer) ([]byte, error) {
	if value.Schema != RoundResultsConsumerSchema || !validRoundResultsExact(value.Source) ||
		value.AuditID == "" || value.RoundID == "" || value.SourceGeneration == 0 || value.ItemCount < 1 || value.ItemCount > MaximumItems ||
		len(value.Pages) < 1 || len(value.Pages) > MaximumRoundResultsPages || len(value.Mappings) != len(value.Pages) {
		return nil, invalid(CodeInvalid, "round_results_consumer")
	}
	count := 0
	sources := make(map[string]bool, len(value.Mappings))
	consumers := make(map[string]bool, len(value.Pages))
	for index, page := range value.Pages {
		if page.Ordinal != index || page.ItemCount < 1 || page.ItemCount > MaximumRoundResultsPageItems ||
			!validRoundResultsExact(page.Artifact) || page.Artifact.MediaType != "application/json" ||
			page.Artifact.SizeBytes > MaximumRoundResultsDocumentBytes || consumers[page.Artifact.Ref.Key()] {
			return nil, invalid(CodeInvalid, "round_results_consumer.pages")
		}
		mapping := value.Mappings[index]
		if !validRoundResultsExact(mapping.Source) || !validRoundResultsExact(mapping.Consumer) ||
			sources[mapping.Source.Ref.Key()] || page.Artifact.Verify(mapping.Consumer) != nil {
			return nil, invalid(CodeInvalid, "round_results_consumer.mappings")
		}
		count += page.ItemCount
		sources[mapping.Source.Ref.Key()], consumers[page.Artifact.Ref.Key()] = true, true
	}
	if count != value.ItemCount {
		return nil, invalid(CodeInvalid, "round_results_consumer.item_count")
	}
	return encodeRoundResults(value)
}

func validRoundResultsExact(value contracts.ExactArtifact) bool {
	return value.Validate() == nil && contentdigest.Valid(value.Digest) && value.HasMetadata()
}

func encodeRoundResults(value any) ([]byte, error) {
	data, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	if len(data) > MaximumRoundResultsDocumentBytes {
		return nil, fmt.Errorf("%w: round results document exceeds its byte bound", invalid(CodeLimitExceeded, "round_results"))
	}
	return data, nil
}
