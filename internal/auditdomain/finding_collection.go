package auditdomain

import (
	"bytes"
	"sort"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
)

const (
	FindingCollectionSchema                  = "contractor.findings.collection.v2"
	PackageKindFindingCollection PackageKind = "finding-collection"
	FindingCollectionMemberID                = "collection"
	MaximumCollectionEntries                 = 256
	MaximumCollectionSources                 = 64
	MaximumCollectionDocuments               = MaximumMembers - 1
	MaximumCollectionJSONBytes               = 1 << 20
)

// FindingCollection describes a complete, frozen selection, not a verification
// worklist. All referenced bytes travel in the package; source refs are provenance
// and never grant a reader access to a different Run.
type FindingCollection struct {
	Schema     string                      `json:"schema"`
	SnapshotAt string                      `json:"snapshot_at"`
	Sources    []FindingCollectionSource   `json:"sources"`
	Documents  []FindingCollectionDocument `json:"documents"`
	Entries    []FindingCollectionEntry    `json:"entries"`
}

type FindingCollectionSource struct {
	Kind string `json:"kind"`
	ID   string `json:"id"`
}

type FindingCollectionDocument struct {
	ID        string                  `json:"id"`
	Scope     FindingCollectionSource `json:"scope"`
	Ref       contracts.ArtifactRef   `json:"ref"`
	Digest    string                  `json:"digest"`
	MediaType string                  `json:"media_type"`
	SizeBytes int64                   `json:"size_bytes"`
}

type FindingCollectionEntry struct {
	SelectedSources    []FindingCollectionSource     `json:"selected_sources"`
	ReceiptID          string                        `json:"receipt_id"`
	ProposalID         string                        `json:"proposal_id"`
	RunID              string                        `json:"run_id"`
	InvocationID       string                        `json:"invocation_id"`
	AuditOrigin        *FindingCollectionAuditOrigin `json:"audit_origin,omitempty"`
	AuditHolds         []string                      `json:"audit_holds,omitempty"`
	Retention          string                        `json:"retention"`
	ProposalDocumentID string                        `json:"proposal_document_id"`
	Evidence           []FindingCollectionEvidence   `json:"evidence"`
	Finding            *FindingCollectionFinding     `json:"finding,omitempty"`
}

type FindingCollectionAuditOrigin struct {
	HypothesisSnapshotDocumentID string                       `json:"hypothesis_snapshot_document_id,omitempty"`
	AuditID                      string                       `json:"audit_id"`
	ExecutionID                  string                       `json:"execution_id"`
	TaskID                       string                       `json:"task_id"`
	Hypothesis                   *auditcontract.HypothesisRef `json:"hypothesis,omitempty"`
}

type FindingCollectionEvidence struct {
	EvidenceID string `json:"evidence_id"`
	DocumentID string `json:"document_id"`
}

// Finding captures canonical Project state at publication time. Imported bytes
// never authorize changes to that state.
type FindingCollectionFinding struct {
	ProjectID         string  `json:"project_id"`
	FindingID         string  `json:"finding_id"`
	Revision          uint64  `json:"revision"`
	State             *string `json:"state" strictjson:"required,nullable"`
	Verdict           string  `json:"verdict,omitempty"`
	Severity          string  `json:"severity,omitempty"`
	DuplicateTargetID string  `json:"duplicate_target_id,omitempty"`
}

// FindingCollectionDocumentID retains the scope as part of identity: identical
// bindings in different Runs cannot alias, even when their bytes are equal.
func FindingCollectionDocumentID(document FindingCollectionDocument) (string, error) {
	if err := validateCollectionDocument(document); err != nil {
		return "", err
	}
	identity, err := canonicalJSON([]any{
		document.Scope.Kind, document.Scope.ID, document.Ref.Namespace, document.Ref.Name,
		*document.Ref.Revision, document.Digest, document.MediaType, document.SizeBytes,
	})
	if err != nil {
		return "", invalid(CodeInvalid, "collection.document_identity")
	}
	return "doc-" + strings.TrimPrefix(DigestBytes(identity), "sha256:"), nil
}

func EncodeFindingCollection(value FindingCollection) ([]byte, error) {
	data, err := encodeDocument(value, validateFindingCollection)
	if err == nil && len(data) > MaximumCollectionJSONBytes {
		return nil, invalid(CodeLimitExceeded, "collection")
	}
	return data, err
}

func DecodeFindingCollection(data []byte) (FindingCollection, error) {
	if len(data) > MaximumCollectionJSONBytes {
		return FindingCollection{}, invalid(CodeLimitExceeded, "collection")
	}
	value, err := decodeDocument(data, validateFindingCollection)
	if err != nil {
		return FindingCollection{}, err
	}
	canonical, err := EncodeFindingCollection(value)
	if err != nil || !bytes.Equal(data, canonical) {
		return FindingCollection{}, invalid(CodeInvalid, "collection.canonical_json")
	}
	return value, nil
}

func validateCollectionDocument(value FindingCollectionDocument) error {
	if value.Scope.Kind != "run" && value.Scope.Kind != "project" && value.Scope.Kind != "user" ||
		validateIdentifier(value.Scope.ID, "scope.id") != nil || value.Ref.ValidateExact() != nil ||
		!contentdigest.Valid(value.Digest) || !validMediaType(value.MediaType) ||
		value.MediaType != normalizedMediaType(value.MediaType) || value.SizeBytes < 0 || value.SizeBytes > MaximumMemberBytes {
		return invalid(CodeReferenceInvalid, "collection.documents")
	}
	return nil
}

func validateFindingCollection(value FindingCollection) error {
	if value.Schema != FindingCollectionSchema {
		return invalid(CodeSchemaUnsupported, "collection.schema")
	}
	if parsed, err := time.Parse(time.RFC3339Nano, value.SnapshotAt); err != nil || !strings.HasSuffix(value.SnapshotAt, "Z") || parsed.Format(time.RFC3339Nano) != value.SnapshotAt {
		return invalid(CodeInvalid, "collection.snapshot_at")
	}
	if len(value.Sources) == 0 || len(value.Sources) > MaximumCollectionSources ||
		value.Entries == nil || len(value.Entries) > MaximumCollectionEntries ||
		value.Documents == nil || len(value.Documents) > MaximumCollectionDocuments {
		return invalid(CodeLimitExceeded, "collection")
	}
	previous := ""
	sources := make(map[string]bool, len(value.Sources))
	for _, source := range value.Sources {
		key := source.Kind + "\x00" + source.ID
		if source.Kind != "run" && source.Kind != "audit" && source.Kind != "project" || validateIdentifier(source.ID, "source.id") != nil || key <= previous {
			return invalid(CodeInvalid, "collection.sources")
		}
		previous = key
		sources[key] = true
	}
	documents := make(map[string]FindingCollectionDocument, len(value.Documents))
	previous = ""
	var total int64
	for _, document := range value.Documents {
		id, err := FindingCollectionDocumentID(document)
		if err != nil || document.ID != id || document.ID <= previous {
			return invalid(CodeReferenceInvalid, "collection.documents")
		}
		total += document.SizeBytes
		if total > MaximumArchiveBytes {
			return invalid(CodeLimitExceeded, "collection.document_bytes")
		}
		documents[id] = document
		previous = id
	}
	used := make(map[string]bool, len(documents))
	proposals := make(map[string]bool, len(value.Entries))
	previous = ""
	for _, entry := range value.Entries {
		if validateIdentifier(entry.ReceiptID, "receipt_id") != nil || entry.ReceiptID <= previous ||
			validateIdentifier(entry.ProposalID, "proposal_id") != nil || proposals[entry.ProposalID] ||
			validateIdentifier(entry.RunID, "run_id") != nil || validateIdentifier(entry.InvocationID, "invocation_id") != nil {
			return invalid(CodeInvalid, "collection.entries")
		}
		switch entry.Retention {
		case "source-held", "project-held", "discarded":
		default:
			return invalid(CodeInvalid, "collection.retention")
		}
		if origin := entry.AuditOrigin; origin != nil {
			if !contracts.ValidResourceID(origin.AuditID) || !contracts.ValidResourceID(origin.ExecutionID) || !contracts.ValidResourceID(origin.TaskID) {
				return invalid(CodeInvalid, "collection.audit_origin")
			}
			if origin.HypothesisSnapshotDocumentID != "" {
				document, ok := documents[origin.HypothesisSnapshotDocumentID]
				if !ok || origin.Hypothesis == nil || document.MediaType != JSONMediaType || document.SizeBytes == 0 || document.SizeBytes > 32<<10 {
					return invalid(CodeReferenceInvalid, "collection.hypothesis_snapshot")
				}
				used[document.ID] = true
			}
			if origin.Hypothesis != nil && (!contracts.ValidResourceID(origin.Hypothesis.ID) || origin.Hypothesis.Revision == 0 || origin.Hypothesis.Revision > contracts.MaxExactInteger) {
				return invalid(CodeInvalid, "hypothesis")
			}

		}
		if len(entry.SelectedSources) == 0 || len(entry.SelectedSources) > MaximumCollectionSources {
			return invalid(CodeLimitExceeded, "collection.selected_sources")
		}
		priorSource := ""
		for _, source := range entry.SelectedSources {
			key := source.Kind + "\x00" + source.ID
			if !sources[key] || key <= priorSource {
				return invalid(CodeReferenceInvalid, "collection.selected_sources")
			}
			priorSource = key
		}
		if len(entry.AuditHolds) > MaximumCollectionSources {
			return invalid(CodeLimitExceeded, "collection.audit_holds")
		}
		priorHold := ""
		for _, auditID := range entry.AuditHolds {
			if validateIdentifier(auditID, "audit_id") != nil || auditID <= priorHold {
				return invalid(CodeInvalid, "collection.audit_holds")
			}
			priorHold = auditID
		}
		proposal, ok := documents[entry.ProposalDocumentID]
		if !ok || proposal.MediaType != JSONMediaType || proposal.SizeBytes == 0 || proposal.SizeBytes > MaximumDocumentBytes {
			return invalid(CodeReferenceInvalid, "collection.proposal_document_id")
		}
		used[proposal.ID] = true
		if entry.Evidence == nil || len(entry.Evidence) > MaximumEvidencePerItem {
			return invalid(CodeLimitExceeded, "collection.entry")
		}
		priorEvidence := ""
		for _, evidence := range entry.Evidence {
			if validateIdentifier(evidence.EvidenceID, "evidence_id") != nil || evidence.EvidenceID <= priorEvidence {
				return invalid(CodeInvalid, "collection.evidence")
			}
			if _, ok := documents[evidence.DocumentID]; !ok {
				return invalid(CodeReferenceInvalid, "collection.evidence.document_id")
			}
			used[evidence.DocumentID] = true
			priorEvidence = evidence.EvidenceID
		}
		if finding := entry.Finding; finding != nil {
			if validateIdentifier(finding.ProjectID, "project_id") != nil || validateIdentifier(finding.FindingID, "finding_id") != nil ||
				finding.Revision < 1 || finding.Revision > contracts.MaxExactInteger ||
				finding.DuplicateTargetID != "" && validateIdentifier(finding.DuplicateTargetID, "duplicate_target_id") != nil {
				return invalid(CodeInvalid, "collection.finding")
			}
			if ValidateFindingAssessment(finding.State, finding.Verdict, finding.Severity, finding.DuplicateTargetID) != nil || finding.DuplicateTargetID == finding.FindingID {
				return invalid(CodeInvalid, "collection.finding")
			}

		}
		previous = entry.ReceiptID
		proposals[entry.ProposalID] = true
	}
	if len(used) != len(documents) {
		return invalid(CodeReferenceInvalid, "collection.unreferenced_documents")
	}
	return nil
}

// BuildFindingCollectionPackage embeds the exact documents in the existing
// bounded canonical archive format. Missing bytes are errors, including when a
// selected artifact has expired; an empty successful selection uses empty arrays.
func BuildFindingCollectionPackage(value FindingCollection, contents map[string][]byte) ([]byte, error) {
	metadata, err := EncodeFindingCollection(value)
	if err != nil {
		return nil, err
	}
	if len(contents) != len(value.Documents) {
		return nil, invalid(CodeReferenceInvalid, "collection.contents")
	}
	inputs := []PackageInput{{ID: FindingCollectionMemberID, Path: "collection.json", MediaType: JSONMediaType, Data: metadata}}
	for _, document := range value.Documents {
		body, ok := contents[document.ID]
		if !ok || DigestBytes(body) != document.Digest || int64(len(body)) != document.SizeBytes {
			return nil, invalid(CodeDigestMismatch, "collection.contents")
		}
		inputs = append(inputs, PackageInput{ID: document.ID, Path: "documents/" + document.ID, MediaType: document.MediaType, Data: body})
	}
	payload, _, err := BuildPackage(collectionPackageID(metadata), PackageKindFindingCollection, "", inputs)
	if err != nil {
		return nil, err
	}
	if _, _, err = DecodeFindingCollectionPackage(payload); err != nil {
		return nil, err
	}
	return payload, nil
}

// DecodeFindingCollectionPackage verifies both archive integrity and semantic
// evidence membership. It proves consistency of retained bytes, not receipt
// authenticity, source ownership or the truth of a finding.
func DecodeFindingCollectionPackage(payload []byte) (FindingCollection, *Package, error) {
	fail := func(err error) (FindingCollection, *Package, error) { return FindingCollection{}, nil, err }
	pkg, err := ValidatePackage(payload)
	if err != nil {
		return fail(err)
	}
	if pkg.Manifest.Kind != PackageKindFindingCollection || pkg.Manifest.EntryPoint != "" {
		return fail(invalid(CodePackageInvalid, "collection.kind"))
	}
	member, ok := pkg.MemberByID(FindingCollectionMemberID)
	if !ok || member.metadata.Path != "collection.json" || member.metadata.MediaType != JSONMediaType {
		return fail(invalid(CodePackageInvalid, "collection.member"))
	}
	value, err := DecodeFindingCollection(member.data)
	if err != nil {
		return fail(err)
	}
	canonical, err := EncodeFindingCollection(value)
	if err != nil || pkg.Manifest.PackageID != collectionPackageID(canonical) || len(pkg.members) != len(value.Documents)+1 {
		return fail(invalid(CodePackageInvalid, "collection.manifest"))
	}
	byID := make(map[string]Member, len(value.Documents))
	for _, document := range value.Documents {
		content, ok := pkg.MemberByID(document.ID)
		if !ok || content.metadata.Path != "documents/"+document.ID || content.metadata.MediaType != document.MediaType ||
			content.metadata.Digest != document.Digest || content.metadata.Size != document.SizeBytes {
			return fail(invalid(CodeDigestMismatch, "collection.documents"))
		}
		byID[document.ID] = content
	}
	for _, entry := range value.Entries {
		proposal, err := DecodeFindingProposal(byID[entry.ProposalDocumentID].data)
		if err != nil {
			return fail(err)
		}
		ids := append([]string(nil), proposal.EvidenceIDs...)
		sort.Strings(ids)
		if len(ids) != len(entry.Evidence) {
			return fail(invalid(CodeReferenceInvalid, "collection.proposal_evidence"))
		}
		if origin := entry.AuditOrigin; origin != nil && origin.HypothesisSnapshotDocumentID != "" {
			member, ok := pkg.MemberByID(origin.HypothesisSnapshotDocumentID)
			if !ok {
				return fail(invalid(CodeReferenceInvalid, "collection.hypothesis_snapshot"))
			}
			h, err := auditcontract.DecodeHypothesis(member.Data())
			if err != nil || h.ID != origin.Hypothesis.ID || h.Revision != origin.Hypothesis.Revision {
				return fail(invalid(CodeReferenceInvalid, "collection.hypothesis_snapshot"))
			}
		}
		for i, id := range ids {
			if id != entry.Evidence[i].EvidenceID {
				return fail(invalid(CodeReferenceInvalid, "collection.proposal_evidence"))
			}
		}
	}
	return value, pkg, nil
}

func collectionPackageID(metadata []byte) string {
	return "collection-" + strings.TrimPrefix(DigestBytes(metadata), "sha256:")
}
