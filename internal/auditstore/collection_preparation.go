package auditstore

import (
	"encoding/json"
)

type collectionItemJSON struct {
	ExecutionItemID     string                   `json:"execution_item_id"`
	Disposition         string                   `json:"disposition"`
	ResultRef           json.RawMessage          `json:"result_ref,omitempty"`
	ResultDigest        *string                  `json:"result_digest,omitempty"`
	Retryable           bool                     `json:"retryable"`
	FinalDisposition    string                   `json:"final_disposition"`
	Status              string                   `json:"status"`
	Requested           []string                 `json:"requested"`
	Completed           []string                 `json:"completed"`
	Gaps                []string                 `json:"gaps"`
	Rationale           string                   `json:"rationale"`
	FindingAssociations []findingAssociationJSON `json:"finding_associations"`
}

type findingAssociationJSON struct {
	AssessmentID       string          `json:"assessment_id"`
	ReceiptID          string          `json:"receipt_id"`
	ProposalRef        json.RawMessage `json:"proposal_ref"`
	ProposalDigest     string          `json:"proposal_digest"`
	ProposalMediaType  string          `json:"proposal_media_type"`
	ProposalSizeBytes  int64           `json:"proposal_size_bytes"`
	SemanticAssessment string          `json:"semantic_assessment"`
}

type artifactLinkJSON struct {
	LogicalKey       string          `json:"logical_key"`
	ArtifactRef      json.RawMessage `json:"artifact_ref"`
	ArtifactDigest   string          `json:"artifact_digest"`
	MediaType        string          `json:"media_type"`
	SizeBytes        int64           `json:"size_bytes"`
	SourceProvenance json.RawMessage `json:"source_provenance"`
	DisplayRef       string          `json:"display_ref"`
}

type collectionWrite struct {
	items, links, retained, sourceRef json.RawMessage
	sourceDigest                      *string
	retainedBytes                     int64
}

// prepareCollectionWrite projects already validated parameters. It runs after
// the replay lookup and performs no reads or durable mutations.
func prepareCollectionWrite(params CollectParams) collectionWrite {
	encodedItems := prepareCollectionItems(params)
	encodedLinks, retainedBytes := prepareRetainedLinks(params)
	retainedSnapshot := append([]ArtifactLink{}, params.Retained...)
	encodedRetained, _ := json.Marshal(retainedSnapshot)
	var sourceRef json.RawMessage
	var sourceDigest *string
	if params.SourceOutput != nil {
		sourceRef, _ = json.Marshal(params.SourceOutput.Ref)
		value := params.SourceOutput.Digest
		sourceDigest = &value
	}
	return collectionWrite{items: encodedItems, links: encodedLinks, retained: encodedRetained, sourceRef: sourceRef, sourceDigest: sourceDigest, retainedBytes: retainedBytes}
}

func prepareCollectionItems(params CollectParams) json.RawMessage {
	items := make([]collectionItemJSON, len(params.Items))
	for index, item := range params.Items {
		var resultRef json.RawMessage
		var resultDigest *string
		if item.Result != nil {
			resultRef, _ = json.Marshal(item.Result.Ref)
			value := item.Result.Digest
			resultDigest = &value
		}
		associations := make([]findingAssociationJSON, len(item.FindingAssociations))
		for associationIndex, association := range item.FindingAssociations {
			proposalRef, _ := json.Marshal(association.Proposal.Ref)
			associations[associationIndex] = findingAssociationJSON{
				AssessmentID: association.AssessmentID, ReceiptID: association.ReceiptID,
				ProposalRef: proposalRef, ProposalDigest: association.Proposal.Digest,
				ProposalMediaType:  association.Proposal.MediaType,
				ProposalSizeBytes:  association.Proposal.SizeBytes,
				SemanticAssessment: association.SemanticAssessment,
			}
		}
		items[index] = collectionItemJSON{
			ExecutionItemID: item.ExecutionItemID, Disposition: string(item.Disposition),
			ResultRef: resultRef, ResultDigest: resultDigest, Retryable: item.Retryable,
			FinalDisposition: string(item.FinalDisposition), Status: string(item.Coverage.Status),
			Requested: nonNilStrings(item.Coverage.Requested), Completed: nonNilStrings(item.Coverage.Completed),
			Gaps: nonNilStrings(item.Coverage.Gaps), Rationale: item.Coverage.Rationale,
			FindingAssociations: associations,
		}
	}
	encoded, _ := json.Marshal(items)
	return encoded
}

func prepareRetainedLinks(params CollectParams) (json.RawMessage, int64) {
	links := make([]artifactLinkJSON, len(params.Retained))
	var retainedBytes int64
	retainedArtifacts := make(map[string]struct{}, len(params.Retained))
	for index, link := range params.Retained {
		ref, _ := json.Marshal(link.Artifact.Ref)
		links[index] = artifactLinkJSON{
			LogicalKey: link.LogicalKey, ArtifactRef: ref,
			ArtifactDigest: link.Artifact.Digest, MediaType: link.Artifact.MediaType,
			SizeBytes: link.Artifact.SizeBytes, SourceProvenance: link.SourceProvenance,
			DisplayRef: link.DisplayRef,
		}
		key := link.Artifact.Ref.Namespace + "\x00" + link.Artifact.Ref.Name + "\x00" + *link.Artifact.Ref.Revision
		if _, counted := retainedArtifacts[key]; !counted {
			retainedArtifacts[key] = struct{}{}
			retainedBytes += link.Artifact.SizeBytes
		}
	}
	encoded, _ := json.Marshal(links)
	return encoded, retainedBytes
}
