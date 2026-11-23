package pgstore

import (
	"encoding/json"

	"github.com/grauwolf32/contractor/internal/auditstore"
)

type artifactLinkJSON struct {
	Scope            string          `json:"artifact_scope"`
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
func prepareCollectionWrite(params auditstore.CollectParams) collectionWrite {
	encodedItems := prepareTaskCollection(params)
	encodedLinks, retainedBytes := prepareRetainedLinks(params)
	retainedSnapshot := append([]auditstore.ArtifactLink{}, params.Retained...)
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

func prepareRetainedLinks(params auditstore.CollectParams) (json.RawMessage, int64) {
	links := make([]artifactLinkJSON, len(params.Retained))
	var retainedBytes int64
	retainedArtifacts := make(map[string]struct{}, len(params.Retained))
	for index, link := range params.Retained {
		ref, _ := json.Marshal(link.Artifact.Ref)
		links[index] = artifactLinkJSON{
			Scope: string(link.Scope), LogicalKey: link.LogicalKey, ArtifactRef: ref,
			ArtifactDigest: link.Artifact.Digest, MediaType: link.Artifact.MediaType,
			SizeBytes: link.Artifact.SizeBytes, SourceProvenance: link.SourceProvenance,
			DisplayRef: link.DisplayRef,
		}
		key := retainedArtifactKey(link)
		if _, counted := retainedArtifacts[key]; !counted {
			retainedArtifacts[key] = struct{}{}
			retainedBytes += link.Artifact.SizeBytes
		}
	}
	encoded, _ := json.Marshal(links)
	return encoded, retainedBytes
}

// Artifact revisions are identified within their owning scope.
func retainedArtifactKey(link auditstore.ArtifactLink) string {
	return string(link.Scope) + "\x00" + link.Artifact.Ref.Namespace + "\x00" + link.Artifact.Ref.Name + "\x00" + *link.Artifact.Ref.Revision
}
