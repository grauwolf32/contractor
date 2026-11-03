package pgstore

import (
	"encoding/json"

	"github.com/grauwolf32/contractor/internal/auditstore"
)

// encodeReportLinks projects the two report links for jsonb_to_recordset.
func encodeReportLinks(machine, summary auditstore.ArtifactLink) json.RawMessage {
	links := []auditstore.ArtifactLink{machine, summary}
	encodedLinks := make([]artifactLinkJSON, len(links))
	for index, link := range links {
		ref, _ := json.Marshal(link.Artifact.Ref)
		encodedLinks[index] = artifactLinkJSON{
			LogicalKey: link.LogicalKey, ArtifactRef: ref,
			ArtifactDigest: link.Artifact.Digest, MediaType: link.Artifact.MediaType,
			SizeBytes: link.Artifact.SizeBytes, SourceProvenance: link.SourceProvenance,
			DisplayRef: link.DisplayRef,
		}
	}
	payload, _ := json.Marshal(encodedLinks)
	return payload
}
