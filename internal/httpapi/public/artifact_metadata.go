package public

import "github.com/grauwolf32/contractor/internal/artifacts"

// Owner upload availability belongs to the public endpoint projection, not the
// stored Artifact or the private Runtime metadata contract.
type artifactMetadataResponse struct {
	artifacts.Metadata
	CanUploadNewVersion bool `json:"canUploadNewVersion"`
}

func ownerArtifactMetadata(store artifacts.ScopedStore, metadata artifacts.Metadata, uploadsAllowed bool) artifactMetadataResponse {
	return artifactMetadataResponse{Metadata: metadata, CanUploadNewVersion: uploadsAllowed && store.CanReplace(metadata)}
}

func ownerArtifactPage(store artifacts.ScopedStore, metadata []artifacts.Metadata, uploadsAllowed bool) []artifactMetadataResponse {
	items := make([]artifactMetadataResponse, len(metadata))
	for i, item := range metadata {
		items[i] = ownerArtifactMetadata(store, item, uploadsAllowed)
	}
	return items
}
