package public

// Artifact write, page and lineage response bodies.

import (
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
)

type artifactWriteResponse struct {
	Artifact  contracts.ArtifactRef `json:"artifact"`
	MediaType string                `json:"mediaType"`
	Size      int64                 `json:"size"`
}

type artifactPageResponse struct {
	Items []artifacts.Metadata `json:"items"`
	Page  pageInfoResponse     `json:"page"`
}

type artifactLineagePageResponse struct {
	Items []artifacts.LineageEdge `json:"items"`
	Page  pageInfoResponse        `json:"page"`
}
