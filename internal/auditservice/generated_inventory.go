package auditservice

import (
	"context"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditbaseline"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

// GeneratedInventoryProjection describes the committed initial worklist without
// exposing its execution manifest or reading unaccepted staging artifacts.
type GeneratedInventoryProjection struct {
	Snapshot                 auditstore.ExactArtifact            `json:"snapshot"`
	Sources                  map[string]auditstore.ExactArtifact `json:"sources"`
	SourceContentDigest      string                              `json:"sourceContentDigest"`
	CanonicalInventoryDigest string                              `json:"canonicalInventoryDigest"`
	Worklist                 auditstore.ExactArtifact            `json:"worklist"`
	ItemCount                int                                 `json:"itemCount"`
	Gaps                     []string                            `json:"gaps"`
}

// generatedInventory projects the committed initial-inventory link of audit
// with its first Round, both read in the commit that audit was read from
// (ASC-R146, ASC-R149). It reads the one exact snapshot the link names, by
// its ref and digest, and checks it against that Round; it never falls back
// to a current binding or reads item pages. The snapshot is immutable, so
// reading it after that commit cannot mix in a later write: a payload that
// fails exact verification is an integrity failure, and one that a purge
// removed since is not found, as the Audit is.
func (s *Service) generatedInventory(
	ctx context.Context, audit auditstore.Audit, link auditstore.ArtifactLink, first auditstore.Round,
) (*GeneratedInventoryProjection, error) {
	project, err := artifacts.NewService(artifacts.NewPostgresRepository(s.pool)).Project(audit.ProjectID)
	if err != nil {
		return nil, err
	}
	read, err := project.ReadExact(ctx, link.Artifact)
	if err != nil {
		return nil, err
	}
	var derived auditbaseline.DerivedInventory
	if read.Payload.MediaType != "application/json" || len(read.Payload.Data) > auditstore.MaxSnapshotBytes || strictjson.Decode(read.Payload.Data, &derived) != nil ||
		derived.Schema != auditbaseline.DerivedInventorySchema || len(derived.Sources) < 1 || len(derived.Sources) > 2 ||
		!contentdigest.Valid(derived.Inventory.SourceContentDigest) || !contentdigest.Valid(derived.Inventory.CanonicalInventoryDigest) ||
		derived.Inventory.Worklist.Validate() != nil || derived.Inventory.Gaps == nil ||
		auditdomain.ValidateDispatchExecutionManifest(derived.Inventory.ExecutionManifest) != nil {
		return nil, artifacts.ErrArtifactIntegrity
	}
	for key, exact := range derived.Sources {
		if (key != "source" && key != "settings") || exact.Validate() != nil || !exact.HasMetadata() {
			return nil, artifacts.ErrArtifactIntegrity
		}
	}
	if source, ok := derived.Sources["source"]; !ok || source.Digest != derived.Inventory.SourceContentDigest {
		return nil, artifacts.ErrArtifactIntegrity
	}
	if !first.Manifest.Ref.SameExact(derived.Inventory.Worklist.Ref) || first.Manifest.Digest != derived.Inventory.Worklist.Digest || first.ExpectedItemCount != len(derived.Inventory.ExecutionManifest.Items) {
		return nil, artifacts.ErrArtifactIntegrity
	}
	return &GeneratedInventoryProjection{Snapshot: link.Artifact, Sources: derived.Sources,
		SourceContentDigest: derived.Inventory.SourceContentDigest, CanonicalInventoryDigest: derived.Inventory.CanonicalInventoryDigest,
		Worklist: derived.Inventory.Worklist, ItemCount: len(derived.Inventory.ExecutionManifest.Items), Gaps: derived.Inventory.Gaps}, nil
}
