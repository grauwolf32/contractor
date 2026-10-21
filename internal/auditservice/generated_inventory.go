package auditservice

import (
	"context"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditbaseline"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/strictjson"
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

// GeneratedInventory reads one accepted link and one exact snapshot, independent
// of item pagination. It deliberately never falls back to a current binding.
func (s *Service) GeneratedInventory(ctx context.Context, ownerID, auditID string, revision uint64) (*GeneratedInventoryProjection, error) {
	audit, err := s.reads().Get(ctx, ownerID, auditID)
	if err != nil {
		return nil, err
	}
	if audit.Revision != revision {
		return nil, auditstore.ErrConflict
	}
	if audit.Phase != auditdomain.AuditPhaseRounds || audit.State == auditstore.AuditDeleting {
		return nil, nil
	}
	link, err := s.reads().GetArtifactLink(ctx, auditID, auditstore.InitialInventoryLogicalKey)
	if err != nil {
		return nil, err
	}
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
	round, err := s.reads().GetRound(ctx, auditID, auditdomain.DeterministicID("round", auditID, "1"))
	if err != nil {
		return nil, err
	}
	if !round.Manifest.Ref.SameExact(derived.Inventory.Worklist.Ref) || round.Manifest.Digest != derived.Inventory.Worklist.Digest || round.ExpectedItemCount != len(derived.Inventory.ExecutionManifest.Items) {
		return nil, artifacts.ErrArtifactIntegrity
	}
	if err := s.checkPageRevision(ctx, ownerID, auditID, revision); err != nil {
		return nil, err
	}
	return &GeneratedInventoryProjection{Snapshot: link.Artifact, Sources: derived.Sources,
		SourceContentDigest: derived.Inventory.SourceContentDigest, CanonicalInventoryDigest: derived.Inventory.CanonicalInventoryDigest,
		Worklist: derived.Inventory.Worklist, ItemCount: len(derived.Inventory.ExecutionManifest.Items), Gaps: derived.Inventory.Gaps}, nil
}
