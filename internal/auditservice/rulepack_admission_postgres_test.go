package auditservice

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/ownerstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func storedRulepackDraft(t *testing.T, ctx context.Context, service *Service, profile auditprofile.PinnedProgram, projectID, ownerID, auditID string, inputs map[string]contracts.ExactArtifact) auditstore.Audit {
	t.Helper()
	snapshot, err := json.Marshal(profile)
	if err != nil {
		t.Fatal(err)
	}
	selection, err := encodeDraftSelection(DraftSelection{Schema: DraftSelectionSchema, Inputs: inputs, RuntimeLabels: []string{}, Labels: map[string]string{}, RequiredPlacementLabels: []string{}})
	if err != nil {
		t.Fatal(err)
	}
	draft, _, err := ownerstore.New(service.pool, artifactpgtest.NewStorage()).CreateDraft(ctx, auditstore.CreateDraftParams{
		AuditID: auditID, OwnerID: ownerID, ProjectID: projectID,
		Profile:         auditstore.ProfileIdentity{Name: profile.Ref.Name, Version: profile.Ref.Version, Digest: profile.Ref.Digest},
		ProfileSnapshot: snapshot, InputSelection: selection,
		Limits:         auditstore.Limits{MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 1, MaxItemsTotal: 1, MaxSubmittedRuns: 3, MaxItemRunAttempts: 3, MaxEvidenceBytes: 67108864},
		IdempotencyKey: auditID, RequestDigest: serviceTestDigest(auditID),
	})
	if err != nil {
		t.Fatal(err)
	}
	return draft
}

type rulepackReadProbe struct {
	artifacts.Repository
	reads       int
	describeErr error
	readErr     error
}

func (p *rulepackReadProbe) Read(ctx context.Context, scope artifacts.Scope, ref contracts.ArtifactRef) (artifacts.ReadResult, error) {
	p.reads++
	if p.readErr != nil {
		return artifacts.ReadResult{}, p.readErr
	}
	return p.Repository.Read(ctx, scope, ref)
}
func (p *rulepackReadProbe) DescribeExact(ctx context.Context, scope artifacts.Scope, refs []contracts.ArtifactRef) ([]contracts.ExactArtifact, error) {
	if p.describeErr != nil {
		return nil, p.describeErr
	}
	return p.Repository.DescribeExact(ctx, scope, refs)
}
