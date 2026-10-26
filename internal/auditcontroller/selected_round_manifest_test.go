package auditcontroller

import (
	"context"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
)

type selectedPreparedLookup struct {
	items                []auditdomain.ExecutionItem
	ids                  []string
	maxBytes, maxRelated int
}

func (s *selectedPreparedLookup) GetArtifactLink(context.Context, string, string) (auditstore.ArtifactLink, error) {
	return auditstore.ArtifactLink{}, auditstore.ErrNotFound
}
func (s *selectedPreparedLookup) ReadSelectedPreparedManifest(_ context.Context, _, _ string, ids []string, maxBytes, maxRelated int) ([]auditdomain.ExecutionItem, error) {
	s.ids = append([]string{}, ids...)
	s.maxBytes, s.maxRelated = maxBytes, maxRelated
	return s.items, nil
}

type countedPreparedArtifacts struct {
	fakeBuilderArtifactAccess
	reads int
}

func (a *countedPreparedArtifacts) ReadProjectExact(ctx context.Context, project string, descriptor auditstore.ExactArtifact) (artifacts.Payload, error) {
	a.reads++
	return a.fakeBuilderArtifactAccess.ReadProjectExact(ctx, project, descriptor)
}

func TestPreparedRoundDispatchReadsOnlyCompactRootAndSelectedPrefix(t *testing.T) {
	digest := auditdomain.DigestBytes([]byte("pin"))
	baseline := []byte("exact baseline")
	revision := "r1"
	source := contracts.ExactArtifact{Ref: contracts.ArtifactRef{Namespace: auditdomain.ArtifactNamespace("audit"), Name: "selection", Revision: &revision}, Digest: digest, MediaType: auditdomain.RoundSelectionRootMediaType, SizeBytes: 100}
	root := auditdomain.PreparedRoundRoot{Schema: auditdomain.PreparedRoundRootSchema, AuditID: "audit", RoundID: "round", Generation: "generation", ProfileDigest: digest, BaselineDigest: auditdomain.DigestBytes(baseline), Source: source, PageCount: 4096, ItemCount: 4096, PagesDigest: digest}
	data, err := auditdomain.EncodePreparedRoundRoot(root)
	if err != nil {
		t.Fatal(err)
	}
	access := &countedPreparedArtifacts{fakeBuilderArtifactAccess: fakeBuilderArtifactAccess{payload: artifacts.Payload{MediaType: auditdomain.PreparedRoundRootMediaType, Data: data}}}
	task := builderExact("task", "task-r1")
	ordinal := 0
	proposal := contracts.ExactArtifact{Ref: contracts.ArtifactRef{Namespace: "findings", Name: "proposal", Revision: &revision}, Digest: digest, MediaType: auditdomain.JSONMediaType, SizeBytes: 2}
	retained := proposal.Clone()
	retained.Ref.Namespace = auditdomain.ArtifactNamespace("audit")
	plan := auditdomain.CheckPlan{Schema: contracts.FindingCheckPlanSchema, Method: "applicability", Phase: "applicability", Objective: "Check the retained claim"}
	planBytes, err := auditdomain.EncodeCheckPlan(plan)
	if err != nil {
		t.Fatal(err)
	}
	accepted := &auditdomain.AcceptedPlan{Schema: auditdomain.AcceptedPlanSchema, OwnerID: "owner", ProjectID: "project", AuditID: "audit", ItemKey: "finding-one", BaselineDigest: root.BaselineDigest, ProfileDigest: digest,
		Source: auditdomain.AcceptedPlanSource{Kind: "finding-receipt", ReceiptID: "receipt", ProposedCheckOrdinal: &ordinal, RunID: "producer", Proposal: proposal, RetainedProposal: retained}, Plan: plan, PlanDigest: auditdomain.DigestBytes(planBytes),
		Route:         auditdomain.AcceptedPlanRoute{MappingID: "named", MappingDigest: digest, WorkflowRole: "renamed", BindingDigest: digest, WorkflowDigest: digest, SkillsDigest: digest, RequiredCapabilities: []contracts.CheckCapabilityRequirement{}},
		Authorization: auditdomain.AcceptedPlanAuthorization{ScopeDigest: digest, ActiveChecks: contracts.ActiveChecksProhibited, ApprovalRequirement: auditdomain.ApprovalNone, LimitsDigest: digest}}
	item := auditdomain.ExecutionItem{ItemKey: "finding-one", Ordinal: 2048, SubjectKey: "subject", TaskPackageID: "task-package", TaskPackageDigest: task.Digest, TaskRef: &task.Ref, Inputs: []auditdomain.ExactInput{}, AcceptedPlan: accepted}
	lookup := &selectedPreparedLookup{items: []auditdomain.ExecutionItem{item}}
	b := &PinnedSubmissionBuilder{artifacts: access, outputs: lookup}
	snapshot := auditstore.ReconcileSnapshot{Audit: auditstore.Audit{AuditID: "audit", ProjectID: "project", Profile: auditstore.ProfileIdentity{Digest: digest}, BaselineSnapshot: baseline}, Round: &auditstore.Round{RoundID: "round", Manifest: source, ExpectedItemCount: 4096}}
	selected := []CheckExecutionMember{{Item: auditstore.Item{ItemID: "selected-one", ItemKey: item.ItemKey, Ordinal: item.Ordinal, SubjectKey: item.SubjectKey}}, {Item: auditstore.Item{ItemID: "selected-two", ItemKey: "other", Ordinal: 2049, SubjectKey: "other"}}}
	manifest, paged, err := b.readSelectedRoundExecutionManifest(t.Context(), snapshot, selected)
	if err != nil || !paged || len(manifest.Items) != 1 || manifest.Items[0].Ordinal != 2048 {
		t.Fatalf("selected manifest = %+v %t %v", manifest, paged, err)
	}
	if access.reads != 1 || len(lookup.ids) != 2 || lookup.ids[0] != "selected-one" || lookup.maxBytes != auditdomain.MaximumPreparedRoundPageBytes-len(data) || lookup.maxRelated != auditdomain.MaximumPreparedRoundRelated-1 {
		t.Fatalf("unbounded or unrelated read: reads=%d lookup=%+v", access.reads, lookup)
	}
	lookup.items[0].Ordinal = 0
	if _, _, err := b.readSelectedRoundExecutionManifest(t.Context(), snapshot, selected); err == nil {
		t.Fatal("global item ordinal drift accepted")
	}
}
