//go:build integration

package auditcontroller

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/findingintake"
)

// The probe delegates every operation to the real Project artifact store. Only
// calls are counted; descriptors, bytes, claims and writes are never replaced.
type dispatchPayloadReadProbe struct {
	ArtifactAccess
	reads map[string]int
}

func (p *dispatchPayloadReadProbe) WithClaim(claim auditstore.ControllerClaim) ArtifactAccess {
	return &dispatchPayloadReadProbe{ArtifactAccess: p.ArtifactAccess.WithClaim(claim), reads: p.reads}
}

func (p *dispatchPayloadReadProbe) ReadProjectExact(ctx context.Context, project string, exact auditstore.ExactArtifact) (artifacts.Payload, error) {
	p.reads[exact.Ref.Key()]++
	return p.ArtifactAccess.ReadProjectExact(ctx, project, exact)
}

func payloadCountingBuilder(t *testing.T, h *postgresControllerHarness, claim auditstore.ControllerClaim) (SubmissionBuilder, *dispatchPayloadReadProbe) {
	t.Helper()
	builder := h.builder(t)
	probe := &dispatchPayloadReadProbe{ArtifactAccess: builder.artifacts, reads: map[string]int{}}
	builder.artifacts = probe
	return builder.WithClaim(claim), probe
}

func claimPayloadReadFixture(t *testing.T, ctx context.Context, h *postgresControllerHarness) (auditstore.ControllerClaim, auditstore.ReconcileSnapshot) {
	t.Helper()
	claims, err := h.audits.Claim(ctx, auditstore.ClaimParams{HolderID: "payload-read-probe", Lease: 5 * time.Minute, Limit: 8})
	if err != nil {
		t.Fatal(err)
	}
	for _, claim := range claims {
		if claim.AuditID != h.started.Audit.AuditID {
			continue
		}
		snapshot, err := h.audits.GetReconcileSnapshot(ctx, claim)
		if err != nil || snapshot.Round == nil {
			t.Fatalf("dispatch snapshot: %+v %v", snapshot.Round, err)
		}
		if snapshot.Round.State == auditstore.RoundAccepted {
			round := snapshot.Round
			if _, err := h.audits.TransitionRound(ctx, auditstore.RoundTransitionParams{Claim: claim, RoundID: round.RoundID, ExpectedRevision: round.Revision, ExpectedState: round.State, TargetState: auditstore.RoundExecuting}); err != nil {
				t.Fatal(err)
			}
			snapshot, err = h.audits.GetReconcileSnapshot(ctx, claim)
			if err != nil {
				t.Fatal(err)
			}
		}
		return claim, snapshot
	}
	t.Fatal("dispatch Audit was not claimed")
	return auditstore.ControllerClaim{}, auditstore.ReconcileSnapshot{}
}

func TestPostgresOrdinarySingletonDispatchReadsNoTaskPayload(t *testing.T) {
	ctx := t.Context()
	h := newPostgresControllerHarness(t, ctx, 1)
	claim, snapshot := claimPayloadReadFixture(t, ctx, h)
	if len(snapshot.Items) != 1 {
		t.Fatal("singleton fixture has unexpected membership")
	}
	builder, probe := payloadCountingBuilder(t, h, claim)
	prepared, err := builder.PrepareBatch(ctx, snapshot, []CheckExecutionMember{{Item: snapshot.Items[0], Attempt: 1}})
	if err != nil || len(prepared.Intent.Members) != 1 {
		t.Fatalf("singleton dispatch: %v", err)
	}
	if reads := probe.reads[snapshot.Items[0].Task.Ref.Key()]; reads != 0 {
		t.Fatalf("ordinary singleton task payload reads=%d, want 0", reads)
	}
}

func TestPostgresTypedSingletonDispatchReadsTaskPayloadOnce(t *testing.T) {
	ctx := t.Context()
	h := newPostgresControllerHarnessWithConfig(t, ctx, 0, 1, typedRoutingControllerConfig(t))
	intake, err := findingintake.New(h.pool, artifactpgtest.NewStorage())
	if err != nil {
		t.Fatal(err)
	}
	controller := h.controllerWithCollector(t, intake)
	controller.roundBuilder = h.service
	for range 2 {
		preparationStep(t, ctx, controller)
	}
	executions, err := h.audits.ListExecutions(ctx, h.started.Audit.AuditID)
	if err != nil || len(executions) != 1 || executions[0].RunID == nil || executions[0].RoundID == nil {
		t.Fatalf("seed execution: %+v %v", executions, err)
	}
	producer := executions[0]
	grant := childFindingGrant(t, ctx, h, *producer.RunID)
	submitChildFindingProposal(t, ctx, intake, grant, "dispatch-read-plan", nil,
		auditdomain.ProposedCheck{Objective: "Verify the retained source.", Method: "verify"})
	finishTypedRoutingCheck(t, ctx, h, producer)
	accepted := false
	for range 24 {
		preparationStep(t, ctx, controller)
		current, err := h.audits.Get(ctx, h.started.Audit.OwnerID, h.started.Audit.AuditID)
		if err != nil {
			t.Fatal(err)
		}
		if current.CurrentRoundID != nil && *current.CurrentRoundID != *producer.RoundID {
			accepted = true
			break
		}
	}
	if !accepted {
		t.Fatal("typed admission did not accept the follow-up Round")
	}
	claim, snapshot := claimPayloadReadFixture(t, ctx, h)
	if len(snapshot.Items) != 1 || snapshot.Items[0].Dependencies == nil {
		t.Fatal("typed singleton has no accepted dependency closure")
	}
	builder, probe := payloadCountingBuilder(t, h, claim)
	prepared, err := builder.PrepareBatch(ctx, snapshot, []CheckExecutionMember{{Item: snapshot.Items[0], Attempt: 1}})
	if err != nil || len(prepared.Intent.Members) != 1 {
		t.Fatalf("typed singleton dispatch: %v", err)
	}
	if reads := probe.reads[snapshot.Items[0].Task.Ref.Key()]; reads != 1 {
		t.Fatalf("typed singleton task payload reads=%d, want 1", reads)
	}
}

func TestPostgresReducedDispatchPrefixReadsOnlyFinalTasksOnce(t *testing.T) {
	ctx := t.Context()
	const count = 64
	catalog := loadControllerConfigWithFindings(t, count, count, false, func(files map[string]string) {
		files["audit-profiles/checklist.yaml"] = strings.Replace(files["audit-profiles/checklist.yaml"], "maxEvidenceBytes: 1048576", "maxEvidenceBytes: 67108864", 1)
	})
	h := newPostgresControllerHarnessWithConfig(t, ctx, 0, 1, catalog)
	project, err := h.artifacts.Project(h.started.Audit.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	document := auditdomain.ChecklistDocument{Schema: auditdomain.ChecklistSchema, Items: make([]auditdomain.ChecklistEntry, count)}
	for index := range document.Items {
		document.Items[index] = auditdomain.ChecklistEntry{Key: fmt.Sprintf("large-%03d", index), Version: "1", Statement: strings.Repeat("s", auditdomain.MaximumStringBytes), Applicability: "always", AllowedMethods: []string{"static"}, RequiredEvidence: []string{}, ReviewPolicy: "automatic"}
	}
	data, err := json.Marshal(document)
	if err != nil {
		t.Fatal(err)
	}
	input, err := project.Write(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "large-checklist"}, artifacts.Payload{MediaType: auditdomain.JSONMediaType, Data: data}, nil)
	if err != nil {
		t.Fatal(err)
	}
	// Valid shared scope contributes bytes to every task without expanding the
	// inventory beyond its own document bound. The resulting 64-task set exceeds
	// the actual 16 MiB ZIP limit, so production must choose a shorter prefix.
	scopeValue := strings.Repeat("c", auditdomain.MaximumStringBytes)
	draft, _, err := h.service.CreateDraft(ctx, auditservice.CreateDraftParams{AuditID: "audit-large-dispatch", OwnerID: h.started.Audit.OwnerID, ProjectID: h.started.Audit.ProjectID, Profile: auditservice.ProfileSelector{Name: "test-checklist", Version: "1"}, Inputs: map[string]contracts.ArtifactRef{"checklist": input.Ref}, RuntimeLabels: []string{}, Scope: auditservice.Scope{Objective: scopeValue, Target: scopeValue, AuthorizationScope: scopeValue}, IdempotencyKey: "create-large-dispatch", RequestDigest: postgresDigest("create-large-dispatch")})
	if err != nil {
		t.Fatal(err)
	}
	h.started, err = h.service.Start(ctx, auditservice.StartParams{OwnerID: draft.OwnerID, AuditID: draft.AuditID, ExpectedRevision: draft.Revision, IdempotencyKey: "start-large-dispatch", RequestDigest: postgresDigest("start-large-dispatch")})
	if err != nil {
		t.Fatal(err)
	}
	claim, snapshot := claimPayloadReadFixture(t, ctx, h)
	if len(snapshot.Items) != count {
		t.Fatalf("large fixture contains %d items, want %d", len(snapshot.Items), count)
	}
	selected := make([]CheckExecutionMember, count)
	for index, item := range snapshot.Items {
		selected[index] = CheckExecutionMember{Item: item, Attempt: 1}
	}
	builder, probe := payloadCountingBuilder(t, h, claim)
	prepared, err := builder.PrepareBatch(ctx, snapshot, selected)
	if err != nil {
		t.Fatal(err)
	}
	kept := len(prepared.Intent.Members)
	if kept < 2 || kept >= count {
		t.Fatalf("actual task-set limit kept %d of %d, want a reduced multi-item prefix", kept, count)
	}
	for index, item := range snapshot.Items {
		want := 0
		if index < kept {
			want = 1
			if prepared.Intent.Members[index].ItemID != item.ItemID {
				t.Fatal("dispatch did not preserve the selected prefix")
			}
		}
		if reads := probe.reads[item.Task.Ref.Key()]; reads != want {
			t.Fatalf("item %d task payload reads=%d, want %d (kept=%d)", index, reads, want, kept)
		}
	}
}
