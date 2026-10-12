package evalstore

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/artifactstest"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/ownerstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runstore/runstoretest"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
)

// TestPostgresEvalRunOutputAndEvidenceReadTheArtifactStore checks the Run
// output and evidence metadata that Eval reads through the ArtifactStore: the
// current output revision of the owner's Run, an exact evidence revision that
// matches or conflicts, and references that name no revision.
func TestPostgresEvalRunOutputAndEvidenceReadTheArtifactStore(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: "outputs", OwnerID: "owner", Kind: projectstore.KindProject, Name: "Outputs",
		IdempotencyKey: "outputs", RequestDigest: "sha256:" + strings.Repeat("a", 64),
	}); err != nil {
		t.Fatal(err)
	}
	project := "outputs"
	if _, err := runstoretest.CreateRun(ctx, runstore.NewPostgresStore(pool), runstore.CreateRunParams{
		RunID: "run-outputs", OwnerID: "owner", ProjectID: &project, WorkflowName: "copy", WorkflowVersion: "1",
		WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: json.RawMessage(`{}`),
		RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot(),
	}); err != nil {
		t.Fatal(err)
	}
	service := artifacts.NewService(artifacts.NewPostgresRepository(pool))
	run, err := service.Run("run-outputs")
	if err != nil {
		t.Fatal(err)
	}
	payload := []byte("observed output")
	written, err := run.Write(ctx, artifacts.ArtifactRef{Namespace: "scratch", Name: "result"},
		artifacts.Payload{MediaType: "text/plain", Data: payload}, nil)
	if err != nil {
		t.Fatal(err)
	}
	bound, err := artifactstest.BindOutput(ctx, service, "run-outputs", "result", written.Ref)
	if err != nil {
		t.Fatal(err)
	}

	store := NewPostgresStore(pool)
	output, err := store.RunOutput(ctx, "owner", "run-outputs", "result")
	want := evaldomain.Artifact{
		Scope: "run", ScopeID: "run-outputs", Namespace: "outputs", Name: "result",
		Revision: *bound.TargetRef.Revision, SHA256: evaldomain.Digest(payload),
		MediaType: "text/plain", SizeBytes: int64(len(payload)),
	}
	if err != nil || output != want {
		t.Fatalf("Run output = (%+v, %v), want %+v", output, err, want)
	}
	for name, read := range map[string]func() error{
		"another owner's Run": func() error { _, err := store.RunOutput(ctx, "other", "run-outputs", "result"); return err },
		"an unbound output":   func() error { _, err := store.RunOutput(ctx, "owner", "run-outputs", "missing"); return err },
		"an invalid name":     func() error { _, err := store.RunOutput(ctx, "owner", "run-outputs", "bad/name"); return err },
	} {
		if err := read(); !code(err, "eval_not_found") {
			t.Errorf("Run output of %s = %v", name, err)
		}
	}

	if err := store.VerifyEvidence(ctx, want); err != nil {
		t.Fatalf("exact evidence = %v", err)
	}
	conflicting := want
	conflicting.SizeBytes++
	if err := store.VerifyEvidence(ctx, conflicting); !code(err, "eval_member_conflict") {
		t.Fatalf("evidence with another size = %v", err)
	}
	for name, mutate := range map[string]func(*evaldomain.Artifact){
		"an unknown revision": func(ref *evaldomain.Artifact) { ref.Revision = "unknown" },
		"another scope":       func(ref *evaldomain.Artifact) { ref.Scope, ref.ScopeID = "project", "outputs" },
		"an unknown scope":    func(ref *evaldomain.Artifact) { ref.Scope = "audit" },
		"an empty revision":   func(ref *evaldomain.Artifact) { ref.Revision = "" },
	} {
		ref := want
		mutate(&ref)
		if err := store.VerifyEvidence(ctx, ref); !code(err, "eval_evidence_unavailable") {
			t.Errorf("evidence with %s = %v", name, err)
		}
	}
}

// TestSharedPostgresEvalBindsOnlyTheAuditItsMemberCreated checks the Audit binding
// of a member: auditstore resolves the Audit that the member's audit-create
// key created, and BindExecution accepts exactly that Audit in the member's
// workspace, not one that another key created there.
func TestSharedPostgresEvalBindsOnlyTheAuditItsMemberCreated(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()
	scope := setupProject(t, pool, "owner", "eval")
	e := createExperiment(t, pool, scope, "exp", "trace-1", "external-audit")
	m := members(t, pool, e)[0]
	if m.ExecutionKind != "audit" {
		t.Fatal(m.ExecutionKind)
	}
	if _, err := admit(t, pool, e, m.MemberID, nil, "submit"); err != nil {
		t.Fatal(err)
	}
	claim := oneClaim(t, pool)
	workspace := Suboperation{Kind: "project-create", Key: m.SubmissionKey + "-project", Request: json.RawMessage(`{"kind":"project"}`)}
	mustTx(t, pool, func(s *Store) error { return s.PutSuboperation(ctx, scope, e.ID, m.MemberID, claim, workspace) })
	mustTx(t, pool, func(s *Store) error {
		_, _, err := projectstore.NewPostgresStore(s.tx).Create(ctx, projectstore.CreateParams{
			OwnerID: scope.OwnerID, ProjectID: "audit-workspace", Kind: projectstore.KindProject, Name: "Audit member",
			IdempotencyKey: workspace.Key, RequestDigest: evaldomain.Digest(workspace.Request),
		})
		if err != nil {
			return err
		}
		return s.RegisterExecutionProject(ctx, scope, e.ID, m.MemberID, "audit-workspace", workspace.Key, claim)
	})
	mustTx(t, pool, func(s *Store) error {
		return s.ResolveSuboperation(ctx, scope, e.ID, m.MemberID, workspace.Kind, claim, []byte(`{"projectId":"audit-workspace"}`), false)
	})
	create := Suboperation{Kind: "audit-create", Key: m.SubmissionKey, Request: json.RawMessage(`{"audit":"member"}`)}
	mustTx(t, pool, func(s *Store) error { return s.PutSuboperation(ctx, scope, e.ID, m.MemberID, claim, create) })
	audits := ownerstore.New(pool)
	createAudit := func(auditID, key string) {
		t.Helper()
		if _, _, err := audits.CreateDraft(ctx, auditstore.CreateDraftParams{
			AuditID: auditID, OwnerID: scope.OwnerID, ProjectID: "audit-workspace",
			Profile:         auditstore.ProfileIdentity{Name: "profile", Version: "1", Digest: evaldomain.Digest([]byte("profile"))},
			ProfileSnapshot: json.RawMessage(`{"name":"profile"}`), InputSelection: json.RawMessage(`{}`),
			Limits: auditstore.Limits{
				MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 1, MaxItemsTotal: 1,
				MaxSubmittedRuns: 1, MaxItemRunAttempts: 1, MaxEvidenceBytes: 1024,
			},
			IdempotencyKey: key, RequestDigest: evaldomain.Digest(create.Request),
		}); err != nil {
			t.Fatal(err)
		}
	}
	bind := func(auditID string) error {
		return transaction(pool, func(s *Store) error {
			return s.BindExecution(ctx, scope, e.ID, m.MemberID, auditID, claim)
		})
	}
	createAudit("audit-other-key", m.SubmissionKey+"-other")
	if err := bind("audit-other-key"); !code(err, "eval_member_conflict") {
		t.Fatalf("binding of an Audit created under another key = %v", err)
	}
	createAudit("audit-member", m.SubmissionKey)
	for range 2 {
		if err := bind("audit-member"); err != nil {
			t.Fatalf("binding of the member's Audit = %v", err)
		}
	}
	submission, err := NewPostgresStore(pool).Submission(ctx, scope.OwnerID, e.ID, m.MemberID)
	if err != nil || submission.State != "accepted" || submission.ExecutionID == nil ||
		*submission.ExecutionID != "audit-member" {
		t.Fatalf("bound submission = (%+v, %v)", submission, err)
	}
}
