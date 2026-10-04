package projectlifecycle

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestProjectDeletionCancelsDrainsPurgesAndRetainsSharedResources(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedPool(t, ctx)
	projects := projectstore.NewPostgresStore(pool)
	runs := runstore.NewPostgresStore(pool)
	artifactService := artifacts.NewService(artifacts.NewPostgresRepository(pool))

	project, _, err := projects.Create(ctx, projectstore.CreateParams{
		ProjectID: "project-delete", OwnerID: "user-1", Kind: projectstore.KindProject,
		Name: "Disposable workspace", Description: "lifecycle integration",
		IdempotencyKey: "create-project-delete",
		RequestDigest:  "sha256:" + strings.Repeat("a", 64),
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
INSERT INTO runtime_credentials (
    credential_id, credential_kind, encryption_schema_version, key_id,
    nonce, ciphertext, created_by, created_at
) VALUES (
    'project-origin', 'http-origin-bearer@1', 'contractor.runtime-credentials/v1', $1,
    decode(repeat('00', 12), 'hex'), decode(repeat('00', 17), 'hex'),
    'user-1', clock_timestamp()
)`, "sha256:"+strings.Repeat("b", 64)); err != nil {
		t.Fatal(err)
	}
	project, err = projects.Update(ctx, projectstore.UpdateParams{
		ProjectID: project.ProjectID, OwnerID: project.OwnerID, ExpectedRevision: project.Revision,
		Name: project.Name, Description: project.Description,
		HTTPTarget: &contracts.HTTPOriginTargetRef{
			URL: "https://app.example.test/api",
			Credential: &contracts.RuntimeCredentialRef{
				CredentialID: "project-origin", Kind: contracts.RuntimeCredentialOriginBearer,
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	userStore, err := artifactService.User(project.OwnerID)
	if err != nil {
		t.Fatal(err)
	}
	projectArtifacts, err := artifactService.Project(project.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	sharedPayload := artifacts.Payload{MediaType: "text/plain", Data: []byte("shared content")}
	userSource, err := userStore.Write(ctx, artifacts.ArtifactRef{
		Namespace: "sources", Name: "shared",
	}, sharedPayload, nil)
	if err != nil {
		t.Fatal(err)
	}
	userSkill, err := userStore.Write(ctx, artifacts.ArtifactRef{
		Namespace: "skills", Name: "review",
	}, artifacts.Payload{MediaType: "application/zip", Data: []byte("skill archive")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	projectSource, err := projectArtifacts.Write(ctx, artifacts.ArtifactRef{
		Namespace: "sources", Name: "shared",
	}, sharedPayload, nil)
	if err != nil {
		t.Fatal(err)
	}

	projectID := project.ProjectID
	active := createProjectRun(t, ctx, runs, "run-active", projectID)
	active, err = runs.TransitionRun(
		ctx, active.RunID, runstore.RunInitializing, runstore.RunRunning,
		runstore.Reason{Code: "initialized"},
	)
	if err != nil {
		t.Fatal(err)
	}
	execution, err := runs.CreateStageExecution(ctx, runstore.CreateStageExecutionParams{
		StageExecutionID: "stage-active", RunID: active.RunID, StageName: "build", Attempt: 1,
		StageSpecSchemaVersion:    contracts.APIVersion,
		StageSpecSnapshot:         json.RawMessage(`{"objective":"build"}`),
		StageContextSchemaVersion: contracts.APIVersion,
		StageContext: runstore.StageContextSnapshot{
			Parameters: map[string]string{}, Artifacts: map[string]runstore.PinnedContextArtifact{},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
INSERT INTO stage_allocations (
    allocation_id, stage_execution_id, logical_agent_name, namespace,
    agent_template_ref, worker_runtime_ref, runtime_agent_instance_id
) VALUES ('allocation-active', $1, 'builder', 'builder', '{}'::jsonb, '{}'::jsonb, 'agent-instance')`,
		execution.StageExecutionID,
	); err != nil {
		t.Fatal(err)
	}

	completed := createProjectRun(t, ctx, runs, "run-completed", projectID)
	if _, err := runs.TransitionRun(
		ctx, completed.RunID, runstore.RunInitializing, runstore.RunFailed,
		runstore.Reason{Code: "fixture_failed"},
	); err != nil {
		t.Fatal(err)
	}
	if _, err := runs.UpdateOwnerQueueControl(ctx, runstore.UpdateOwnerQueueControlParams{
		OwnerID: project.OwnerID, ExpectedRevision: 0, Paused: true,
	}); err != nil {
		t.Fatal(err)
	}
	audits := auditstore.NewPostgresStore(pool)
	audit, _, err := audits.CreateDraft(ctx, auditstore.CreateDraftParams{
		AuditID: "audit-project-delete", OwnerID: project.OwnerID, ProjectID: project.ProjectID,
		Profile: auditstore.ProfileIdentity{
			Name: "checklist", Version: "1", Digest: "sha256:" + strings.Repeat("d", 64),
		},
		ProfileSnapshot: json.RawMessage(`{"ref":{"name":"checklist","version":"1"}}`),
		InputSelection:  json.RawMessage(`{}`),
		Limits: auditstore.Limits{
			MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 1, MaxItemsTotal: 1,
			MaxSubmittedRuns: 1, MaxItemRunAttempts: 1, MaxEvidenceBytes: 1024,
		},
		IdempotencyKey: "create-project-audit", RequestDigest: "sha256:" + strings.Repeat("e", 64),
	})
	if err != nil {
		t.Fatal(err)
	}

	deleting, accepted, err := projects.BeginDeletion(ctx, projectstore.BeginDeletionParams{
		ProjectID: project.ProjectID, OwnerID: project.OwnerID, ExpectedRevision: project.Revision,
	})
	if err != nil || !accepted || deleting.Lifecycle != projectstore.LifecycleDeleting {
		t.Fatalf("begin deletion = (%+v, %t, %v)", deleting, accepted, err)
	}
	if _, err := projects.Update(ctx, projectstore.UpdateParams{
		ProjectID: project.ProjectID, OwnerID: project.OwnerID, ExpectedRevision: deleting.Revision,
		Name: "fenced", Description: "fenced",
	}); !errors.Is(err, projectstore.ErrDeleting) {
		t.Fatalf("metadata fence error = %v", err)
	}
	if _, err := projectArtifacts.Write(ctx, artifacts.ArtifactRef{
		Namespace: "sources", Name: "late",
	}, sharedPayload, nil); !errors.Is(err, artifacts.ErrScopeDeleting) {
		t.Fatalf("Artifact fence error = %v", err)
	}
	if _, err := runs.CreateRun(ctx, projectRunParams("run-late", projectID)); !errors.Is(err, runstore.ErrProjectDeleting) {
		t.Fatalf("Run fence error = %v", err)
	}
	if _, _, err := runs.CreateRunIdempotent(ctx, runstore.CreateRunIdempotentParams{
		CreateRunParams: projectRunParams("run-late-idempotent", projectID),
		IdempotencyKey:  "late-project-run",
		RequestDigest:   "sha256:" + strings.Repeat("c", 64),
	}); !errors.Is(err, runstore.ErrProjectDeleting) {
		t.Fatalf("idempotent Run fence error = %v", err)
	}

	notifier := &recordingNotifier{}
	first := newTestController(t, pool, runs, notifier, "first")
	worked, err := first.RunOnce(ctx)
	if err != nil || !worked || len(notifier.runIDs) != 0 {
		t.Fatalf("Audit deletion-intent iteration = (%t, %v), notifications=%v", worked, err, notifier.runIDs)
	}
	deletingAudit, err := audits.Get(ctx, audit.OwnerID, audit.AuditID)
	if err != nil || deletingAudit.State != auditstore.AuditDeleting || deletingAudit.DeletionRequestedAt == nil ||
		deletingAudit.Revision != audit.Revision+1 || deletingAudit.EventSequence != audit.EventSequence+1 {
		t.Fatalf("Project-owned Audit deletion intent = (%+v, %v)", deletingAudit, err)
	}
	deletionEvents, err := audits.ListEvents(ctx, audit.AuditID, audit.EventSequence, 1)
	if err != nil || len(deletionEvents) != 1 || deletionEvents[0].Kind != "audit.delete_requested" ||
		deletionEvents[0].EntityID != audit.AuditID {
		t.Fatalf("Project-owned Audit deletion event = (%+v, %v)", deletionEvents, err)
	}
	var deletionSummary map[string]any
	if err := json.Unmarshal(deletionEvents[0].Summary, &deletionSummary); err != nil ||
		deletionSummary["source"] != "project-deletion" {
		t.Fatalf("Project-owned Audit deletion summary = (%+v, %v)", deletionSummary, err)
	}
	worked, err = first.RunOnce(ctx)
	if err != nil || !worked || len(notifier.runIDs) != 1 || notifier.runIDs[0] != active.RunID {
		t.Fatalf("cancellation iteration = (%t, %v), notifications=%v", worked, err, notifier.runIDs)
	}
	cancelling, err := runs.GetRun(ctx, active.RunID)
	if err != nil || cancelling.State != runstore.RunCancelling {
		t.Fatalf("cancelling Run = (%+v, %v)", cancelling, err)
	}
	if _, err := runs.TransitionRun(
		ctx, active.RunID, runstore.RunCancelling, runstore.RunCancelled,
		runstore.Reason{Code: runstore.CancellationUserRequested},
	); err != nil {
		t.Fatal(err)
	}

	// A fresh controller resumes only from the durable Project phase.
	restarted := newTestController(t, pool, runs, notifier, "restarted")
	worked, err = restarted.RunOnce(ctx)
	if err != nil || !worked {
		t.Fatalf("restart phase advance = (%t, %v)", worked, err)
	}
	current, err := projects.Get(ctx, project.OwnerID, project.ProjectID)
	if err != nil || current.Deletion == nil || current.Deletion.Phase != projectstore.DeletionDraining {
		t.Fatalf("draining Project = (%+v, %v)", current, err)
	}
	worked, err = restarted.RunOnce(ctx)
	if err != nil || worked {
		t.Fatalf("pending allocation drain = (%t, %v)", worked, err)
	}
	if _, err := projects.Get(ctx, project.OwnerID, project.ProjectID); err != nil {
		t.Fatalf("Project disappeared before release: %v", err)
	}
	if err := runs.MarkStageAllocationReleased(ctx, "allocation-active"); err != nil {
		t.Fatal(err)
	}
	expireDeletionClaim(t, ctx, pool, project.ProjectID)
	if worked, err = restarted.RunOnce(ctx); err != nil || worked {
		t.Fatalf("pending Audit drain = (%t, %v)", worked, err)
	}
	claims, err := audits.Claim(ctx, auditstore.ClaimParams{
		HolderID: "project-test-audit-controller", Lease: time.Minute, Limit: 1,
	})
	if err != nil || len(claims) != 1 || claims[0].AuditID != audit.AuditID {
		t.Fatalf("claim deleting Audit = (%+v, %v)", claims, err)
	}
	if err := audits.PurgeClaimed(ctx, claims[0], auditdomain.ArtifactNamespace(audit.AuditID)); err != nil {
		t.Fatalf("purge deleting Audit = %v", err)
	}
	expireDeletionClaim(t, ctx, pool, project.ProjectID)

	for iteration := 0; iteration < 12; iteration++ {
		_, err := restarted.RunOnce(ctx)
		if err != nil {
			t.Fatalf("cleanup iteration %d: %v", iteration, err)
		}
		if _, err := projects.Get(ctx, project.OwnerID, project.ProjectID); errors.Is(err, projectstore.ErrNotFound) {
			break
		} else if err != nil {
			t.Fatal(err)
		}
	}
	if _, err := projects.Get(ctx, project.OwnerID, project.ProjectID); !errors.Is(err, projectstore.ErrNotFound) {
		t.Fatalf("deleted Project lookup error = %v", err)
	}

	for label, query := range map[string]string{
		"Project Runs":      `SELECT count(*) FROM workflow_runs WHERE project_id = 'project-delete'`,
		"Project scope":     `SELECT count(*) FROM artifact_scopes WHERE scope_kind = 'project' AND scope_id = 'project-delete'`,
		"Project revisions": `SELECT count(*) FROM artifact_binding_revisions WHERE scope_kind = 'project' AND scope_id = 'project-delete'`,
	} {
		var count int
		if err := pool.QueryRow(ctx, query).Scan(&count); err != nil || count != 0 {
			t.Fatalf("%s count = %d, error = %v", label, count, err)
		}
	}
	if _, err := userStore.Read(ctx, userSource.Ref); err != nil {
		t.Fatalf("shared User Artifact was removed: %v", err)
	}
	if _, err := userStore.Read(ctx, userSkill.Ref); err != nil {
		t.Fatalf("User Skill was removed: %v", err)
	}
	var credentialCount int
	if err := pool.QueryRow(ctx, `
SELECT count(*) FROM runtime_credentials WHERE credential_id = 'project-origin'`,
	).Scan(&credentialCount); err != nil || credentialCount != 1 {
		t.Fatalf("Runtime credential count = %d, error = %v", credentialCount, err)
	}
	queue, err := runs.GetOwnerQueueControl(ctx, project.OwnerID)
	if err != nil || !queue.Paused {
		t.Fatalf("paused Queue after cleanup = (%+v, %v)", queue, err)
	}
	var projectRevisionCount int
	if err := pool.QueryRow(ctx, `
SELECT count(*)
FROM artifact_binding_revisions
WHERE scope_kind = 'project' AND scope_id = 'project-delete'
  AND revision = $1`, *projectSource.Ref.Revision).Scan(&projectRevisionCount); err != nil || projectRevisionCount != 0 {
		t.Fatalf("Project Artifact revision count = %d, error = %v", projectRevisionCount, err)
	}
}

func TestProjectDeletionWaitsForLockedAuditBeforeDraining(t *testing.T) {
	for _, phase := range []projectstore.DeletionPhase{projectstore.DeletionCancelling, projectstore.DeletionDraining} {
		t.Run(string(phase), func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()
			pool := isolatedPool(t, ctx)
			projects := projectstore.NewPostgresStore(pool)
			project, _, err := projects.Create(ctx, projectstore.CreateParams{
				ProjectID: "project-locked-audit", OwnerID: "user-1", Kind: projectstore.KindProject,
				Name: "Locked Audit", IdempotencyKey: "create-project",
				RequestDigest: "sha256:" + strings.Repeat("a", 64),
			})
			if err != nil {
				t.Fatal(err)
			}
			audits := auditstore.NewPostgresStore(pool)
			audit, _, err := audits.CreateDraft(ctx, auditstore.CreateDraftParams{
				AuditID: "locked-audit", OwnerID: project.OwnerID, ProjectID: project.ProjectID,
				Profile: auditstore.ProfileIdentity{
					Name: "checklist", Version: "1", Digest: "sha256:" + strings.Repeat("b", 64),
				},
				ProfileSnapshot: json.RawMessage(`{"ref":{"name":"checklist","version":"1"}}`),
				InputSelection:  json.RawMessage(`{}`),
				Limits: auditstore.Limits{
					MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 1, MaxItemsTotal: 1,
					MaxSubmittedRuns: 1, MaxItemRunAttempts: 1, MaxEvidenceBytes: 1024,
				},
				IdempotencyKey: "create-audit", RequestDigest: "sha256:" + strings.Repeat("c", 64),
			})
			if err != nil {
				t.Fatal(err)
			}
			if _, _, err := projects.BeginDeletion(ctx, projectstore.BeginDeletionParams{
				ProjectID: project.ProjectID, OwnerID: project.OwnerID, ExpectedRevision: project.Revision,
			}); err != nil {
				t.Fatal(err)
			}
			runs := runstore.NewPostgresStore(pool)
			controller := newTestController(t, pool, runs, &recordingNotifier{}, "locked")
			if phase == projectstore.DeletionDraining {
				// Reproduce the durable state left by an older controller that
				// advanced while the last Audit row was locked.
				claim, err := controller.claim(ctx, "old-controller-claim")
				if err != nil {
					t.Fatal(err)
				}
				if _, _, err := controller.advancePhase(ctx, claim, phase); err != nil {
					t.Fatal(err)
				}
				if err := controller.releaseClaim(ctx, claim); err != nil {
					t.Fatal(err)
				}
			}
			locked, err := pool.Begin(ctx)
			if err != nil {
				t.Fatal(err)
			}
			defer locked.Rollback(context.Background())
			if _, err := locked.Exec(ctx, `SELECT audit_id FROM audits WHERE audit_id = $1 FOR UPDATE`, audit.AuditID); err != nil {
				t.Fatal(err)
			}
			if worked, err := controller.RunOnce(ctx); err != nil || worked {
				t.Fatalf("locked Audit deletion = (%t, %v), want wait without phase advance", worked, err)
			}
			current, err := projects.Get(ctx, project.OwnerID, project.ProjectID)
			if err != nil || current.Deletion == nil || current.Deletion.Phase != phase {
				t.Fatalf("Project advanced past unrequested locked Audit: (%+v, %v)", current, err)
			}
			if err := locked.Rollback(ctx); err != nil {
				t.Fatal(err)
			}
			expireDeletionClaim(t, ctx, pool, project.ProjectID)
			controller = newTestController(t, pool, runs, &recordingNotifier{}, "restarted")
			if worked, err := controller.RunOnce(ctx); err != nil || !worked {
				t.Fatalf("resume Audit deletion = (%t, %v)", worked, err)
			}
			deleting, err := audits.Get(ctx, audit.OwnerID, audit.AuditID)
			if err != nil || deleting.DeletionRequestedAt == nil || deleting.State != auditstore.AuditDeleting {
				t.Fatalf("unlocked Audit was not requested for deletion: (%+v, %v)", deleting, err)
			}
			claims, err := audits.Claim(ctx, auditstore.ClaimParams{
				HolderID: "audit-controller", Lease: time.Minute, Limit: 1,
			})
			if err != nil || len(claims) != 1 {
				t.Fatalf("claim deleting Audit = (%+v, %v)", claims, err)
			}
			if err := audits.PurgeClaimed(ctx, claims[0], auditdomain.ArtifactNamespace(audit.AuditID)); err != nil {
				t.Fatal(err)
			}
			iterations := 4
			if phase == projectstore.DeletionDraining {
				iterations = 3
			}
			for iteration := 0; iteration < iterations; iteration++ {
				if worked, err := controller.RunOnce(ctx); err != nil || !worked {
					t.Fatalf("Project cleanup %d = (%t, %v)", iteration, worked, err)
				}
			}
			if _, err := projects.Get(ctx, project.OwnerID, project.ProjectID); !errors.Is(err, projectstore.ErrNotFound) {
				t.Fatalf("Project did not finish deletion: %v", err)
			}
		})
	}
}

// TestProjectDeletionPurgesAuditWithDecidedReviewsAndAssessments finishes a
// Project whose Audit holds an owner decision and a collected finding
// assessment, rows that outlive the Audit's deleted child Runs.
func TestProjectDeletionPurgesAuditWithDecidedReviewsAndAssessments(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	pool := isolatedPool(t, ctx)
	projects := projectstore.NewPostgresStore(pool)
	project, _, err := projects.Create(ctx, projectstore.CreateParams{
		ProjectID: "project-decided-audit", OwnerID: "user-1", Kind: projectstore.KindProject,
		Name: "Decided Audit", IdempotencyKey: "create-project",
		RequestDigest: "sha256:" + strings.Repeat("a", 64),
	})
	if err != nil {
		t.Fatal(err)
	}
	audits := auditstore.NewPostgresStore(pool)
	audit, _, err := audits.CreateDraft(ctx, auditstore.CreateDraftParams{
		AuditID: "decided-audit", OwnerID: project.OwnerID, ProjectID: project.ProjectID,
		Profile: auditstore.ProfileIdentity{
			Name: "checklist", Version: "1", Digest: "sha256:" + strings.Repeat("b", 64),
		},
		ProfileSnapshot: json.RawMessage(`{"ref":{"name":"checklist","version":"1"}}`),
		InputSelection:  json.RawMessage(`{}`),
		Limits: auditstore.Limits{
			MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 1, MaxItemsTotal: 1,
			MaxSubmittedRuns: 1, MaxItemRunAttempts: 1, MaxEvidenceBytes: 1024,
		},
		IdempotencyKey: "create-audit", RequestDigest: "sha256:" + strings.Repeat("c", 64),
	})
	if err != nil {
		t.Fatal(err)
	}
	seedDecidedAuditHistory(t, ctx, pool, audit)
	if _, _, err := projects.BeginDeletion(ctx, projectstore.BeginDeletionParams{
		ProjectID: project.ProjectID, OwnerID: project.OwnerID, ExpectedRevision: project.Revision,
	}); err != nil {
		t.Fatal(err)
	}
	controller := newTestController(t, pool, runstore.NewPostgresStore(pool), &recordingNotifier{}, "decided")
	if worked, err := controller.RunOnce(ctx); err != nil || !worked {
		t.Fatalf("request Project-owned Audit deletion = (%t, %v)", worked, err)
	}
	claims, err := audits.Claim(ctx, auditstore.ClaimParams{
		HolderID: "audit-controller", Lease: time.Minute, Limit: 1,
	})
	if err != nil || len(claims) != 1 {
		t.Fatalf("claim deleting Audit = (%+v, %v)", claims, err)
	}
	if err := audits.PurgeClaimed(ctx, claims[0], auditdomain.ArtifactNamespace(audit.AuditID)); err != nil {
		t.Fatalf("purge Audit with decided review and assessment: %v", err)
	}
	for iteration := 0; iteration < 4; iteration++ {
		if worked, err := controller.RunOnce(ctx); err != nil || !worked {
			t.Fatalf("Project cleanup %d = (%t, %v)", iteration, worked, err)
		}
	}
	if _, err := projects.Get(ctx, project.OwnerID, project.ProjectID); !errors.Is(err, projectstore.ErrNotFound) {
		t.Fatalf("Project did not finish deletion: %v", err)
	}
	var decisions, assessments int
	if err := pool.QueryRow(ctx, `
SELECT (SELECT count(*) FROM audit_review_decisions),
       (SELECT count(*) FROM audit_finding_assessments)`).Scan(&decisions, &assessments); err != nil ||
		decisions != 0 || assessments != 0 {
		t.Fatalf("Audit decisions/assessments after Project deletion = (%d, %d, %v)", decisions, assessments, err)
	}
}

// seedDecidedAuditHistory records the rows an Audit keeps after its child Run
// is deleted: a collected attempt whose result assessed a retained finding,
// and the owner's decision confirming that finding.
func seedDecidedAuditHistory(t *testing.T, ctx context.Context, pool *pgxpool.Pool, audit auditstore.Audit) {
	t.Helper()
	digest := "sha256:" + strings.Repeat("d", 64)
	sourceRevision := "checks-r1"
	origin, err := json.Marshal(auditstore.ItemOrigin{
		Schema: auditstore.ItemOriginSchema, EntryKey: "check-decided",
		SourceRef:           &contracts.ArtifactRef{Namespace: "inputs", Name: "checks", Revision: &sourceRevision},
		SourceContentDigest: digest, SourceMediaType: "application/json",
		CanonicalInventoryDigest: digest,
	})
	if err != nil {
		t.Fatal(err)
	}
	workflow := &auditstore.WorkflowProvenance{
		Name: "check", Version: "1", SchemaVersion: contracts.APIVersion, ClosureDigest: digest,
	}
	workflow.ConfigurationRef.Name, workflow.ConfigurationRef.Version = workflow.Name, workflow.Version
	runProvenance, err := json.Marshal(auditstore.RunProvenance{
		Schema: "contractor.audit.run-provenance.v1", RunID: "deleted-run-decided", Workflow: workflow,
	})
	if err != nil {
		t.Fatal(err)
	}
	proposalRef := `{"namespace":"audit-finding-proposals","name":"candidate-decided","revision":"proposal-r1"}`
	taskRef := `{"namespace":"audit-task-packages","name":"check-decided","revision":"task-r1"}`
	resultRef := `{"namespace":"audit-results","name":"check-decided","revision":"result-r1"}`
	statements := []struct {
		sql  string
		args []any
	}{
		{`
INSERT INTO finding_proposal_receipts (
    receipt_id, proposal_id, allocation_id, runtime_agent_id,
    runtime_instance_id, stage_execution_id, logical_agent_name,
    invocation_id, submission_id, client_key, request_digest,
    run_id, owner_id, project_id,
    workflow_name, workflow_version, workflow_schema_version,
    workflow_configuration_ref, workflow_closure_digest,
    proposal_ref, proposal_digest, proposal_media_type,
    proposal_size_bytes, evidence
) VALUES ('receipt-decided', 'proposal-decided', 'allocation-decided', 'runtime-decided',
          'instance-decided', 'stage-decided', 'worker', 'invocation-decided',
          'submission-decided', 'candidate-decided', $1, 'deleted-run-decided', $2, $3,
          'finding-source', '1', 'contractor/v1alpha1',
          '{"name":"finding-source","version":"1"}'::jsonb, $1,
          $4::jsonb, $1, 'application/json', 128, '[]'::jsonb)`,
			[]any{digest, audit.OwnerID, audit.ProjectID, proposalRef}},
		{`
INSERT INTO finding_proposal_retention (receipt_id, state, source_run_deleted_at)
VALUES ('receipt-decided', 'audit-held', clock_timestamp())`, nil},
		{`
INSERT INTO finding_proposal_audit_holds (receipt_id, audit_id, project_id, proposal_ref, evidence)
VALUES ('receipt-decided', $1, $2,
        jsonb_build_object('ref', $3::jsonb, 'digest', $4::text,
                           'mediaType', 'application/json', 'sizeBytes', 128),
        '[]'::jsonb)`, []any{audit.AuditID, audit.ProjectID, proposalRef, digest}},
		{`
INSERT INTO audit_rounds (
    round_id, audit_id, ordinal, manifest_ref, manifest_digest, state, expected_item_count
) VALUES ('round-decided', $1, 1,
          '{"namespace":"audit-rounds","name":"round-decided","revision":"round-r1"}'::jsonb,
          $2, 'closed', 1)`, []any{audit.AuditID, digest}},
		{`
INSERT INTO audit_items (
    item_id, audit_id, round_id, item_key, ordinal, kind, subject_key,
    task_ref, task_digest, origin, workflow_role, state
) VALUES ('item-decided', $1, 'round-decided', 'check-decided', 0, 'check',
          'component-decided', $2::jsonb, $3, $4::jsonb, 'check-role', 'ready')`,
			[]any{audit.AuditID, taskRef, digest, origin}},
		{`
INSERT INTO audit_executions (
    execution_id, audit_id, round_id, role, workflow_role, manifest_ref, manifest_digest,
    submission_key, request_digest, run_id, state, terminal_outcome,
    terminal_run_generation, terminal_run_sequence, terminal_observed_at,
    run_provenance, run_deleted_at
) VALUES ('execution-decided', $1, 'round-decided', 'check', 'check-role',
          '{"namespace":"audit-executions","name":"check-decided","revision":"execution-r1"}'::jsonb,
          $2, 'submission-decided', $2, 'deleted-run-decided', 'collected', 'succeeded',
          'generation-one', 1, clock_timestamp(), $3::jsonb, clock_timestamp())`,
			[]any{audit.AuditID, digest, runProvenance}},
		{`
INSERT INTO audit_execution_items (
    execution_item_id, execution_id, audit_id, round_id, item_id,
    batch_ordinal, item_attempt, task_ref, task_digest, input_refs,
    state, collection_disposition, result_ref, result_digest, collected_at
) VALUES ('execution-item-decided', 'execution-decided', $1, 'round-decided', 'item-decided',
          0, 1, $2::jsonb, $3, '[]'::jsonb, 'settled', 'accepted-result',
          $4::jsonb, $3, clock_timestamp())`, []any{audit.AuditID, taskRef, digest, resultRef}},
		{`
INSERT INTO audit_collection_receipts (
    receipt_id, audit_id, execution_id, run_id, terminal_outcome,
    terminal_run_generation, terminal_run_sequence, disposition,
    source_output_ref, source_output_digest, retained_refs, request_digest
) VALUES ('collection-decided', $1, 'execution-decided', 'deleted-run-decided', 'succeeded',
          'generation-one', 1, 'accepted-result', $2::jsonb, $3, '[]'::jsonb, $3)`,
			[]any{audit.AuditID, resultRef, digest}},
		{`
INSERT INTO audit_finding_assessments (
    assessment_id, finding_id, audit_id, receipt_id, item_id,
    execution_item_id, collection_receipt_id, semantic_assessment,
    result_ref, result_digest
)
SELECT 'assessment-decided', finding_id, audit_id, first_receipt_id,
       'item-decided', 'execution-item-decided', 'collection-decided', 'supported',
       $2::jsonb, $3
  FROM audit_findings WHERE audit_id = $1`, []any{audit.AuditID, resultRef, digest}},
		{`
UPDATE audit_items
   SET state = 'settled', final_disposition = 'accepted-result',
       accepted_result_ref = $2::jsonb, accepted_result_digest = $3,
       last_execution_item_id = 'execution-item-decided'
 WHERE audit_id = $1 AND item_id = 'item-decided'`, []any{audit.AuditID, resultRef, digest}},
		{`
INSERT INTO audit_review_requests (
    request_id, audit_id, finding_id, subject_kind, subject_id, kind,
    subject_revision, subject_digest, requested_actions, state,
    idempotency_key, request_digest
)
SELECT 'review-decided', audit_id, finding_id, 'finding', finding_id,
       'finding-triage', 1, $2, '["true_positive","false_positive"]'::jsonb, 'decided',
       'review-decided', $2
  FROM audit_findings WHERE audit_id = $1`, []any{audit.AuditID, digest}},
		{`
INSERT INTO audit_review_decisions (
    decision_id, request_id, audit_id, finding_id, actor_id, action, verdict,
    severity, rationale, subject_revision, subject_digest, idempotency_key, request_digest
)
SELECT 'decision-decided', 'review-decided', audit_id, finding_id, $2,
       'true_positive', 'true_positive', 'high', 'Confirmed before deletion.',
       1, $3, 'decision-decided', $3
  FROM audit_findings WHERE audit_id = $1`, []any{audit.AuditID, audit.OwnerID, digest}},
		{`
UPDATE audit_findings
   SET state = 'confirmed', current_decision_id = 'decision-decided',
       current_assessment_id = 'assessment-decided', revision = revision + 1
 WHERE audit_id = $1`, []any{audit.AuditID}},
	}
	err = persistencepostgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		for _, statement := range statements {
			if _, err := tx.Exec(ctx, statement.sql, statement.args...); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

// delayedPurgeRuns makes one Run purge outlast an ordinary operation budget.
type delayedPurgeRuns struct {
	*runstore.PostgresStore
	delay    time.Duration
	onDelete func(context.Context)
}

func (s *delayedPurgeRuns) DeleteReleasedTerminalRun(ctx context.Context, ownerID, runID string) error {
	if s.onDelete != nil {
		s.onDelete(ctx)
	}
	if s.delay < 0 {
		<-ctx.Done()
		return ctx.Err()
	}
	select {
	case <-time.After(s.delay):
	case <-ctx.Done():
		return ctx.Err()
	}
	return s.PostgresStore.DeleteReleasedTerminalRun(ctx, ownerID, runID)
}

// projectAwaitingRunPurge returns a deleting Project whose only terminal Run
// is ready for the purging_runs phase.
func projectAwaitingRunPurge(t *testing.T, ctx context.Context) (*pgxpool.Pool, *runstore.PostgresStore, string) {
	t.Helper()
	pool := isolatedPool(t, ctx)
	projects := projectstore.NewPostgresStore(pool)
	runs := runstore.NewPostgresStore(pool)
	project, _, err := projects.Create(ctx, projectstore.CreateParams{
		ProjectID: "project-purge-budget", OwnerID: "user-1", Kind: projectstore.KindProject,
		Name: "Purge budget", IdempotencyKey: "create-project",
		RequestDigest: "sha256:" + strings.Repeat("a", 64),
	})
	if err != nil {
		t.Fatal(err)
	}
	run := createProjectRun(t, ctx, runs, "run-purge-budget", project.ProjectID)
	if _, err := runs.TransitionRun(
		ctx, run.RunID, runstore.RunInitializing, runstore.RunFailed, runstore.Reason{Code: "fixture_failed"},
	); err != nil {
		t.Fatal(err)
	}
	if _, _, err := projects.BeginDeletion(ctx, projectstore.BeginDeletionParams{
		ProjectID: project.ProjectID, OwnerID: project.OwnerID, ExpectedRevision: project.Revision,
	}); err != nil {
		t.Fatal(err)
	}
	controller := newTestController(t, pool, runs, &recordingNotifier{}, "setup")
	for iteration := 0; iteration < 2; iteration++ {
		if worked, err := controller.RunOnce(ctx); err != nil || !worked {
			t.Fatalf("advance to Run purge %d = (%t, %v)", iteration, worked, err)
		}
	}
	current, err := projects.Get(ctx, project.OwnerID, project.ProjectID)
	if err != nil || current.Deletion == nil || current.Deletion.Phase != projectstore.DeletionPurgingRuns {
		t.Fatalf("Project before Run purge = (%+v, %v)", current, err)
	}
	return pool, runs, project.ProjectID
}

func TestProjectRunPurgeHasItsOwnBudgetAndClaimLease(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	pool, runs, projectID := projectAwaitingRunPurge(t, ctx)
	operationTimeout, purgeTimeout, claimDuration := 500*time.Millisecond, 10*time.Second, time.Minute
	var lease time.Duration
	slow := &delayedPurgeRuns{PostgresStore: runs, delay: 3 * operationTimeout, onDelete: func(ctx context.Context) {
		if err := pool.QueryRow(ctx, `
SELECT deletion_claim_expires_at - deletion_claimed_at FROM projects WHERE project_id = $1`,
			projectID).Scan(&lease); err != nil {
			t.Error(err)
		}
	}}
	controller, err := New(pool, slow, &recordingNotifier{}, Options{
		PollInterval: time.Millisecond, ClaimDuration: claimDuration,
		OperationTimeout: operationTimeout, PurgeTimeout: purgeTimeout,
	})
	if err != nil {
		t.Fatal(err)
	}
	if worked, err := controller.RunOnce(ctx); err != nil || !worked {
		t.Fatalf("slow Run purge = (%t, %v)", worked, err)
	}
	if _, err := runs.GetRun(ctx, "run-purge-budget"); !errors.Is(err, runstore.ErrNotFound) {
		t.Fatalf("purged Run lookup error = %v", err)
	}
	// The purge claim keeps the ordinary margin over the longer budget.
	want := claimDuration + purgeTimeout - operationTimeout
	if lease < want || lease > want+time.Second {
		t.Fatalf("Run purge claim lease = %s, want %s", lease, want)
	}
}

func TestProjectDeletionDefersClaimAfterPhaseBudgetExpires(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	pool, runs, projectID := projectAwaitingRunPurge(t, ctx)
	blocked := &delayedPurgeRuns{PostgresStore: runs, delay: -1}
	controller, err := New(pool, blocked, &recordingNotifier{}, Options{
		PollInterval: time.Second, ClaimDuration: time.Hour,
		OperationTimeout: time.Second, PurgeTimeout: 300 * time.Millisecond,
	})
	if err != nil {
		t.Fatal(err)
	}
	if worked, err := controller.RunOnce(ctx); !errors.Is(err, context.DeadlineExceeded) || worked {
		t.Fatalf("expired Run purge = (%t, %v)", worked, err)
	}
	// The failed attempt's exhausted context must not prevent deferral: the
	// Project becomes claimable after one poll, not after the whole lease.
	var deferred bool
	if err := pool.QueryRow(ctx, `
SELECT deletion_claim_expires_at <= clock_timestamp() + interval '1 minute'
FROM projects WHERE project_id = $1`, projectID).Scan(&deferred); err != nil || !deferred {
		t.Fatalf("Project deletion claim deferred = (%t, %v)", deferred, err)
	}
}

func createProjectRun(
	t *testing.T,
	ctx context.Context,
	store *runstore.PostgresStore,
	runID string,
	projectID string,
) runstore.WorkflowRun {
	t.Helper()
	run, err := store.CreateRun(ctx, projectRunParams(runID, projectID))
	if err != nil {
		t.Fatalf("create Project Run %q: %v", runID, err)
	}
	return run
}

func projectRunParams(runID, projectID string) runstore.CreateRunParams {
	return runstore.CreateRunParams{
		RunID: runID, OwnerID: "user-1", ProjectID: &projectID,
		WorkflowName: "artifact-copy", WorkflowVersion: "1",
		WorkflowSchemaVersion: contracts.APIVersion,
		WorkflowSnapshot:      json.RawMessage(`{"ref":{"name":"artifact-copy","version":"1"}}`),
		Parameters:            map[string]string{},
		RuntimeConfig:         runtimeconfig.BuiltInRunSnapshot(),
	}
}

type recordingNotifier struct{ runIDs []string }

func (n *recordingNotifier) Cancel(runID string) { n.runIDs = append(n.runIDs, runID) }

func newTestController(
	t *testing.T,
	pool *pgxpool.Pool,
	runs *runstore.PostgresStore,
	notifier *recordingNotifier,
	id string,
) *Controller {
	t.Helper()
	sequence := 0
	controller, err := New(pool, runs, notifier, Options{
		PollInterval: time.Millisecond, ClaimDuration: time.Minute,
		OperationTimeout: 5 * time.Second,
		NewID: func(prefix string) (string, error) {
			sequence++
			return prefix + id + "-" + string(rune('a'+sequence)), nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	return controller
}

func expireDeletionClaim(t *testing.T, ctx context.Context, pool *pgxpool.Pool, projectID string) {
	t.Helper()
	if _, err := pool.Exec(ctx, `
UPDATE projects
SET deletion_claimed_at = clock_timestamp() - interval '2 seconds',
    deletion_claim_expires_at = clock_timestamp() - interval '1 second'
WHERE project_id = $1 AND deletion_claim_id IS NOT NULL`, projectID); err != nil {
		t.Fatal(err)
	}
}

func isolatedPool(t *testing.T, ctx context.Context) *pgxpool.Pool {
	t.Helper()
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	adminConfig, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	admin, err := pgxpool.NewWithConfig(ctx, adminConfig)
	if err != nil {
		t.Fatal(err)
	}
	if err := admin.Ping(ctx); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	random := make([]byte, 8)
	if _, err := rand.Read(random); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	schema := "contractor_project_lifecycle_test_" + hex.EncodeToString(random)
	identifier := pgx.Identifier{schema}.Sanitize()
	if _, err := admin.Exec(ctx, `CREATE SCHEMA `+identifier); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	config, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		admin.Close()
		t.Fatal(err)
	}
	config.ConnConfig.RuntimeParams["search_path"] = schema
	pool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		admin.Close()
		t.Fatal(err)
	}
	if _, err := persistencepostgres.ApplyMigrations(ctx, pool); err != nil {
		pool.Close()
		admin.Close()
		t.Fatal(err)
	}
	t.Cleanup(func() {
		pool.Close()
		cleanupContext, cleanupCancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cleanupCancel()
		_, _ = admin.Exec(cleanupContext, `DROP SCHEMA `+identifier+` CASCADE`)
		admin.Close()
	})
	return pool
}
