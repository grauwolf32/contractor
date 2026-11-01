package projectlifecycle

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifactpolicy"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/artifactstest"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
	"github.com/grauwolf32/contractor/internal/auditstore/hookstore"
	"github.com/grauwolf32/contractor/internal/auditstore/ownerstore"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/findingintake"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/persistence/postgrestest"
	"github.com/grauwolf32/contractor/internal/projectsitemap"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/rundeletion"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runstore/runstoretest"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestProducerRunDeletionPreservesProjectSitemapAndReceipt(t *testing.T) {
	ctx := t.Context()
	pool := isolatedPool(t, ctx)
	projects := projectstore.NewPostgresStore(pool)
	project, _, err := projects.Create(ctx, projectstore.CreateParams{
		ProjectID: "surface-project", OwnerID: "user-1", Kind: projectstore.KindProject,
		Name: "Surface", IdempotencyKey: "create-surface", RequestDigest: "sha256:" + strings.Repeat("a", 64),
	})
	if err != nil {
		t.Fatal(err)
	}
	runs := runstore.NewPostgresStore(pool)
	producer := createProjectRun(t, ctx, runs, "surface-producer", project.ProjectID)
	service := projectsitemap.NewService(pool)
	authority := projectsitemap.Authority{
		OwnerID: project.OwnerID, ProjectID: project.ProjectID, RunID: producer.RunID,
		SourceID: "http-transport", Kind: "http", CanObserve: true,
	}
	command := projectsitemap.Mutation{Operation: "observe", Observations: []projectsitemap.Observation{{
		URL: "https://example.test/surface", Method: "GET", Outcome: "response", Status: 200, At: "2026-10-26T00:00:00Z",
	}}}
	accepted, err := service.Mutate(ctx, authority, "collect", command)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runs.TransitionRun(ctx, producer.RunID, runstore.RunInitializing, runstore.RunFailed, runstore.Reason{Code: "fixture_failed"}); err != nil {
		t.Fatal(err)
	}
	if err := rundeletion.New(pool).DeleteReleasedTerminalRun(ctx, project.OwnerID, producer.RunID); err != nil {
		t.Fatal(err)
	}
	if _, err := runs.GetRun(ctx, producer.RunID); !errors.Is(err, runstore.ErrNotFound) {
		t.Fatalf("producer still exists: %v", err)
	}
	projectArtifacts, err := artifacts.NewService(artifacts.NewPostgresRepository(pool)).Project(project.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	read, err := projectArtifacts.Read(ctx, contracts.ArtifactRef{Namespace: accepted.Ref.Namespace, Name: accepted.Ref.Name})
	if err != nil {
		t.Fatal(err)
	}
	document, err := projectsitemap.Decode(read.Payload.Data)
	if err != nil {
		t.Fatal(err)
	}
	if !read.Ref.SameExact(accepted.Ref) || len(document.Groups) != 1 ||
		document.Groups[0].Members[0].Summaries[0].Provenance.SourceID != authority.SourceID {
		t.Fatal("Run deletion changed self-contained Project metadata")
	}
	replay, err := service.Mutate(ctx, authority, "collect", command)
	if err != nil || !replay.Replayed || !replay.Ref.SameExact(accepted.Ref) {
		t.Fatalf("Run deletion lost Project-owned receipt: %+v %v", replay, err)
	}
}

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
		HTTPTarget: &runtimesettings.HTTPOriginTargetRef{
			URL: "https://app.example.test/api",
			Credential: &runtimesettings.RuntimeCredentialRef{
				CredentialID: "project-origin", Kind: runtimesettings.RuntimeCredentialOriginBearer,
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
		ctx, active.RunID, runstore.RunInitializing, runstore.RunPending,
		runstore.Reason{Code: "initialized"},
	)
	if err == nil {
		active, err = runs.TransitionRun(
			ctx, active.RunID, runstore.RunPending, runstore.RunRunning,
			runstore.Reason{Code: "admitted"},
		)
	}
	if err != nil {
		t.Fatal(err)
	}
	execution, err := runs.CreateStageExecution(ctx, runstore.CreateStageExecutionParams{
		LaneClass:        resolved.LaneClassModel,
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
	if _, err := projectsitemap.NewService(pool).Mutate(ctx, projectsitemap.Authority{
		OwnerID: project.OwnerID, ProjectID: projectID, RunID: active.RunID, SourceID: "http-transport", Kind: "http", CanObserve: true,
	}, "collect-surface", projectsitemap.Mutation{Operation: "observe", Observations: []projectsitemap.Observation{{URL: "https://example.test/surface", Method: "GET", Outcome: "response", Status: 200, At: "2026-10-26T00:00:00Z"}}}); err != nil {
		t.Fatal(err)
	}
	if _, err := runs.TransitionRun(
		ctx, completed.RunID, runstore.RunInitializing, runstore.RunFailed,
		runstore.Reason{Code: "fixture_failed"},
	); err != nil {
		t.Fatal(err)
	}
	// The Run purge leaves these receipts as discarded tombstones; the Project
	// purge deletes them, but not the receipt of a Run outside any Project.
	seedFindingReceipts(t, ctx, pool, "completed", completed.RunID, &projectID, 2)
	seedFindingReceipts(t, ctx, pool, "standalone", "run-standalone", nil, 1)
	if _, err := runs.UpdateOwnerQueueControl(ctx, runstore.UpdateOwnerQueueControlParams{
		OwnerID: project.OwnerID, ExpectedRevision: 0, Paused: true,
	}); err != nil {
		t.Fatal(err)
	}
	audits := auditstoretest.NewStore(pool).WithPurgeDependencies(findingintake.DeleteAuditHoldsBatch)
	audit, _, err := audits.CreateDraft(ctx, auditstore.CreateDraftParams{
		AuditID: "audit-project-delete", OwnerID: project.OwnerID, ProjectID: project.ProjectID,
		Profile: auditstore.ProfileIdentity{
			Name: "checklist", Version: "1", Digest: "sha256:" + strings.Repeat("d", 64),
		},
		ProfileSnapshot: json.RawMessage(`{"ref":{"name":"checklist","version":"1"}}`),
		InputSelection:  json.RawMessage(`{}`),
		Limits: auditstore.Limits{
			MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 1, MaxItemsTotal: 1,
			MaxSubmittedRuns: 1, MaxItemRunAttempts: 1, MaxEvidenceBytes: 1 << 20,
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
	if _, err := runstoretest.CreateRun(ctx, runs, projectRunParams("run-late", projectID)); !errors.Is(err, runstore.ErrProjectDeleting) {
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
	deletionEvents, err := audits.EventsAfter(ctx, audit.OwnerID, audit.AuditID, audit.EventSequence, 1)
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
		"Project Runs":              `SELECT count(*) FROM workflow_runs WHERE project_id = 'project-delete'`,
		"Project scope":             `SELECT count(*) FROM artifact_scopes WHERE scope_kind = 'project' AND scope_id = 'project-delete'`,
		"Project revisions":         `SELECT count(*) FROM artifact_binding_revisions WHERE scope_kind = 'project' AND scope_id = 'project-delete'`,
		"Project receipts":          `SELECT count(*) FROM finding_proposal_receipts WHERE project_id = 'project-delete'`,
		"Project mutation receipts": `SELECT count(*) FROM mutation_receipts WHERE project_id='project-delete' AND key_scope[1]='project'`,
	} {
		var count int
		if err := pool.QueryRow(ctx, query).Scan(&count); err != nil || count != 0 {
			t.Fatalf("%s count = %d, error = %v", label, count, err)
		}
	}
	var standaloneReceipts int
	if err := pool.QueryRow(ctx, `
SELECT count(*) FROM finding_proposal_receipts WHERE run_id = 'run-standalone' AND project_id IS NULL`,
	).Scan(&standaloneReceipts); err != nil || standaloneReceipts != 1 {
		t.Fatalf("receipts of a Run outside any Project = %d, error = %v", standaloneReceipts, err)
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
			audits := auditstoretest.NewStore(pool).WithPurgeDependencies(findingintake.DeleteAuditHoldsBatch)
			audit, _, err := audits.CreateDraft(ctx, auditstore.CreateDraftParams{
				AuditID: "locked-audit", OwnerID: project.OwnerID, ProjectID: project.ProjectID,
				Profile: auditstore.ProfileIdentity{
					Name: "checklist", Version: "1", Digest: "sha256:" + strings.Repeat("b", 64),
				},
				ProfileSnapshot: json.RawMessage(`{"ref":{"name":"checklist","version":"1"}}`),
				InputSelection:  json.RawMessage(`{}`),
				Limits: auditstore.Limits{
					MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 1, MaxItemsTotal: 1,
					MaxSubmittedRuns: 1, MaxItemRunAttempts: 1, MaxEvidenceBytes: 1 << 20,
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

// TestProjectDeletionPurgesRetainedProjectFindingHistory finishes a
// Project whose canonical Findings retain independent material and analyst
// history after the producer Run and Audit have gone.
func TestProjectDeletionPurgesRetainedProjectFindingHistory(t *testing.T) {
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
	audits := auditstoretest.NewStore(pool).WithPurgeDependencies(findingintake.DeleteAuditHoldsBatch)
	audit, _, err := audits.CreateDraft(ctx, auditstore.CreateDraftParams{
		AuditID: "decided-audit", OwnerID: project.OwnerID, ProjectID: project.ProjectID,
		Profile: auditstore.ProfileIdentity{
			Name: "checklist", Version: "1", Digest: "sha256:" + strings.Repeat("b", 64),
		},
		ProfileSnapshot: json.RawMessage(`{"ref":{"name":"checklist","version":"1"}}`),
		InputSelection:  json.RawMessage(`{}`),
		Limits: auditstore.Limits{
			MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 1, MaxItemsTotal: 1,
			MaxSubmittedRuns: 1, MaxItemRunAttempts: 1, MaxEvidenceBytes: 1 << 20,
		},
		IdempotencyKey: "create-audit", RequestDigest: "sha256:" + strings.Repeat("c", 64),
	})
	if err != nil {
		t.Fatal(err)
	}
	seedDecidedAuditHistory(t, ctx, pool, audit)
	// The finding and its decision stepped the Project's finding correlation
	// fence, initialized by the Project (FND-R177).
	var fenceRevision int64
	if err := pool.QueryRow(ctx, `SELECT revision FROM finding_correlation_fences WHERE project_id = $1`,
		project.ProjectID).Scan(&fenceRevision); err != nil || fenceRevision < 2 {
		t.Fatalf("finding correlation fence before Project deletion = %d, %v", fenceRevision, err)
	}
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
	for iteration := 0; iteration < 16; iteration++ {
		if worked, err := controller.RunOnce(ctx); err != nil || !worked {
			t.Fatalf("Project cleanup %d = (%t, %v)", iteration, worked, err)
		}
		if _, err := projects.Get(ctx, project.OwnerID, project.ProjectID); errors.Is(err, projectstore.ErrNotFound) {
			break
		}
	}
	if _, err := projects.Get(ctx, project.OwnerID, project.ProjectID); !errors.Is(err, projectstore.ErrNotFound) {
		t.Fatalf("Project did not finish deletion: %v", err)
	}
	var decisions, assessments, receipts, fences int
	if err := pool.QueryRow(ctx, `
SELECT (SELECT count(*) FROM audit_review_decisions),
       (SELECT count(*) FROM project_findings WHERE project_id=$1),
       (SELECT count(*) FROM finding_proposal_receipts WHERE project_id = $1),
       (SELECT count(*) FROM finding_correlation_fences WHERE project_id = $1)`, project.ProjectID,
	).Scan(&decisions, &assessments, &receipts, &fences); err != nil ||
		decisions != 0 || assessments != 0 || receipts != 0 || fences != 0 {
		t.Fatalf("Audit decisions/Findings/receipts/fences after Project deletion = (%d, %d, %d, %d, %v)",
			decisions, assessments, receipts, fences, err)
	}
}

// Seed canonical state through its actual owner, with frozen retained source,
// result and analyst-history Artifacts, after the producer has been deleted.
func seedDecidedAuditHistory(t *testing.T, ctx context.Context, pool *pgxpool.Pool, audit auditstore.Audit) {
	t.Helper()
	doc := auditdomain.FindingProposal{Schema: auditdomain.FindingProposalSchema, ClientKey: "candidate-decided", Title: "Retained issue", Description: "The exact evidence establishes the reported issue.", Preconditions: []string{}, StandardRefs: []auditdomain.StandardReference{}, EvidenceIDs: []string{}, ProposedChecks: []auditdomain.ProposedCheck{}, Limitations: []string{}}
	fp := auditdomain.UnavailableFindingFingerprint("not_recorded")
	anchors := auditdomain.UnavailableFindingAnchors("not_recorded")
	first := seedProjectFindingForDeletion(t, ctx, pool, audit.ProjectID, audit.OwnerID, audit.AuditID, "decided", doc, fp, anchors)
	doc.ClientKey = "candidate-duplicate"
	second := seedProjectFindingForDeletion(t, ctx, pool, audit.ProjectID, audit.OwnerID, audit.AuditID, "duplicate", doc, fp, anchors)
	store := artifacts.NewService(artifacts.NewPostgresRepository(pool))
	write := func(name string) auditstore.ExactArtifact {
		payload := []byte(name)
		file, err := artifactstest.WriteAuditArtifact(ctx, store, audit.ProjectID, contracts.ArtifactRef{Namespace: artifactpolicy.FindingMaterialsNamespace, Name: name}, artifacts.Payload{MediaType: "text/plain", Data: payload})
		if err != nil {
			t.Fatal(err)
		}
		return auditstore.ExactArtifact{Ref: file.Ref, Digest: auditdomain.DigestBytes(payload), MediaType: file.MediaType, SizeBytes: file.Size}
	}
	result, contract := write("retained-result"), write("retained-contract")
	err := persistencepostgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		high := auditstore.SeverityHigh
		if _, err := ownerstore.NewTx(tx).AssessProjectFinding(ctx, auditstore.ProjectFindingEdit{OwnerID: audit.OwnerID, ProjectID: audit.ProjectID, FindingID: first, ExpectedRevision: 1, Verdict: auditstore.VerdictTruePositive, Severity: &high, Rationale: "Confirmed before deletion."}); err != nil {
			return err
		}
		if _, err := ownerstore.NewTx(tx).AssessProjectFinding(ctx, auditstore.ProjectFindingEdit{OwnerID: audit.OwnerID, ProjectID: audit.ProjectID, FindingID: second, ExpectedRevision: 1, Verdict: auditstore.VerdictDuplicate, DuplicateTargetID: &first, Rationale: "Same problem; preserve independent source."}); err != nil {
			return err
		}
		return hookstore.NewFindingEventsTx(tx).CommitDirectFindingAssessment(ctx, auditstore.DirectFindingAssessment{ProjectID: audit.ProjectID, AuditID: audit.AuditID, ReceiptID: "receipt-decided", AssessmentID: "direct-decided", SemanticAssessment: "supported", Result: auditstore.ArtifactLink{LogicalKey: "direct/result", Artifact: result, SourceProvenance: json.RawMessage(`{}`), DisplayRef: "Exact result"}, Contract: auditstore.ArtifactLink{LogicalKey: "direct/contract", Artifact: contract, SourceProvenance: json.RawMessage(`{}`), DisplayRef: "Exact contract"}, RetainedBytes: result.SizeBytes + contract.SizeBytes})
	})
	if err != nil {
		t.Fatal(err)
	}
}
func projectFindingDeletionDigest(value string) string { return auditdomain.DigestBytes([]byte(value)) }
func seedProjectFindingForDeletion(
	t *testing.T, ctx context.Context, pool *pgxpool.Pool,
	projectID, ownerID, auditID, suffix string, document auditdomain.FindingProposal, fingerprint auditdomain.FindingFingerprint, anchors auditdomain.FindingAnchorProjection,
) string {
	t.Helper()
	if len(document.EvidenceIDs) == 0 {
		document.EvidenceIDs = []string{"evidence-1"}
	}
	retainedEvidence := make([]findingintake.ExactArtifact, 0, len(document.EvidenceIDs))
	for _, id := range document.EvidenceIDs {
		body := []byte("Retained evidence " + suffix + " " + id)
		file, err := artifactstest.WriteAuditArtifact(ctx, artifacts.NewService(artifacts.NewPostgresRepository(pool)), projectID, contracts.ArtifactRef{Namespace: artifactpolicy.FindingMaterialsNamespace, Name: suffix + "-" + id}, artifacts.Payload{MediaType: "text/plain", Data: body})
		if err != nil {
			t.Fatal(err)
		}
		retainedEvidence = append(retainedEvidence, findingintake.ExactArtifact{Ref: file.Ref, Digest: auditdomain.DigestBytes(body), MediaType: file.MediaType, SizeBytes: file.Size})
	}
	evidenceJSON, _ := json.Marshal(retainedEvidence)
	payload, err := auditdomain.EncodeFindingProposal(document)
	if err != nil {
		t.Fatal(err)
	}
	projection, err := auditdomain.EncodeFindingReportProjection(auditdomain.ProjectFindingReport(document))
	if err != nil {
		t.Fatal(err)
	}
	namespace := artifactpolicy.FindingMaterialsNamespace
	written, err := artifactstest.WriteAuditArtifact(ctx, artifacts.NewService(artifacts.NewPostgresRepository(pool)), projectID,
		contracts.ArtifactRef{Namespace: namespace, Name: "candidate-" + suffix},
		artifacts.Payload{MediaType: "application/json", Data: payload})
	if err != nil {
		t.Fatal(err)
	}
	proposal := findingintake.ExactArtifact{
		Ref: written.Ref, Digest: auditdomain.DigestBytes(payload), MediaType: written.MediaType, SizeBytes: written.Size,
	}
	proposalJSON, _ := json.Marshal(proposal)
	proposalRefJSON, _ := json.Marshal(proposal.Ref)
	fingerprintJSON, err := auditdomain.EncodeFindingFingerprint(fingerprint)
	if err != nil {
		t.Fatal(err)
	}
	anchorJSON, err := auditdomain.EncodeFindingAnchors(anchors)
	if err != nil {
		t.Fatal(err)
	}
	receiptID := "receipt-" + suffix
	if _, err := pool.Exec(ctx, `
INSERT INTO finding_proposal_receipts (
    receipt_id, proposal_id, allocation_id, runtime_agent_id,
    runtime_instance_id, stage_execution_id, logical_agent_name,
    invocation_id, submission_id, client_key, request_digest,
    run_id, owner_id, project_id,
    workflow_name, workflow_version, workflow_schema_version,
    workflow_configuration_ref, workflow_closure_digest,
    proposal_ref, proposal_digest, proposal_media_type,
    proposal_size_bytes, evidence, report_projection, fingerprint_projection, anchor_projection,
 retention_state,source_run_deleted_at,project_proposal,project_evidence,publication_requested
, checks_projection) VALUES ($1, $2, $3, 'runtime-review', 'instance-review', 'stage-review', 'worker',
          $4, $5, $6, $7, $8, $9, $10,
          'finding-source', '1', 'contractor/v1alpha1',
          '{"name":"finding-source","version":"1"}'::jsonb, $11,
          $12::jsonb, $13, 'application/json', $14, $18::jsonb, $15, $16,$17,'project-held',clock_timestamp(),$19::jsonb,$18::jsonb,true, json_build_object('schema','contractor.audit.finding-checks-projection.v1','proposal_digest',($13)::text,'size_bytes',($14)::bigint,'discarded',false,'subject',NULL,'limitations',NULL,'checks','[]'::json))`,
		receiptID, "proposal-"+suffix, "allocation-"+suffix,
		"invocation-"+suffix, "submission-"+suffix, document.ClientKey,
		projectFindingDeletionDigest("request-"+suffix), "deleted-run-"+suffix, ownerID, projectID,
		projectFindingDeletionDigest("workflow-"+suffix), proposalRefJSON, proposal.Digest, proposal.SizeBytes,
		string(projection), string(fingerprintJSON), string(anchorJSON), evidenceJSON, proposalJSON,
	); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `INSERT INTO finding_proposal_audit_holds(receipt_id,audit_id,project_id,proposal_ref,evidence) VALUES($1,$2,$3,$4::jsonb,$5::jsonb)`, receiptID, auditID, projectID, proposalJSON, evidenceJSON); err != nil {
		t.Fatal(err)
	}
	var findingID string
	err = persistencepostgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		events := hookstore.NewFindingEventsTx(tx)
		var err error
		findingID, err = events.AdmitProjectFinding(ctx, projectID, receiptID, "finding-"+suffix)
		if err != nil {
			return err
		}
		return events.RecordFindingSourceAdded(ctx, auditID, receiptID, "deleted-run-"+suffix)
	})
	if err != nil {
		t.Fatal(err)
	}
	return findingID
}

// seedFindingReceipts inserts count finding proposal receipts of runID, named
// prefix-0001 onwards, for the Project, or for a Run outside any Project when
// projectID is nil.
func seedFindingReceipts(
	t *testing.T, ctx context.Context, pool *pgxpool.Pool,
	prefix, runID string, projectID *string, count int,
) {
	t.Helper()
	if _, err := pool.Exec(ctx, `
INSERT INTO finding_proposal_receipts (
    receipt_id, proposal_id, allocation_id, runtime_agent_id,
    runtime_instance_id, stage_execution_id, logical_agent_name,
    invocation_id, submission_id, client_key, request_digest,
    run_id, owner_id, project_id,
    workflow_name, workflow_version, workflow_schema_version,
    workflow_configuration_ref, workflow_closure_digest,
    proposal_ref, proposal_digest, proposal_media_type,
    proposal_size_bytes, evidence, report_projection
, checks_projection)
SELECT $1 || '-' || lpad(n::text, 4, '0'), $1 || '-proposal-' || n, $1 || '-allocation',
       'runtime', 'instance', 'stage', 'worker', 'invocation', $1 || '-submission-' || n,
       'candidate-' || n, 'sha256:' || repeat('a', 64), $2, 'user-1', $3,
       'finding-source', '1', 'contractor/v1alpha1', '{"name":"finding-source","version":"1"}'::jsonb,
       'sha256:' || repeat('b', 64),
       jsonb_build_object('namespace', 'findings', 'name', 'candidate-' || n, 'revision', 'r1'),
       'sha256:' || repeat('c', 64), 'application/json', 128, '[]'::jsonb,
       '{"description":"Test finding.","limitations":[],"preconditions":[],"severity_suggestion":"","standard_refs":[],"subject":null,"title":"Test finding."}', json_build_object('schema','contractor.audit.finding-checks-projection.v1','proposal_digest',('sha256:' || repeat('c', 64))::text,'size_bytes',(128)::bigint,'discarded',false,'subject',NULL,'limitations',NULL,'checks','[]'::json)
  FROM generate_series(1, $4::integer) AS n`, prefix, runID, projectID, count); err != nil {
		t.Fatalf("seed finding proposal receipts: %v", err)
	}
}

// delayedPurgeRuns makes one Run purge outlast an ordinary operation budget.
type delayedPurgeRuns struct {
	*rundeletion.Deleter
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
	return s.Deleter.DeleteReleasedTerminalRun(ctx, ownerID, runID)
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
	slow := &delayedPurgeRuns{Deleter: rundeletion.New(pool), delay: 3 * operationTimeout, onDelete: func(ctx context.Context) {
		if err := pool.QueryRow(ctx, `
SELECT deletion_claim_expires_at - deletion_claimed_at FROM projects WHERE project_id = $1`,
			projectID).Scan(&lease); err != nil {
			t.Error(err)
		}
	}}
	controller, err := New(pool, runs, slow, &recordingNotifier{}, Options{
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
	blocked := &delayedPurgeRuns{Deleter: rundeletion.New(pool), delay: -1}
	controller, err := New(pool, runs, blocked, &recordingNotifier{}, Options{
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
	run, err := runstoretest.CreateRun(ctx, store, projectRunParams(runID, projectID))
	if err != nil {
		t.Fatalf("create Project Run %q: %v", runID, err)
	}
	return run
}

func projectRunParams(runID, projectID string) runstore.CreateRunParams {
	return runstore.CreateRunParams{
		LaneClass: resolved.LaneClassModel,
		RunID:     runID, OwnerID: "user-1", ProjectID: &projectID,
		WorkflowName: "artifact-copy", WorkflowVersion: "1",
		WorkflowSchemaVersion: contracts.APIVersion,
		WorkflowSnapshot:      json.RawMessage(`{"ref":{"name":"artifact-copy","version":"1"}}`),
		Parameters:            map[string]string{},
		RuntimeConfig:         runtimeconfigtest.BuiltInRunSnapshot(),
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
	controller, err := New(pool, runs, rundeletion.New(pool), notifier, Options{
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
	pool := postgrestest.NewPool(t, ctx, "", postgrestest.Options{})
	if _, err := persistencepostgres.ApplyMigrations(ctx, pool, persistencepostgres.MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	return pool
}
