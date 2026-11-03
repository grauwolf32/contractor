package scheduler

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/configtest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/jackc/pgx/v5"
)

func TestAuditManagedRunSkipsGenericProjectOutputPublication(t *testing.T) {
	if os.Getenv("CONTRACTOR_TEST_DATABASE_URL") == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	pool := isolatedSchedulerPool(t, ctx)
	project, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: "project-audit-publication", OwnerID: "owner-audit-publication",
		Kind: projectstore.KindProject, Name: "Audit publication",
		IdempotencyKey: "project-audit-publication", RequestDigest: schedulerAuditDigest("1"),
	})
	if err != nil {
		t.Fatal(err)
	}
	root := configtest.CopyWithPolicies(t, "../config/testdata/valid")
	if err := os.CopyFS(root, os.DirFS("../config/testdata/audit-common-catalog")); err != nil {
		t.Fatal(err)
	}
	catalog, _, err := configload.LoadUnionReadOnly(root, filepath.Join(t.TempDir(), "managed"))
	if err != nil {
		t.Fatal(err)
	}
	pinned, err := catalog.AuditProfile("empty-tasks@1")
	if err != nil {
		t.Fatal(err)
	}
	profile, _ := json.Marshal(pinned)
	store := auditstoretest.NewStore(pool)
	audit, _, err := store.CreateDraft(ctx, auditstore.CreateDraftParams{
		AuditID: "audit-publication", OwnerID: project.OwnerID, ProjectID: project.ProjectID,
		Profile:         auditstore.ProfileIdentity{Name: pinned.Ref.Name, Version: pinned.Ref.Version, Digest: pinned.Ref.Digest},
		ProfileSnapshot: profile,
		InputSelection:  json.RawMessage(`{"input":"test"}`),
		Limits: auditstore.Limits{
			MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 1, MaxItemsTotal: 1,
			MaxSubmittedRuns: 1, MaxItemRunAttempts: 1, MaxEvidenceBytes: 1024,
		},
		IdempotencyKey: "audit-publication", RequestDigest: schedulerAuditDigest("3"),
	})
	if err != nil {
		t.Fatal(err)
	}
	audit, _, err = store.Activate(ctx, auditstore.ActivateParams{OwnerID: audit.OwnerID, AuditID: audit.AuditID, ExpectedRevision: audit.Revision, BaselineSnapshot: json.RawMessage(`{"schema":"contractor.audit.baseline.v1","inputs":{},"skills":[]}`), DeadlineAt: time.Now().Add(time.Hour), IdempotencyKey: "start", RequestDigest: schedulerAuditDigest("4")})
	if err != nil {
		t.Fatal(err)
	}
	accepted, _, err := store.AcceptTasks(ctx, auditstore.AcceptTasksParams{OwnerID: audit.OwnerID, AuditID: audit.AuditID, ExpectedRevision: audit.Revision, IdempotencyKey: "accept", RequestDigest: schedulerAuditDigest("5"), Tasks: []auditstore.AcceptedTask{{Document: auditcontract.Task{Schema: auditcontract.TaskSchema, AuditID: audit.AuditID, TaskID: "publication-task", Workflow: "inspect", Inputs: map[string]auditcontract.Material{}, Parameters: map[string]string{}, Objective: "Check generic publication suppression"}, Source: auditstore.TaskSource{Kind: "manual", Key: "publication-task"}}}})
	if err != nil {
		t.Fatal(err)
	}
	claims, err := store.Claim(ctx, auditstore.ClaimParams{HolderID: "controller-publication", Lease: time.Minute, Limit: 1})
	if err != nil || len(claims) != 1 {
		t.Fatalf("claim Audit = (%+v, %v)", claims, err)
	}
	claim := claims[0]
	execution, _, err := store.SelectTasks(ctx, auditstore.SelectTasksParams{Claim: claim, TaskIDs: accepted.TaskIDs})
	if err != nil {
		t.Fatal(err)
	}
	runID := "run-audit-publication"
	if err := persistencepostgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		runs := runstore.NewPostgresStore(tx, artifactpgtest.NewStorage())
		if _, err := runs.CreateAuditRun(ctx, runstore.CreateAuditRunParams{
			CreateRunParams: runstore.CreateRunParams{
				LaneClass: resolved.LaneClassModel,
				RunID:     runID, OwnerID: project.OwnerID, ProjectID: &project.ProjectID,
				WorkflowName: "audit-check", WorkflowVersion: "1",
				WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: json.RawMessage(`{}`),
				Parameters: map[string]string{}, MetadataLabels: runstore.RunMetadataLabels{},
				RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot(),
			},
			AuditExecutionID: execution.ExecutionID, AuditSubmissionKey: execution.SubmissionKey,
		}); err != nil {
			return err
		}
		if _, err := auditstoretest.NewStore(tx, artifactpgtest.NewStorage()).BindRun(ctx, auditstore.BindRunParams{
			Claim: claim, ExecutionID: execution.ExecutionID, RunID: runID,
		}); err != nil {
			return err
		}
		if _, err := runs.TransitionRun(
			ctx, runID, runstore.RunInitializing, runstore.RunPending, runstore.Reason{Code: "initialized"},
		); err != nil {
			return err
		}
		_, err := runs.TransitionRun(
			ctx, runID, runstore.RunPending, runstore.RunRunning, runstore.Reason{Code: "admitted"},
		)
		return err
	}); err != nil {
		t.Fatal(err)
	}
	artifactService := artifacts.NewService(artifactpgtest.NewStorage().Repository(pool))
	runArtifacts, err := artifactService.Run(runID)
	if err != nil {
		t.Fatal(err)
	}
	workerResult, err := runArtifacts.Write(
		ctx, contracts.ArtifactRef{Namespace: "builder", Name: "result"},
		artifacts.Payload{MediaType: "text/plain", Data: []byte("audit result")}, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := bindOutput(ctx, artifactService, runID, "result", workerResult.Ref); err != nil {
		t.Fatal(err)
	}
	if err := freezeOutputs(ctx, artifactService, runID); err != nil {
		t.Fatal(err)
	}
	if err := persistencepostgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		return publishProjectOutputs(ctx, tx, runID, map[string]resolved.ArtifactSlot{
			"result": {Required: true, MediaTypes: []string{"text/plain"}},
		}, 0, artifactpgtest.NewStorage())
	}); err != nil {
		t.Fatalf("skip Audit-managed output publication: %v", err)
	}
	projectArtifacts, err := artifactService.Project(project.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := projectArtifacts.Metadata(ctx, contracts.ArtifactRef{Namespace: "outputs", Name: "result"}); !errors.Is(err, artifacts.ErrArtifactNotFound) {
		t.Fatalf("Audit-managed Run published generic Project output: %v", err)
	}
	publications, err := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage()).ListRunOutputPublications(ctx, runID)
	if err != nil || len(publications) != 0 {
		t.Fatalf("Audit-managed publication receipts = (%+v, %v)", publications, err)
	}
}

func schedulerAuditDigest(character string) string {
	value := "sha256:"
	for range 64 {
		value += character
	}
	return value
}
