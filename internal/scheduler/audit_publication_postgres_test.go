package scheduler

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
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
	store := auditstoretest.NewStore(pool)
	audit, _, err := store.CreateDraft(ctx, auditstore.CreateDraftParams{
		AuditID: "audit-publication", OwnerID: project.OwnerID, ProjectID: project.ProjectID,
		Profile:         auditstore.ProfileIdentity{Name: "test", Version: "1", Digest: schedulerAuditDigest("2")},
		ProfileSnapshot: json.RawMessage(`{"profile":"test","workflows":{"check":{"kind":"check"}}}`),
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
	manifest := auditstore.ExactArtifact{
		Ref:    contracts.ArtifactRef{Namespace: "audit-test", Name: "manifest", Revision: schedulerString("manifest-r1")},
		Digest: schedulerAuditDigest("4"),
	}
	task := auditstore.ExactArtifact{
		Ref:    contracts.ArtifactRef{Namespace: "audit-test", Name: "task", Revision: schedulerString("task-r1")},
		Digest: schedulerAuditDigest("5"),
	}
	roundID := "round-audit-publication"
	if _, _, err := store.MaterializeRound(ctx, auditstore.MaterializeRoundParams{
		OwnerID: audit.OwnerID, AuditID: audit.AuditID, ExpectedRevision: audit.Revision,
		RoundID: roundID, RoundOrdinal: 1, Manifest: manifest,
		BaselineSnapshot: json.RawMessage(`{"baseline":"test"}`), DeadlineAt: time.Now().Add(time.Hour),
		Items: []auditstore.MaterializedItem{{
			ItemID: "item-audit-publication", ItemKey: "check", Ordinal: 0,
			Kind: "test", SubjectKey: "subject", Task: task, Origin: schedulerAuditOrigin("check", task), WorkflowRole: "check",
			InitialState: auditstore.ItemReady,
			Coverage: auditstore.Coverage{
				Status: auditstore.CoverageNotTested, Requested: []string{}, Completed: []string{}, Gaps: []string{},
			},
		}},
		IdempotencyKey: "start-audit-publication", RequestDigest: schedulerAuditDigest("6"),
	}); err != nil {
		t.Fatal(err)
	}
	claims, err := store.Claim(ctx, auditstore.ClaimParams{HolderID: "controller-publication", Lease: time.Minute, Limit: 1})
	if err != nil || len(claims) != 1 {
		t.Fatalf("claim Audit = (%+v, %v)", claims, err)
	}
	claim := claims[0]
	if _, err := store.TransitionRound(ctx, auditstore.RoundTransitionParams{
		Claim: claim, RoundID: roundID, ExpectedRevision: 1,
		ExpectedState: auditstore.RoundAccepted, TargetState: auditstore.RoundExecuting,
	}); err != nil {
		t.Fatal(err)
	}
	execution, _, err := store.CreateExecutionIntent(ctx, auditstore.CreateExecutionIntentParams{
		Claim: claim, ExecutionID: "execution-audit-publication", RoundID: &roundID,
		Role: auditstore.ExecutionCheck, WorkflowRole: "check", Manifest: manifest,
		SubmissionKey: "audit-publication-submission", RequestDigest: schedulerAuditDigest("7"),
		Members: []auditstore.ExecutionMemberIntent{{
			ExecutionItemID: "execution-item-audit-publication", ItemID: "item-audit-publication",
			BatchOrdinal: 0, ItemAttempt: 1, Task: task, Inputs: []auditstore.ExactArtifact{},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	runID := "run-audit-publication"
	if err := persistencepostgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		runs := runstore.NewPostgresStore(tx)
		if _, err := runs.CreateAuditRun(ctx, runstore.CreateAuditRunParams{
			CreateRunParams: runstore.CreateRunParams{
				RunID: runID, OwnerID: project.OwnerID, ProjectID: &project.ProjectID,
				WorkflowName: "audit-check", WorkflowVersion: "1",
				WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: json.RawMessage(`{}`),
				Parameters: map[string]string{}, MetadataLabels: runstore.RunMetadataLabels{},
				RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot(),
			},
			AuditExecutionID: execution.ExecutionID, AuditSubmissionKey: execution.SubmissionKey,
		}); err != nil {
			return err
		}
		if _, err := auditstoretest.NewStore(tx).BindRun(ctx, auditstore.BindRunParams{
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
	artifactService := artifacts.NewService(artifacts.NewPostgresRepository(pool))
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
		}, 0)
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
	publications, err := runstore.NewPostgresStore(pool).ListRunOutputPublications(ctx, runID)
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

func schedulerAuditOrigin(entryKey string, source auditstore.ExactArtifact) auditstore.ItemOrigin {
	ref := source.Ref
	return auditstore.ItemOrigin{
		Schema: auditstore.ItemOriginSchema, SourceRef: &ref,
		SourceContentDigest: schedulerAuditDigest("8"), SourceMediaType: "application/json",
		CanonicalInventoryDigest: schedulerAuditDigest("9"), EntryKey: entryKey,
	}
}

func schedulerString(value string) *string { return &value }
