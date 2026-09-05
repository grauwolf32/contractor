package runservice

import (
	"context"
	cryptorand "crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/agentskills"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestPostgresTrustedAuditRunIsAtomicReplayableAndPinsExactSkill(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedRunServicePool(t, ctx, databaseURL)
	projects := projectstore.NewPostgresStore(pool)
	project, _, err := projects.Create(ctx, projectstore.CreateParams{
		ProjectID: "project-run-service", OwnerID: "owner-run-service", Kind: projectstore.KindProject,
		Name: "Audit Run Service", IdempotencyKey: "project-run-service", RequestDigest: testDigest("1"),
	})
	if err != nil {
		t.Fatal(err)
	}
	artifactService := artifacts.NewService(artifacts.NewPostgresRepository(pool))
	projectArtifacts, err := artifactService.Project(project.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	source := writeExact(t, ctx, projectArtifacts, "sources", "application", "text/plain", []byte("source"))
	task := writeExact(t, ctx, projectArtifacts, "audit-test", "task", "application/zip", []byte("task"))
	manifest := writeExact(t, ctx, projectArtifacts, "audit-test", "manifest", "application/zip", []byte("manifest"))

	userArtifacts, err := artifactService.User("owner-run-service")
	if err != nil {
		t.Fatal(err)
	}
	skillAWrite, err := userArtifacts.Write(
		ctx, contracts.ArtifactRef{Namespace: contracts.AgentSkillNamespace, Name: "review"},
		artifacts.Payload{MediaType: agentskills.MediaType, Data: []byte("skill revision A")}, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	skillAMetadata, err := userArtifacts.Metadata(ctx, skillAWrite.Ref)
	if err != nil {
		t.Fatal(err)
	}

	audits := auditstore.NewPostgresStore(pool)
	draft, _, err := audits.CreateDraft(ctx, auditstore.CreateDraftParams{
		AuditID: "audit-run-service", OwnerID: "owner-run-service", ProjectID: project.ProjectID,
		Profile:         auditstore.ProfileIdentity{Name: "test", Version: "1", Digest: testDigest("2")},
		ProfileSnapshot: []byte(`{"profile":"test"}`), InputSelection: []byte(`{"source":"pinned"}`),
		Limits: auditstore.Limits{
			MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 10, MaxItemsTotal: 10,
			MaxSubmittedRuns: 10, MaxItemRunAttempts: 2, MaxEvidenceBytes: 1 << 20,
		},
		IdempotencyKey: "audit-run-service", RequestDigest: testDigest("3"),
	})
	if err != nil {
		t.Fatal(err)
	}
	roundID := "round-run-service"
	if _, _, err := audits.MaterializeRound(ctx, auditstore.MaterializeRoundParams{
		OwnerID: draft.OwnerID, AuditID: draft.AuditID, ExpectedRevision: draft.Revision,
		RoundID: roundID, RoundOrdinal: 1, Manifest: manifest,
		BaselineSnapshot: []byte(`{"baseline":"pinned"}`), DeadlineAt: time.Now().Add(time.Hour),
		Items: []auditstore.MaterializedItem{{
			ItemID: "item-run-service", ItemKey: "check-1", Ordinal: 0, Kind: "test",
			SubjectKey: "subject-1", Task: task, Origin: runServiceAuditOrigin("check-1", task), WorkflowRole: "check",
			InitialState: auditstore.ItemReady,
			Coverage: auditstore.Coverage{
				Status: auditstore.CoverageNotTested, Requested: []string{"check"},
				Completed: []string{}, Gaps: []string{},
			},
		}},
		IdempotencyKey: "audit-run-service-start", RequestDigest: testDigest("4"),
	}); err != nil {
		t.Fatal(err)
	}
	claims, err := audits.Claim(ctx, auditstore.ClaimParams{HolderID: "controller-run-service", Lease: time.Minute, Limit: 1})
	if err != nil || len(claims) != 1 {
		t.Fatalf("claim Audit = (%+v, %v)", claims, err)
	}
	claim := claims[0]
	if _, err := audits.TransitionRound(ctx, auditstore.RoundTransitionParams{
		Claim: claim, RoundID: roundID, ExpectedRevision: 1,
		ExpectedState: auditstore.RoundAccepted, TargetState: auditstore.RoundExecuting,
	}); err != nil {
		t.Fatal(err)
	}
	execution, _, err := audits.CreateExecutionIntent(ctx, auditstore.CreateExecutionIntentParams{
		Claim: claim, ExecutionID: "execution-run-service", RoundID: &roundID,
		Role: auditstore.ExecutionCheck, Manifest: manifest,
		SubmissionKey: "audit-run-service-check-1-attempt-1", RequestDigest: testDigest("5"),
		Members: []auditstore.ExecutionMemberIntent{{
			ExecutionItemID: "execution-item-run-service", ItemID: "item-run-service",
			BatchOrdinal: 0, ItemAttempt: 1, Task: task, Inputs: []auditstore.ExactArtifact{source},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}

	manager, err := config.NewManager(config.ManagerOptions{
		OperatorRoot: filepath.Join("..", "config", "testdata", "valid"),
		ManagedRoot:  filepath.Join(t.TempDir(), "managed"), Descriptors: config.MVPDescriptors(),
	})
	if err != nil {
		t.Fatal(err)
	}
	workflow, err := manager.Workflow("artifact-copy@1")
	if err != nil {
		t.Fatal(err)
	}
	addWorkflowSkill(&workflow, contracts.ArtifactRef{Namespace: contracts.AgentSkillNamespace, Name: "review"})
	skillSnapshot := contracts.RunSkillSnapshot{
		Name: "review", Source: exactRefPointer(skillAMetadata.Ref),
		SourceDigest: skillAMetadata.Digest, SourceSize: skillAMetadata.Size,
	}
	if err := skillSnapshot.Validate(); err != nil {
		t.Fatal(err)
	}
	workflowSnapshot, err := json.Marshal(workflow)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runstore.NewPostgresStore(pool).CreateAuditRun(ctx, runstore.CreateAuditRunParams{
		CreateRunParams: runstore.CreateRunParams{
			RunID: "run-orphan-must-rollback", OwnerID: project.OwnerID, ProjectID: &project.ProjectID,
			WorkflowName: workflow.Ref.Name, WorkflowVersion: workflow.Ref.Version,
			WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: workflowSnapshot,
			Parameters: map[string]string{"objective": "check"}, MetadataLabels: runstore.RunMetadataLabels{},
			RuntimeConfig: runtimeconfig.BuiltInRunSnapshot(),
		},
		AuditExecutionID: execution.ExecutionID, AuditSubmissionKey: execution.SubmissionKey,
	}); persistencepostgres.SQLState(err) != "23514" {
		t.Fatalf("unbound Audit-managed Run error = %v", err)
	}
	if _, err := runstore.NewPostgresStore(pool).GetRun(ctx, "run-orphan-must-rollback"); !errors.Is(err, runstore.ErrNotFound) {
		t.Fatalf("unbound Audit-managed Run committed: %v", err)
	}
	// Advance the logical binding before dispatch. The trusted path must retain
	// revision A and must never resolve this new current revision B.
	skillBWrite, err := userArtifacts.Write(
		ctx, contracts.ArtifactRef{Namespace: contracts.AgentSkillNamespace, Name: "review"},
		artifacts.Payload{MediaType: agentskills.MediaType, Data: []byte("skill revision B")},
		skillAWrite.Ref.Revision,
	)
	if err != nil {
		t.Fatal(err)
	}

	service, err := New(Options{
		Runs: runstore.NewPostgresStore(pool), Workflows: manager,
		LLMCredentials: emptyCredentialLookup{}, CredentialGuard: openCredentialGuard{},
		RuntimeCredentials: acceptRuntimeCredentials{}, Projects: projects,
		SkillInitializationAvailable: true,
		PublicTransaction:            postgresPublicTransaction(pool),
		AuditTransaction:             postgresAuditTransaction(pool),
	})
	if err != nil {
		t.Fatal(err)
	}
	params := AuditCreateParams{
		Claim: claim, ExecutionID: execution.ExecutionID, Workflow: workflow,
		RuntimeConfig: runtimeconfig.BuiltInRunSnapshot(), Skills: []contracts.RunSkillSnapshot{skillSnapshot},
		Parameters: map[string]string{"objective": "check"}, Inputs: map[string]auditstore.ExactArtifact{"source": source},
		ExecutionManifest: manifest, RequestDigest: execution.RequestDigest,
	}
	var generated atomic.Int64
	params.NewRunID = func() (string, error) {
		value := generated.Add(1)
		return "run-audit-" + string(rune('0'+value)), nil
	}

	results := make([]CreateResult, 2)
	errorsByCall := make([]error, 2)
	var wait sync.WaitGroup
	for index := range results {
		wait.Add(1)
		go func(index int) {
			defer wait.Done()
			results[index], errorsByCall[index] = service.CreateAudit(ctx, params)
		}(index)
	}
	wait.Wait()
	for index, callErr := range errorsByCall {
		if callErr != nil {
			t.Fatalf("concurrent trusted create %d: %v", index, callErr)
		}
	}
	if results[0].Run.RunID != results[1].Run.RunID || generated.Load() != 1 ||
		results[0].Created == results[1].Created {
		t.Fatalf("concurrent create results = %+v / %+v, generated=%d", results[0], results[1], generated.Load())
	}
	stored, err := runstore.NewPostgresStore(pool).GetRun(ctx, results[0].Run.RunID)
	if err != nil || stored.PublicationMode != runstore.PublicationAuditManaged ||
		stored.AuditExecutionID == nil || *stored.AuditExecutionID != execution.ExecutionID ||
		stored.AuditSubmissionKey == nil || *stored.AuditSubmissionKey != execution.SubmissionKey ||
		len(stored.SkillSnapshot) != 1 || stored.SkillSnapshot[0].Source == nil ||
		*stored.SkillSnapshot[0].Source.Revision != *skillAWrite.Ref.Revision ||
		stored.SkillSnapshot[0].SourceDigest != skillAMetadata.Digest {
		t.Fatalf("stored trusted Run = (%+v, %v)", stored, err)
	}
	currentSkill, err := userArtifacts.Metadata(ctx, contracts.ArtifactRef{Namespace: "skills", Name: "review"})
	if err != nil || currentSkill.Ref.Revision == nil || *currentSkill.Ref.Revision != *skillBWrite.Ref.Revision {
		t.Fatalf("current Skill binding = (%+v, %v)", currentSkill, err)
	}
	params.NewRunID = func() (string, error) { return "", errors.New("replay resolved a new identity") }
	replay, err := service.CreateAudit(ctx, params)
	if err != nil || !replay.Replayed || replay.Run.RunID != stored.RunID {
		t.Fatalf("trusted response-loss replay = (%+v, %v)", replay, err)
	}
}

type emptyCredentialLookup struct{}

func (emptyCredentialLookup) LookupLLMCredential(context.Context, string) (config.CredentialMetadata, error) {
	return config.CredentialMetadata{}, errors.New("credential unavailable")
}

type openCredentialGuard struct{}

func (openCredentialGuard) WithRunCreation(ctx context.Context, fn func() error) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	return fn()
}

type acceptRuntimeCredentials struct{}

func (acceptRuntimeCredentials) ValidateRuntimeCredential(context.Context, string, ...string) error {
	return nil
}

func postgresPublicTransaction(pool *pgxpool.Pool) PublicTransaction {
	return func(ctx context.Context, fn func(PublicRunWriter, *artifacts.Service) error) error {
		return persistencepostgres.InTx(ctx, pool, pgx.TxOptions{IsoLevel: pgx.RepeatableRead}, func(tx pgx.Tx) error {
			return fn(runstore.NewPostgresStore(tx), artifacts.NewService(artifacts.NewPostgresRepository(tx)))
		})
	}
}

func postgresAuditTransaction(pool *pgxpool.Pool) AuditTransaction {
	return func(ctx context.Context, fn func(AuditRunWriter, *artifacts.Service, AuditExecutionWriter) error) error {
		return persistencepostgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
			return fn(
				runstore.NewPostgresStore(tx), artifacts.NewService(artifacts.NewPostgresRepository(tx)),
				auditstore.NewPostgresStore(tx),
			)
		})
	}
}

func writeExact(
	t *testing.T,
	ctx context.Context,
	store artifacts.ScopedStore,
	namespace string,
	name string,
	mediaType string,
	payload []byte,
) auditstore.ExactArtifact {
	t.Helper()
	written, err := store.Write(
		ctx, contracts.ArtifactRef{Namespace: namespace, Name: name},
		artifacts.Payload{MediaType: mediaType, Data: payload}, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	metadata, err := store.Metadata(ctx, written.Ref)
	if err != nil {
		t.Fatal(err)
	}
	return auditstore.ExactArtifact{
		Ref: metadata.Ref, Digest: metadata.Digest, MediaType: metadata.MediaType, SizeBytes: metadata.Size,
	}
}

func addWorkflowSkill(workflow *config.ResolvedWorkflow, skill contracts.ArtifactRef) {
	for stageName, stage := range workflow.Stages {
		for agentName, binding := range stage.Agents {
			binding.Template.Skills = []contracts.ArtifactRef{skill}
			stage.Agents[agentName] = binding
			workflow.Stages[stageName] = stage
			return
		}
	}
}

func exactRefPointer(ref contracts.ArtifactRef) *contracts.ArtifactRef {
	copy := ref
	if ref.Revision != nil {
		revision := *ref.Revision
		copy.Revision = &revision
	}
	return &copy
}

func testDigest(character string) string {
	return "sha256:" + repeat(character, 64)
}

func runServiceAuditOrigin(entryKey string, source auditstore.ExactArtifact) auditstore.ItemOrigin {
	ref := source.Ref
	return auditstore.ItemOrigin{
		Schema: auditstore.ItemOriginSchema, SourceRef: &ref,
		SourceContentDigest: testDigest("8"), SourceMediaType: "application/json",
		CanonicalInventoryDigest: testDigest("9"), EntryKey: entryKey,
	}
}

func repeat(value string, count int) string {
	result := ""
	for range count {
		result += value
	}
	return result
}

func isolatedRunServicePool(
	t *testing.T,
	ctx context.Context,
	databaseURL string,
) *pgxpool.Pool {
	t.Helper()
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
		t.Skipf("PostgreSQL is unavailable: %v", err)
	}
	suffix := make([]byte, 6)
	if _, err := cryptorand.Read(suffix); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	databaseName := "contractor_runservice_" + hex.EncodeToString(suffix)
	if _, err := admin.Exec(ctx, `CREATE DATABASE `+databaseName); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	t.Cleanup(func() {
		cleanupContext, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_, _ = admin.Exec(cleanupContext, `DROP DATABASE IF EXISTS `+databaseName+` WITH (FORCE)`)
		admin.Close()
	})
	config := adminConfig.Copy()
	config.ConnConfig.Database = databaseName
	pool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(pool.Close)
	if _, err := persistencepostgres.ApplyMigrations(ctx, pool); err != nil {
		t.Fatal(err)
	}
	return pool
}
