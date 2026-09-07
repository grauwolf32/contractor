//go:build integration

package findingintake

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestPostgresFindingReceiptAuditImportDirectVerificationAndRunDeletion(t *testing.T) {
	testPostgresFindingReceiptAuditImportDirectVerificationAndRunDeletion(t)
}

func TestPostgresFindingReceiptReplayRetentionAndRunDeletion(t *testing.T) {
	testPostgresFindingReceiptAuditImportDirectVerificationAndRunDeletion(t)
}

func testPostgresFindingReceiptAuditImportDirectVerificationAndRunDeletion(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	pool := isolatedFindingPool(t, ctx)
	if os.Getenv("CONTRACTOR_TEST_ARTIFACT_BACKEND") == "filesystem" {
		if err := artifacts.ClaimBlobBackend(ctx, pool, artifacts.BlobFilesystem); err != nil {
			t.Fatal(err)
		}
		files, err := artifacts.OpenFilesystemBlobStore(ctx, t.TempDir())
		if err != nil {
			t.Fatal(err)
		}
		defer files.Close()
		ctx = artifacts.WithBlobRuntime(ctx, artifacts.NewBlobRuntime(files, nil))
	}

	const (
		ownerID      = "finding-owner"
		projectID    = "finding-project"
		runID        = "finding-run"
		stageID      = "finding-stage"
		allocationID = "finding-allocation"
	)
	if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: projectID, OwnerID: ownerID, Kind: projectstore.KindProject,
		Name: "Finding project", IdempotencyKey: "finding-project-create",
		RequestDigest: digestBytes([]byte("finding-project")),
	}); err != nil {
		t.Fatal(err)
	}

	snapshot, err := workflowconfig.Load("../config/testdata/valid", workflowconfig.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	workflow, err := snapshot.Workflow("artifact-copy@1")
	if err != nil {
		t.Fatal(err)
	}
	stage := workflow.Stages[workflow.EntryStage]
	binding := stage.Agents["builder"]
	binding.Template.Toolsets = append(binding.Template.Toolsets, contracts.ToolsetSelection{
		Ref:   contracts.ToolsetRef{ToolsetID: "security-findings", Version: "1"},
		Tools: []string{"finding"},
	})
	stage.Agents["builder"] = binding
	workflow.Stages[workflow.EntryStage] = stage
	workflow.Outputs["result"] = workflowconfig.ArtifactSlot{
		Required: true, Primary: true,
		MediaTypes: []string{auditdomain.DirectVerificationsMediaType},
	}
	workflowJSON, err := json.Marshal(workflow)
	if err != nil {
		t.Fatal(err)
	}
	stageJSON, err := json.Marshal(stage)
	if err != nil {
		t.Fatal(err)
	}

	runs := runstore.NewPostgresStore(pool)
	projectRef := projectID
	if _, err := runs.CreateRun(ctx, runstore.CreateRunParams{
		RunID: runID, OwnerID: ownerID, ProjectID: &projectRef,
		WorkflowName: workflow.Ref.Name, WorkflowVersion: workflow.Ref.Version,
		WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: workflowJSON,
		Parameters: map[string]string{}, RuntimeConfig: runtimeconfig.BuiltInRunSnapshot(),
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := runs.TransitionRun(ctx, runID, runstore.RunInitializing, runstore.RunRunning,
		runstore.Reason{Code: "finding_test_started"}); err != nil {
		t.Fatal(err)
	}
	if _, err := runs.CreateStageExecution(ctx, runstore.CreateStageExecutionParams{
		StageExecutionID: stageID, RunID: runID, StageName: workflow.EntryStage, Attempt: 1,
		StageSpecSchemaVersion: contracts.APIVersion, StageSpecSnapshot: stageJSON,
		StageContextSchemaVersion: contracts.APIVersion,
		StageContext: runstore.StageContextSnapshot{
			Parameters: map[string]string{}, Artifacts: map[string]runstore.PinnedContextArtifact{},
		},
	}); err != nil {
		t.Fatal(err)
	}
	runtimeAgentID := strings.Repeat("a", 64)
	runtimeInstanceID := "finding-runtime-instance"
	if err := runs.RecordStageAllocation(ctx, runstore.StageAllocation{
		AllocationID: allocationID, StageExecutionID: stageID,
		LogicalAgentName: "builder", Namespace: binding.Namespace,
		AgentTemplateRef: binding.Template.Ref, WorkerRuntimeRef: binding.Template.Runtime,
		RuntimeAgentID: runtimeAgentID, RuntimeAgentInstanceID: runtimeInstanceID,
		RuntimeAgentLabelRevision:         1,
		RuntimeConfigurationSchemaVersion: runstore.AllocationRuntimeConfigurationSchemaVersion,
		RuntimeConfiguration:              findingRuntimeConfiguration(),
		PerformanceCollectionPolicy:       contracts.PerformanceCollectionDisabled,
	}); err != nil {
		t.Fatal(err)
	}
	grant := controlplane.AllocationGrant{
		AllocationID: allocationID, RuntimeAgentID: runtimeAgentID,
		RuntimeInstanceID: runtimeInstanceID, RunID: runID, StageExecutionID: stageID,
		LogicalAgentName: "builder", Namespace: binding.Namespace,
	}

	artifactService := artifacts.NewService(artifacts.NewPostgresRepository(pool))
	runArtifacts, err := artifactService.Run(runID)
	if err != nil {
		t.Fatal(err)
	}
	evidenceWrite, err := runArtifacts.Write(ctx,
		artifacts.ArtifactRef{Namespace: "builder", Name: "trace"},
		artifacts.Payload{MediaType: "text/plain", Data: []byte("trusted evidence")}, nil)
	if err != nil {
		t.Fatal(err)
	}

	service, err := New(pool)
	if err != nil {
		t.Fatal(err)
	}
	firstInput := testSubmission("worker-invocation-1", "candidate-1", []contracts.ArtifactRef{evidenceWrite.Ref})
	type submitResult struct {
		receipt  Receipt
		replayed bool
		err      error
	}
	start := make(chan struct{})
	results := make(chan submitResult, 2)
	var workers sync.WaitGroup
	for range 2 {
		workers.Add(1)
		go func() {
			defer workers.Done()
			<-start
			receipt, replayed, submitErr := service.Submit(ctx, grant, firstInput)
			results <- submitResult{receipt: receipt, replayed: replayed, err: submitErr}
		}()
	}
	close(start)
	workers.Wait()
	close(results)
	accepted := make([]submitResult, 0, 2)
	for result := range results {
		if result.err != nil {
			t.Fatal(result.err)
		}
		accepted = append(accepted, result)
	}
	if len(accepted) != 2 || accepted[0].receipt.ReceiptID != accepted[1].receipt.ReceiptID ||
		accepted[0].replayed == accepted[1].replayed {
		t.Fatalf("concurrent receipt results = %+v", accepted)
	}
	firstReceipt := accepted[0].receipt
	storedRun, err := runs.GetRun(ctx, runID)
	if err != nil {
		t.Fatal(err)
	}
	if firstReceipt.Origin.Workflow.ClosureDigest != digestBytes(storedRun.WorkflowSnapshot) ||
		firstReceipt.Origin.RunID != runID || firstReceipt.Origin.InvocationID != firstInput.InvocationID {
		t.Fatalf("trusted receipt origin = %+v", firstReceipt.Origin)
	}
	changed := firstInput
	changed.Proposal.Title = "Changed after acceptance"
	if _, _, err := service.Submit(ctx, grant, changed); !errors.Is(err, ErrConflict) {
		t.Fatalf("changed replay error = %v, want conflict", err)
	}
	foreign := grant
	foreign.AllocationID = "foreign-allocation"
	foreign.RuntimeAgentID = strings.Repeat("b", 64)
	foreignInput := testSubmission("worker-invocation-2", "foreign", nil)
	if _, _, err := service.Submit(ctx, foreign, foreignInput); !errors.Is(err, ErrAccessDenied) {
		t.Fatalf("foreign allocation error = %v, want access denied", err)
	}

	secondInput := testSubmission("worker-invocation-3", "candidate-unimported", nil)
	secondReceipt, replayed, err := service.Submit(ctx, grant, secondInput)
	if err != nil || replayed {
		t.Fatalf("second receipt = (%+v, replay=%v, %v)", secondReceipt, replayed, err)
	}
	claimOnlyInput := testSubmission("worker-invocation-4", "candidate-claim-only", nil)
	claimOnlyReceipt, replayed, err := service.Submit(ctx, grant, claimOnlyInput)
	if err != nil || replayed {
		t.Fatalf("claim-only receipt = (%+v, replay=%v, %v)", claimOnlyReceipt, replayed, err)
	}
	directOutput, err := auditdomain.EncodeDirectVerificationSet(auditdomain.DirectVerificationSet{
		Schema: auditdomain.DirectVerificationsSchema,
		Verifications: []auditdomain.DirectVerificationResult{{
			InvocationID: firstInput.InvocationID, ClientKey: firstInput.Proposal.ClientKey,
			Assessment: "supported", Summary: "The exact retained trace verifies the candidate.",
			EvidenceIDs: []string{"evidence-1"},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	directSource, err := runArtifacts.Write(ctx,
		artifacts.ArtifactRef{Namespace: "builder", Name: "direct-result"},
		artifacts.Payload{
			MediaType: auditdomain.DirectVerificationsMediaType,
			Data:      directOutput,
		}, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifactService.BindOutputExact(
		ctx, runID, "result", directSource.Ref, nil,
	); err != nil {
		t.Fatal(err)
	}

	const auditID = "finding-audit"
	if _, _, err := auditstore.NewPostgresStore(pool).CreateDraft(ctx, auditstore.CreateDraftParams{
		AuditID: auditID, OwnerID: ownerID, ProjectID: projectID,
		Profile:         auditstore.ProfileIdentity{Name: "finding-review", Version: "1", Digest: digestBytes([]byte("profile"))},
		ProfileSnapshot: json.RawMessage(`{"interaction":{"findingConfirmation":"human-required"}}`),
		InputSelection:  json.RawMessage(`{}`),
		Limits: auditstore.Limits{
			MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 1, MaxItemsTotal: 1,
			MaxSubmittedRuns: 1, MaxItemRunAttempts: 1, MaxEvidenceBytes: 1 << 20,
		},
		IdempotencyKey: "finding-audit-create", RequestDigest: digestBytes([]byte("audit")),
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `UPDATE finding_proposal_receipts SET workflow_name = 'forged' WHERE receipt_id = $1`, firstReceipt.ReceiptID); persistencepostgres.SQLState(err) != "23514" {
		t.Fatalf("receipt provenance rewrite SQLSTATE = %q, error = %v", persistencepostgres.SQLState(err), err)
	}
	hold, replayed, err := service.ImportIntoAudit(ctx, ImportRequest{
		OwnerID: ownerID, AuditID: auditID, RunID: runID, Proposal: firstReceipt.Proposal.Ref,
	})
	if err != nil || replayed || hold.Proposal.Ref.Revision == nil || len(hold.Evidence) != 1 {
		t.Fatalf("first Audit import = (%+v, replay=%v, %v)", hold, replayed, err)
	}
	replayHold, replayed, err := service.ImportIntoAudit(ctx, ImportRequest{
		OwnerID: ownerID, AuditID: auditID, RunID: runID, Proposal: firstReceipt.Proposal.Ref,
	})
	if err != nil || !replayed || !sameRef(replayHold.Proposal.Ref, hold.Proposal.Ref) {
		t.Fatalf("replayed Audit import = (%+v, replay=%v, %v)", replayHold, replayed, err)
	}
	var prematureAssessments int
	if err := pool.QueryRow(ctx, `
SELECT count(*) FROM audit_finding_assessments
 WHERE receipt_id = $1 AND direct_verification`, firstReceipt.ReceiptID).Scan(
		&prematureAssessments,
	); err != nil || prematureAssessments != 0 {
		t.Fatalf("direct assessment before successful frozen Run = (%d, %v)", prematureAssessments, err)
	}
	projectArtifacts, err := artifactService.Project(projectID)
	if err != nil {
		t.Fatal(err)
	}
	retained, err := projectArtifacts.Read(ctx, hold.Proposal.Ref)
	if err != nil || retained.Payload.MediaType != proposalMediaType {
		t.Fatalf("retained proposal = (%+v, %v)", retained, err)
	}

	if err := runs.MarkStageAllocationReleased(ctx, allocationID); err != nil {
		t.Fatal(err)
	}
	if err := artifactService.FreezeRunOutputs(ctx, runID); err != nil {
		t.Fatal(err)
	}
	if _, err := runs.TransitionRun(ctx, runID, runstore.RunRunning, runstore.RunSucceeded,
		runstore.Reason{Code: "finding_test_succeeded"}); err != nil {
		t.Fatal(err)
	}
	if _, replayed, err := service.ImportIntoAudit(ctx, ImportRequest{
		OwnerID: ownerID, AuditID: auditID, RunID: runID, Proposal: firstReceipt.Proposal.Ref,
	}); err != nil || !replayed {
		t.Fatalf("terminal direct-verification import replay = (replay=%v, %v)", replayed, err)
	}
	if _, replayed, err := service.ImportIntoAudit(ctx, ImportRequest{
		OwnerID: ownerID, AuditID: auditID, RunID: runID, Proposal: claimOnlyReceipt.Proposal.Ref,
	}); err != nil || replayed {
		t.Fatalf("claim-only terminal import = (replay=%v, %v)", replayed, err)
	}
	var resultRefJSON, contractRefJSON []byte
	var directAssessmentID, directResultDigest, directContractDigest string
	if err := pool.QueryRow(ctx, `
SELECT assessment_id, result_ref, result_digest, contract_ref, contract_digest
  FROM audit_finding_assessments
 WHERE receipt_id = $1 AND direct_verification`, firstReceipt.ReceiptID).Scan(
		&directAssessmentID, &resultRefJSON, &directResultDigest,
		&contractRefJSON, &directContractDigest,
	); err != nil {
		t.Fatalf("read accepted direct verification: %v", err)
	}
	var directResultRef, directContractRef contracts.ArtifactRef
	if json.Unmarshal(resultRefJSON, &directResultRef) != nil ||
		json.Unmarshal(contractRefJSON, &directContractRef) != nil ||
		directResultRef.ValidateExact() != nil || directContractRef.ValidateExact() != nil ||
		directAssessmentID == "" || directResultDigest == "" || directContractDigest == "" {
		t.Fatalf("direct verification refs = (%s, %+v, %s, %+v, %s)",
			directAssessmentID, directResultRef, directResultDigest, directContractRef, directContractDigest)
	}
	var claimOnlyAssessments int
	if err := pool.QueryRow(ctx, `
SELECT count(*) FROM audit_finding_assessments
 WHERE receipt_id = $1 AND direct_verification`, claimOnlyReceipt.ReceiptID).Scan(
		&claimOnlyAssessments,
	); err != nil || claimOnlyAssessments != 0 {
		t.Fatalf("successful Run claim without exact result assessment = (%d, %v)", claimOnlyAssessments, err)
	}
	var retainedBefore int64
	if err := pool.QueryRow(ctx, `
SELECT retained_evidence_bytes FROM audits WHERE audit_id = $1`, auditID).Scan(&retainedBefore); err != nil {
		t.Fatal(err)
	}
	if _, replayed, err := service.ImportIntoAudit(ctx, ImportRequest{
		OwnerID: ownerID, AuditID: auditID, RunID: runID, Proposal: firstReceipt.Proposal.Ref,
	}); err != nil || !replayed {
		t.Fatalf("accepted direct-verification replay = (replay=%v, %v)", replayed, err)
	}
	var retainedAfter int64
	var directLinks int
	if err := pool.QueryRow(ctx, `
SELECT retained_evidence_bytes FROM audits WHERE audit_id = $1`, auditID).Scan(&retainedAfter); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(ctx, `
SELECT count(*) FROM audit_artifact_links
 WHERE audit_id = $1 AND logical_key LIKE 'finding/%/direct-%'`, auditID).Scan(&directLinks); err != nil {
		t.Fatal(err)
	}
	if retainedAfter != retainedBefore || directLinks != 2 {
		t.Fatalf("direct replay retention = (before=%d after=%d links=%d)",
			retainedBefore, retainedAfter, directLinks)
	}
	if err := runs.DeleteReleasedTerminalRun(ctx, ownerID, runID); err != nil {
		t.Fatal(err)
	}
	listed, err := service.ListAuditInbox(ctx, ownerID, auditID, ListQuery{Limit: 10})
	if err != nil || len(listed) != 2 || listed[0].Retention != RetentionAuditHeld ||
		!listed[0].Origin.RunDeleted || len(listed[0].AuditHolds) != 1 ||
		listed[1].Retention != RetentionAuditHeld || !listed[1].Origin.RunDeleted {
		t.Fatalf("Audit inbox after Run deletion = (%+v, %v)", listed, err)
	}
	secondAfterDelete, err := readReceiptByID(ctx, pool, secondReceipt.ReceiptID)
	if err != nil || secondAfterDelete.Retention != RetentionDiscarded || !secondAfterDelete.Origin.RunDeleted {
		t.Fatalf("discarded receipt after Run deletion = (%+v, %v)", secondAfterDelete, err)
	}
	if _, err := runArtifacts.Read(ctx, firstReceipt.Proposal.Ref); !errors.Is(err, artifacts.ErrArtifactNotFound) {
		t.Fatalf("source proposal read after Run deletion = %v", err)
	}
	if _, err := projectArtifacts.Read(ctx, hold.Proposal.Ref); err != nil {
		t.Fatalf("retained proposal after Run deletion: %v", err)
	}
	for name, ref := range map[string]contracts.ArtifactRef{
		"direct result": directResultRef, "direct contract": directContractRef,
	} {
		if _, err := projectArtifacts.Read(ctx, ref); err != nil {
			t.Fatalf("retained %s after Run deletion: %v", name, err)
		}
	}
	var sourcePins int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM artifact_pins WHERE run_id = $1`, runID).Scan(&sourcePins); err != nil {
		t.Fatal(err)
	}
	if sourcePins != 0 {
		t.Fatalf("source Run retained %d finding pins", sourcePins)
	}
}

func findingRuntimeConfiguration() *runstore.AllocationRuntimeConfiguration {
	gateway := contracts.LLMGatewayConfigRef{
		GatewayID: "local-litellm", Version: "1", Digest: "sha256:" + strings.Repeat("b", 64),
	}
	return &runstore.AllocationRuntimeConfiguration{
		ModelPolicy: contracts.ModelPolicyRef{
			PolicyID: "worker", Version: "1", Digest: "sha256:" + strings.Repeat("c", 64),
		},
		Origins: runtimeconfig.ResolvedRuntimeConfigOrigins{
			LLMGateway: &runtimeconfig.RuntimeFieldOrigin{Layer: runtimeconfig.LayerWorkflow},
		},
		Provenance: contracts.ResolvedRuntimeConfigProvenanceV2{
			Default: contracts.RuntimeLabelBindingProvenanceV2{
				Label: "default", BindingRevision: 1,
				Config: contracts.RuntimeConfigRefV2{
					Name: runtimeconfig.BuiltInName, Version: runtimeconfig.BuiltInVersion,
					Digest: runtimeconfig.BuiltInDigest,
				},
			},
			RunLabels:       []contracts.RuntimeLabelBindingProvenanceV2{},
			AgentLabels:     []contracts.RuntimeLabelBindingProvenanceV2{},
			RuntimeAdapters: []contracts.RuntimeAdapterRef{}, LLMGatewayConfig: &gateway,
			RuntimeCredentialRefs: []contracts.RuntimeCredentialRefV2{},
		},
	}
}

func isolatedFindingPool(t *testing.T, ctx context.Context) *pgxpool.Pool {
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
	schema := "contractor_finding_test_" + hex.EncodeToString(random)
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
		cleanupCtx, cleanupCancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cleanupCancel()
		_, _ = admin.Exec(cleanupCtx, `DROP SCHEMA `+identifier+` CASCADE`)
		admin.Close()
	})
	return pool
}
