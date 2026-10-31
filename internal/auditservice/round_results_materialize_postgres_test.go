package auditservice

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifactpolicy"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runservice"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/grauwolf32/contractor/internal/strictjson"
	"github.com/jackc/pgx/v5"
)

// A page with 65 distinct retained artifacts requires three fork transactions.
// Use the real accepted collection, source capture, ArtifactStore and Run CAS;
// only the producer's opaque payloads stand in for an already imported package.
func TestPostgresRoundResultsMaterializerResumesDependencyChunks(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	pool := isolatedAuditServicePool(t, ctx, databaseURL)
	audits := auditstoretest.NewStore(pool)
	service := artifacts.NewService(artifacts.NewPostgresRepository(pool))
	project, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: "project", OwnerID: "owner", Kind: projectstore.KindProject, Name: "Dependency chunks",
		IdempotencyKey: "project", RequestDigest: serviceTestDigest("project"),
	})
	if err != nil {
		t.Fatal(err)
	}
	projectScope, _ := artifacts.ProjectScope(project.ProjectID)
	auditID, roundID := "audit", "round"
	namespace := auditdomain.ArtifactNamespace(auditID)
	write := func(scope artifacts.Scope, namespace, name, media string, content []byte) contracts.ExactArtifact {
		t.Helper()
		written, err := service.CreateFrozen(ctx, scope, contracts.ArtifactRef{Namespace: namespace, Name: name}, artifacts.Payload{MediaType: media, Data: content})
		if err != nil {
			t.Fatal(err)
		}
		return contracts.ExactContent(written.Ref, written.MediaType, content)
	}
	task := write(projectScope, namespace, "task", "application/zip", []byte("opaque accepted task"))
	manifest := write(projectScope, namespace, "execution", "application/json", []byte(`{"schema":"fixture"}`))
	audit, _, err := audits.CreateDraft(ctx, auditstore.CreateDraftParams{
		AuditID: auditID, OwnerID: project.OwnerID, ProjectID: project.ProjectID,
		Profile:         auditstore.ProfileIdentity{Name: "fixture", Version: "1", Digest: serviceTestDigest("profile")},
		ProfileSnapshot: json.RawMessage(`{"workflows":{"check":{"kind":"check"},"assess":{"kind":"assessment"}}}`),
		InputSelection:  json.RawMessage(`{}`), Limits: auditstore.Limits{MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 1,
			MaxItemsTotal: 1, MaxSubmittedRuns: 2, MaxItemRunAttempts: 1, MaxEvidenceBytes: 8 << 20},
		IdempotencyKey: "audit", RequestDigest: serviceTestDigest("audit"),
	})
	if err != nil {
		t.Fatal(err)
	}
	origin := auditstore.ItemOrigin{Schema: auditstore.ItemOriginSchema, SourceRef: &task.Ref,
		SourceContentDigest: task.Digest, SourceMediaType: task.MediaType,
		CanonicalInventoryDigest: serviceTestDigest("inventory"), EntryKey: "check", EntryVersion: "1"}
	audit, _, err = audits.MaterializeRound(ctx, auditstore.MaterializeRoundParams{
		OwnerID: audit.OwnerID, AuditID: audit.AuditID, ExpectedRevision: audit.Revision, RoundID: roundID, RoundOrdinal: 1,
		Manifest: manifest, BaselineSnapshot: json.RawMessage(`{"labels":{},"requiredPlacementLabels":[]}`), DeadlineAt: time.Now().Add(time.Hour),
		Items: []auditstore.MaterializedItem{{ItemID: "item", ItemKey: "check", Ordinal: 0, Kind: "checklist", SubjectKey: "subject",
			Task: task, Origin: origin, WorkflowRole: "check", InitialState: auditstore.ItemReady,
			Coverage: auditstore.Coverage{Status: auditstore.CoverageNotTested, Requested: []string{}, Completed: []string{}, Gaps: []string{}}}},
		IdempotencyKey: "round", RequestDigest: serviceTestDigest("round"),
	})
	if err != nil {
		t.Fatal(err)
	}
	claims, err := audits.Claim(ctx, auditstore.ClaimParams{HolderID: "controller", Lease: time.Minute, Limit: 1})
	if err != nil || len(claims) != 1 {
		t.Fatalf("claim: %+v %v", claims, err)
	}
	claim := claims[0]
	if _, err = audits.TransitionRound(ctx, auditstore.RoundTransitionParams{Claim: claim, RoundID: roundID, ExpectedRevision: 1,
		ExpectedState: auditstore.RoundAccepted, TargetState: auditstore.RoundExecuting}); err != nil {
		t.Fatal(err)
	}
	check, _, err := audits.CreateExecutionIntent(ctx, auditstore.CreateExecutionIntentParams{Claim: claim, ExecutionID: "check-execution",
		RoundID: &roundID, Role: auditstore.ExecutionCheck, WorkflowRole: "check", LaneClass: resolved.LaneClassModel, Manifest: manifest,
		SubmissionKey: "check-submission", RequestDigest: serviceTestDigest("check"), Members: []auditstore.ExecutionMemberIntent{{
			ExecutionItemID: "member", ItemID: "item", BatchOrdinal: 0, ItemAttempt: 1, Task: task, Inputs: []auditstore.ExactArtifact{},
		}}})
	if err != nil {
		t.Fatal(err)
	}
	profile, err := loadAuditServiceProfiles(t).AuditProfile("test-checklist@1")
	if err != nil {
		t.Fatal(err)
	}
	workflow := profile.Workflows["check"].Workflow
	workflowBytes, err := json.Marshal(workflow)
	if err != nil {
		t.Fatal(err)
	}
	bind := func(execution auditstore.Execution, runID string) runstore.WorkflowRun {
		t.Helper()
		var run runstore.WorkflowRun
		err := persistencepostgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
			var err error
			run, err = runstore.NewPostgresStore(tx).CreateAuditRun(ctx, runstore.CreateAuditRunParams{
				CreateRunParams: runstore.CreateRunParams{RunID: runID, OwnerID: project.OwnerID, ProjectID: &project.ProjectID,
					WorkflowName: workflow.Ref.Name, WorkflowVersion: workflow.Ref.Version, WorkflowSchemaVersion: contracts.APIVersion,
					WorkflowSnapshot: workflowBytes, Parameters: map[string]string{}, MetadataLabels: runstore.RunMetadataLabels{},
					RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot(), LaneClass: resolved.LaneClassModel},
				AuditExecutionID: execution.ExecutionID, AuditSubmissionKey: execution.SubmissionKey})
			if err != nil {
				return err
			}
			_, err = auditstoretest.NewTxStore(tx).BindRun(ctx, auditstore.BindRunParams{Claim: claim, ExecutionID: execution.ExecutionID, RunID: runID})
			return err
		})
		if err != nil {
			t.Fatal(err)
		}
		return run
	}
	producer := bind(check, "producer")
	producerScope, _ := artifacts.RunScope(producer.RunID)
	retained := make([]auditstore.ArtifactLink, 0, 65)
	expectedBytes := make(map[string][]byte, 65)
	originals := make(map[string]contracts.ExactArtifact, 65)
	for index := range 65 {
		content := []byte(fmt.Sprintf("accepted opaque dependency %03d", index))
		original := write(producerScope, "worker", fmt.Sprintf("content-%03d", index), "text/plain", content)
		fork, err := service.Fork(ctx, artifacts.Fork{Kind: artifacts.LineageAuditImport, Source: producerScope, SourceRef: original.Ref,
			Target: projectScope, TargetRef: contracts.ArtifactRef{Namespace: namespace, Name: fmt.Sprintf("retained-%03d", index)}, Frozen: true, Replay: true})
		if err != nil {
			t.Fatal(err)
		}
		exact := contracts.ExactContent(fork.TargetRef, original.MediaType, content)
		logical := fmt.Sprintf("evidence/%s/evidence-%03d", check.ExecutionID, index)
		provenance, _ := json.Marshal(map[string]any{"source": map[string]any{"executionItemId": "member"},
			"evidence": map[string]any{"id": fmt.Sprintf("evidence-%03d", index), "kind": "source", "artifact": original.Ref}})
		if index == 0 {
			logical = "result/member"
			provenance = []byte(`{"executionItemId":"member"}`)
		}
		retained = append(retained, auditstore.ArtifactLink{Scope: "project", LogicalKey: logical, Artifact: exact, SourceProvenance: provenance})
		expectedBytes[exact.Ref.Key()], originals[exact.Ref.Key()] = content, original
	}
	runs := runstore.NewPostgresStore(pool)
	for _, transition := range [][2]runstore.WorkflowRunState{{runstore.RunInitializing, runstore.RunPending}, {runstore.RunPending, runstore.RunRunning}, {runstore.RunRunning, runstore.RunSucceeded}} {
		if _, err := runs.TransitionRun(ctx, producer.RunID, transition[0], transition[1], runstore.Reason{Code: "fixture"}); err != nil {
			t.Fatal(err)
		}
	}
	cursor, err := runs.GetRunEventCursor(ctx, producer.RunID)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := audits.ObserveTerminal(ctx, auditstore.ObserveTerminalParams{Claim: claim, ExecutionID: check.ExecutionID,
		RunID: producer.RunID, Generation: cursor.Generation, Sequence: uint64(cursor.Sequence)}); err != nil {
		t.Fatal(err)
	}
	result := retained[0].Artifact
	sourceResult := originals[result.Ref.Key()]
	if _, _, err := audits.Collect(ctx, auditstore.CollectParams{Claim: claim, ReceiptID: "accepted-receipt", ExecutionID: check.ExecutionID,
		Disposition: auditstore.CollectionAccepted, SourceOutput: &sourceResult, Retained: retained, RequestDigest: serviceTestDigest("collection"),
		Items: []auditstore.CollectionItem{{ExecutionItemID: "member", Disposition: auditstore.CollectionAccepted, FinalDisposition: auditstore.FinalAccepted,
			Result: &result, Coverage: auditstore.Coverage{Status: auditstore.CoverageInconclusive, Requested: []string{"source"}, Completed: []string{}, Gaps: []string{"fixture-gap"}}}}}); err != nil {
		t.Fatal(err)
	}
	round, err := audits.GetRound(ctx, auditID, roundID)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := audits.TransitionRound(ctx, auditstore.RoundTransitionParams{Claim: claim, RoundID: roundID, ExpectedRevision: round.Revision,
		ExpectedState: auditstore.RoundExecuting, TargetState: auditstore.RoundAssessing}); err != nil {
		t.Fatal(err)
	}
	step := auditstore.RoundResultsStepParams{Claim: claim, RoundID: roundID}
	snapshot, err := audits.AdvanceRoundResults(ctx, step)
	if err != nil || !snapshot.Sealed || snapshot.PageCount != 1 {
		t.Fatalf("capture: %+v %v", snapshot, err)
	}
	pageBytes, err := audits.ReadRoundResultsPage(ctx, step, snapshot.Generation, 0)
	if err != nil {
		t.Fatal(err)
	}
	page, err := auditdomain.DecodeRoundResultsPage(pageBytes)
	if err != nil || len(page.Items) != 1 || len(page.Items[0].Attachments) != 64 {
		t.Fatalf("source dependency closure: %+v %v", page, err)
	}
	pageExact := write(projectScope, namespace, "snapshot-page", "application/json", pageBytes)
	if err := audits.PublishRoundResultsPage(ctx, auditstore.PublishRoundResultsPageParams{RoundResultsStepParams: step,
		Generation: snapshot.Generation, Ordinal: 0, Artifact: pageExact}); err != nil {
		t.Fatal(err)
	}
	rootDoc := auditdomain.RoundResultsManifest{Schema: auditdomain.RoundResultsSchema, AuditID: auditID, RoundID: roundID,
		SourceGeneration: snapshot.Generation, ItemCount: 1, MetadataBytes: snapshot.MetadataBytes,
		Pages: []auditdomain.RoundResultsPageRef{{Ordinal: 0, ItemCount: 1, Artifact: pageExact}}}
	rootBytes, err := auditdomain.EncodeRoundResultsManifest(rootDoc)
	if err != nil {
		t.Fatal(err)
	}
	root := write(projectScope, namespace, "snapshot-root", "application/json", rootBytes)
	if err := audits.PublishRoundResultsManifest(ctx, auditstore.PublishRoundResultsManifestParams{RoundResultsStepParams: step,
		Generation: snapshot.Generation, Manifest: root, Content: rootDoc}); err != nil {
		t.Fatal(err)
	}
	roleAttempt := 1
	assessment, _, err := audits.CreateExecutionIntent(ctx, auditstore.CreateExecutionIntentParams{Claim: claim, ExecutionID: "assessment",
		RoundID: &roundID, Role: auditstore.ExecutionAssessment, RoleAttempt: &roleAttempt, WorkflowRole: "assess", LaneClass: resolved.LaneClassModel,
		Manifest: manifest, SubmissionKey: "assessment-submission", RequestDigest: serviceTestDigest("assessment")})
	if err != nil {
		t.Fatal(err)
	}
	child := bind(assessment, "consumer")
	roots := map[string]auditstore.ExactArtifact{"snapshot": root}
	identity, materialize, err := RoundResultsInputMaterializer(roots)
	if err != nil {
		t.Fatal(err)
	}
	// All forks and their exact revision checkpoint commit or roll back together.
	slice := func(callback runservice.AuditInputMaterializer, rollback bool) (runstore.AuditInputMaterialization, error) {
		var next runstore.AuditInputMaterialization
		err := persistencepostgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
			auditTx, runTx := auditstoretest.NewTxStore(tx), runstore.NewPostgresStore(tx)
			intent, err := auditTx.GetRunCreationIntent(ctx, claim, assessment.ExecutionID)
			if err != nil {
				return err
			}
			accepted, err := auditTx.GetAcceptedRoundResults(ctx, auditID, roundID)
			if err != nil || accepted.Verify(root) != nil {
				return fmt.Errorf("accepted authority changed: %w", err)
			}
			state, err := runTx.GetAuditInputMaterialization(ctx, child.RunID)
			if err != nil {
				return err
			}
			if state == nil {
				state = &runstore.AuditInputMaterialization{Identity: identity, DeadlineAt: child.CreatedAt.Add(runstore.AuditInputMaterializationTimeout),
					Checkpoint: json.RawMessage(`{}`), Refs: []contracts.ArtifactRef{}}
			}
			next, err = callback(ctx, artifacts.NewService(artifacts.NewPostgresRepository(tx)), intent, child, *state)
			if err != nil {
				return err
			}
			next.Revision = state.Revision + 1
			if err = runTx.SetAuditInputMaterialization(ctx, child.RunID, state.Revision, next); err != nil {
				return err
			}
			if rollback {
				return errors.New("injected transaction rollback after fork/checkpoint writes")
			}
			return nil
		})
		return next, err
	}
	countForks := func() int {
		t.Helper()
		var count int
		if err := pool.QueryRow(ctx, `SELECT count(*) FROM artifact_lineage WHERE source_scope_kind='project' AND source_scope_id=$1 AND target_scope_kind='run' AND target_scope_id=$2`, project.ProjectID, child.RunID).Scan(&count); err != nil {
			t.Fatal(err)
		}
		return count
	}
	first, err := slice(materialize, false)
	if err != nil || first.Complete || first.Revision != 1 || countForks() != 32 || len(first.Refs) != 0 {
		t.Fatalf("first32: %+v %v forks=%d", first, err, countForks())
	}
	// Lose the acknowledgement, reconstruct the initializer and read only its
	// persisted checkpoint. A transient failure after the next 32 writes must
	// leave no forks, holds or progress from that rolled-back transaction.
	_, materialize, err = RoundResultsInputMaterializer(roots)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := slice(materialize, true); err == nil || countForks() != 32 {
		t.Fatalf("rollback leaked dependency forks: %v count=%d", err, countForks())
	}
	stored, err := runs.GetAuditInputMaterialization(ctx, child.RunID)
	if err != nil || stored.Revision != 1 || !bytes.Equal(stored.Checkpoint, first.Checkpoint) {
		t.Fatalf("rollback changed checkpoint: %+v %v", stored, err)
	}
	second, err := slice(materialize, false)
	if err != nil || second.Complete || second.Revision != 2 || countForks() != 64 {
		t.Fatalf("second32: %+v %v forks=%d", second, err, countForks())
	}
	_, materialize, err = RoundResultsInputMaterializer(roots)
	if err != nil {
		t.Fatal(err)
	}
	last, err := slice(materialize, false)
	if err != nil || !last.Complete || last.Revision != 3 || countForks() != 65 || len(last.Refs) != 3 {
		t.Fatalf("complete: %+v %v forks=%d", last, err, countForks())
	}
	consumerStore, _ := service.Run(child.RunID)
	inputMetadata, err := consumerStore.Metadata(ctx, last.Refs[2])
	if err != nil {
		t.Fatal(err)
	}
	input, err := consumerStore.ReadExact(ctx, inputMetadata.Exact())
	if err != nil {
		t.Fatal(err)
	}
	var consumer auditdomain.RoundResultsConsumer
	if strictjson.Decode(input.Payload.Data, &consumer) != nil {
		t.Fatal("consumer descriptor is not strict JSON")
	}
	if _, err := auditdomain.EncodeRoundResultsConsumer(consumer); err != nil || len(consumer.Pages) != 1 || consumer.Source.Verify(root) != nil {
		t.Fatalf("consumer: %+v %v", consumer, err)
	}
	localBytes, err := consumerStore.ReadExact(ctx, consumer.Pages[0].Artifact)
	if err != nil {
		t.Fatal(err)
	}
	local, err := auditdomain.DecodeRoundResultsPage(localBytes.Payload.Data)
	if err != nil || len(local.Mappings) != 65 || local.Items[0].Coverage.Status != "inconclusive" || len(local.Items[0].Attachments) != 64 {
		t.Fatalf("rewritten page: %+v %v", local, err)
	}
	localBySource := make(map[string]contracts.ExactArtifact, len(local.Mappings))
	for _, mapping := range local.Mappings {
		if mapping.Consumer.Ref.Namespace != artifactpolicy.AuditContextNamespace || mapping.Source.Digest != mapping.Consumer.Digest {
			t.Fatalf("invalid local attribution: %+v", mapping)
		}
		content, err := consumerStore.ReadExact(ctx, mapping.Consumer)
		if err != nil || !bytes.Equal(content.Payload.Data, expectedBytes[mapping.Source.Ref.Key()]) {
			t.Fatalf("rewritten exact dependency %s: %v", mapping.Source.Ref.Key(), err)
		}
		localBySource[mapping.Source.Ref.Key()] = mapping.Consumer
	}
	if len(localBySource) != 65 || local.Items[0].Result.Verify(localBySource[result.Ref.Key()]) != nil ||
		local.Items[0].SelectedAttempt.SourceResult.Verify(originals[result.Ref.Key()]) != nil {
		t.Fatal("result local authority or original provenance changed")
	}
	for index, attachment := range local.Items[0].Attachments {
		source := page.Items[0].Attachments[index]
		if attachment.Artifact.Verify(localBySource[source.Artifact.Ref.Key()]) != nil || attachment.Original == nil || attachment.Original.Verify(originals[source.Artifact.Ref.Key()]) != nil {
			t.Fatalf("attachment provenance/local authority changed: %+v", attachment)
		}
	}
	var children, submitted int
	if err := pool.QueryRow(ctx, `SELECT (SELECT count(*) FROM workflow_runs WHERE audit_execution_id=$1),submitted_run_count FROM audits WHERE audit_id=$2`, assessment.ExecutionID, auditID).Scan(&children, &submitted); err != nil || children != 1 || submitted != 2 {
		t.Fatalf("replay changed allocation: children=%d debit=%d %v", children, submitted, err)
	}
}
