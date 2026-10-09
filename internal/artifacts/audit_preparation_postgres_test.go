package artifacts_test

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	. "github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/configtest"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type preparationFixture struct {
	ctx       context.Context
	pool      *pgxpool.Pool
	store     *auditstore.PostgresStore
	artifacts *Service
	audit     auditstore.Audit
	claim     auditstore.ControllerClaim
	profile   config.ResolvedAuditProfile
	start     auditstore.StartPreparationParams
	input     auditstore.ExactArtifact
}

func newPreparationFixture(t *testing.T, dependency bool) preparationFixture {
	t.Helper()
	ctx, cancel := context.WithTimeout(t.Context(), 60*time.Second)
	t.Cleanup(cancel)
	pool := isolatedArtifactPool(t, ctx)
	ctx = testBlobContext(t, ctx, pool)
	root := configtest.CopyWithPolicies(t, "../config/testdata/valid")
	for _, topic := range []string{"audit-scan-catalog", "audit-preparation-catalog"} {
		if err := os.CopyFS(root, os.DirFS("../config/testdata/"+topic)); err != nil {
			t.Fatal(err)
		}
	}
	data, err := os.ReadFile("../../api/testdata/audit-composition/prepared-openapi-scan.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if dependency {
		data = []byte(strings.Replace(string(data), "    scan:\n", `    refine-api:
      kind: prepare
      ref: openapi-from-workspace@7
      maxRunAttempts: 2
      inputs:
        source: {source: audit-input, name: source}
        existing_openapi: {source: prepare-output, role: generate-api, name: api}
      parameters: {}
      outputs: {api: openapi, validation: openapi_validation_report}
    scan:
`, 1))
	}
	if err := os.WriteFile(filepath.Join(root, "audit-profiles/prepared.yaml"), data, 0o644); err != nil {
		t.Fatal(err)
	}
	catalog, err := config.Load(root, config.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	profile, err := catalog.AuditProfile("prepared-openapi-scan@1")
	if err != nil {
		t.Fatal(err)
	}
	project, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: "preparation-project", OwnerID: "preparation-owner", Kind: projectstore.KindProject,
		Name: "Preparation", IdempotencyKey: "create", RequestDigest: prepDigest("project"),
	})
	if err != nil {
		t.Fatal(err)
	}
	service := NewService(NewPostgresRepository(pool))
	projectArtifacts, _ := service.Project(project.ProjectID)
	inputs := map[string]auditstore.ExactArtifact{}
	for name, media := range map[string]string{"source": "application/zip", "settings": "application/json"} {
		payload := []byte(name)
		written, err := projectArtifacts.Write(ctx, ArtifactRef{Namespace: "inputs", Name: name}, Payload{MediaType: media, Data: payload}, nil)
		if err != nil {
			t.Fatal(err)
		}
		inputs[name] = auditstore.ExactArtifact{Ref: written.Ref, Digest: auditdomain.DigestBytes(payload), MediaType: media, SizeBytes: int64(len(payload))}
	}
	store := auditstore.NewPostgresStore(pool)
	raw, _ := json.Marshal(profile)
	audit, _, err := store.CreateDraft(ctx, auditstore.CreateDraftParams{
		AuditID: "preparation-audit", OwnerID: project.OwnerID, ProjectID: project.ProjectID,
		Profile:         auditstore.ProfileIdentity{Name: profile.Ref.Name, Version: profile.Ref.Version, Digest: profile.Ref.Digest},
		ProfileSnapshot: raw, InputSelection: json.RawMessage(`{}`),
		Limits: auditstore.Limits{MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 16, MaxItemsTotal: 16,
			MaxSubmittedRuns: 50, MaxItemRunAttempts: 3, MaxEvidenceBytes: 1 << 20},
		IdempotencyKey: "create-audit", RequestDigest: prepDigest("audit"),
	})
	if err != nil {
		t.Fatal(err)
	}
	baseline, _ := json.Marshal(map[string]any{"schema": "contractor.audit.baseline.v1", "inputs": inputs,
		"scope": map[string]string{}, "llmCredentialIds": []string{"preparation-credential"}, "runtimeCredentialIds": []string{}})
	start := auditstore.StartPreparationParams{OwnerID: audit.OwnerID, AuditID: audit.AuditID, ExpectedRevision: audit.Revision,
		BaselineSnapshot: baseline, DeadlineAt: time.Now().Add(time.Hour), IdempotencyKey: "start", RequestDigest: prepDigest("start")}
	audit, created, err := store.StartPreparation(ctx, start)
	if err != nil || !created {
		t.Fatalf("start preparation: created=%t %v", created, err)
	}
	claims, err := store.Claim(ctx, auditstore.ClaimParams{HolderID: "controller", Lease: time.Minute, Limit: 1})
	if err != nil || len(claims) != 1 {
		t.Fatalf("claim: %+v %v", claims, err)
	}
	return preparationFixture{ctx, pool, store, service, audit, claims[0], profile, start, inputs["source"]}
}

func prepDigest(value string) string { return auditdomain.DigestBytes([]byte(value)) }

func (f preparationFixture) intent(role string, attempt int) auditstore.CreateExecutionIntentParams {
	revision := "manifest-revision"
	id := auditdomain.DeterministicID("prepare", f.audit.AuditID, role, string(rune('0'+attempt)))
	return auditstore.CreateExecutionIntentParams{
		Claim: f.claim, ExecutionID: id, Role: auditstore.ExecutionPrepare, WorkflowRole: role, RoleAttempt: &attempt,
		Manifest:      auditstore.ExactArtifact{Ref: ArtifactRef{Namespace: auditdomain.ArtifactNamespace(f.audit.AuditID), Name: "empty-manifest", Revision: &revision}, Digest: prepDigest("manifest")},
		SubmissionKey: id, RequestDigest: prepDigest(id),
		Preparation: &auditstore.PreparationSnapshot{Inputs: map[string]auditstore.ExactArtifact{"source": f.input}, Parameters: map[string]string{}},
	}
}

func (f preparationFixture) current(t *testing.T) auditstore.Audit {
	t.Helper()
	audit, err := f.store.Get(f.ctx, f.audit.OwnerID, f.audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	return audit
}

func (f preparationFixture) collecting(t *testing.T, intent auditstore.CreateExecutionIntentParams, outcome string) (auditstore.Execution, []auditstore.PreparationOutput) {
	t.Helper()
	execution, _, err := f.store.CreateExecutionIntent(f.ctx, intent)
	if err != nil {
		t.Fatal(err)
	}
	runID := "run-" + execution.ExecutionID
	err = persistencepostgres.InTx(f.ctx, f.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		binding := f.profile.Workflows[intent.WorkflowRole]
		workflow, _ := json.Marshal(binding.Workflow)
		_, err := runstore.NewPostgresStore(tx).CreateAuditRun(f.ctx, runstore.CreateAuditRunParams{
			CreateRunParams: runstore.CreateRunParams{RunID: runID, OwnerID: f.audit.OwnerID, ProjectID: &f.audit.ProjectID,
				WorkflowName: binding.Workflow.Ref.Name, WorkflowVersion: binding.Workflow.Ref.Version, WorkflowSchemaVersion: "contractor/v1alpha1",
				WorkflowSnapshot: workflow, Parameters: intent.Preparation.Parameters, RuntimeConfig: runtimeconfig.BuiltInRunSnapshot()},
			AuditExecutionID: execution.ExecutionID, AuditSubmissionKey: execution.SubmissionKey,
		})
		if err != nil {
			return err
		}
		execution, err = auditstore.NewPostgresStore(tx).BindRun(f.ctx, auditstore.BindRunParams{Claim: f.claim, ExecutionID: execution.ExecutionID, RunID: runID})
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
	var outputs []auditstore.PreparationOutput
	if outcome == "succeeded" {
		run, _ := f.artifacts.Run(runID)
		for _, names := range [][2]string{{"api", "openapi"}, {"validation", "openapi_validation_report"}} {
			payload := []byte(`{"name":"` + names[0] + `"}`)
			media := "application/yaml"
			if names[0] == "validation" {
				media = "text/markdown"
			}
			written, err := run.Write(f.ctx, ArtifactRef{Namespace: "stage", Name: names[1]}, Payload{MediaType: media, Data: payload}, nil)
			if err != nil {
				t.Fatal(err)
			}
			bound, err := f.artifacts.BindOutputExact(f.ctx, runID, names[1], written.Ref, nil)
			if err != nil {
				t.Fatal(err)
			}
			outputs = append(outputs, auditstore.PreparationOutput{LogicalName: names[0], WorkflowOutput: names[1],
				Source: auditstore.ExactArtifact{Ref: bound.TargetRef, Digest: auditdomain.DigestBytes(payload), MediaType: media, SizeBytes: int64(len(payload))}})
		}
		if err := f.artifacts.FreezeRunOutputs(f.ctx, runID); err != nil {
			t.Fatal(err)
		}
		for i := range outputs {
			retained, err := f.artifacts.ImportAuditArtifact(f.ctx, runID, outputs[i].Source.Ref, f.audit.ProjectID,
				ArtifactRef{Namespace: auditdomain.ArtifactNamespace(f.audit.AuditID), Name: execution.ExecutionID + "-" + outputs[i].LogicalName})
			if err != nil {
				t.Fatal(err)
			}
			outputs[i].Retained = outputs[i].Source
			outputs[i].Retained.Ref = retained.TargetRef
		}
	}
	if _, err := f.pool.Exec(f.ctx, `UPDATE workflow_runs SET state=$2, state_reason_code='test-terminal', started_at=clock_timestamp(), finished_at=clock_timestamp() WHERE run_id=$1`, runID, outcome); err != nil {
		t.Fatal(err)
	}
	var generation string
	var sequence uint64
	if err := f.pool.QueryRow(f.ctx, `SELECT run_event_generation, next_run_event_sequence-1 FROM workflow_runs WHERE run_id=$1`, runID).Scan(&generation, &sequence); err != nil {
		t.Fatal(err)
	}
	execution, err = f.store.ObserveTerminal(f.ctx, auditstore.ObserveTerminalParams{Claim: f.claim, ExecutionID: execution.ExecutionID, RunID: runID, Generation: generation, Sequence: sequence})
	if err != nil {
		t.Fatal(err)
	}
	return execution, outputs
}

func (f preparationFixture) collection(execution auditstore.Execution, outputs []auditstore.PreparationOutput) auditstore.CollectParams {
	p := auditstore.CollectParams{Claim: f.claim, ExecutionID: execution.ExecutionID, ReceiptID: "receipt-" + execution.ExecutionID,
		RequestDigest: prepDigest("receipt-" + execution.ExecutionID), Disposition: auditstore.CollectionExecutionFailed}
	if len(outputs) > 0 {
		p.Disposition = auditstore.CollectionAccepted
		p.SourceOutput = &outputs[0].Source
		p.PreparationOutputs = outputs
	}
	return p
}

func TestPostgresAuditPreparationIdentityAndReplay(t *testing.T) {
	f := newPreparationFixture(t, false)
	if roles, err := f.store.ListPreparationRoles(f.ctx, f.audit.AuditID); err != nil || len(roles) != 1 || roles[0].Status != auditdomain.PreparationPending || roles[0].Attempts != 0 {
		t.Fatalf("pending preparation projection: %+v %v", roles, err)
	}
	if f.audit.Phase != auditdomain.AuditPhasePreparing || f.audit.CurrentRoundID != nil {
		t.Fatal("preparation invented a Round")
	}
	for _, table := range []string{"audit_rounds", "audit_items"} {
		var count int
		if err := f.pool.QueryRow(f.ctx, "SELECT count(*) FROM "+table).Scan(&count); err != nil || count != 0 {
			t.Fatalf("%s: %d %v", table, count, err)
		}
	}
	if replay, created, err := f.store.StartPreparation(f.ctx, f.start); err != nil || created || replay.AuditID != f.audit.AuditID {
		t.Fatalf("start replay: %t %v", created, err)
	}
	p := f.intent("generate-api", 1)
	round := "forbidden-round"
	p.RoundID = &round
	if _, _, err := f.store.CreateExecutionIntent(f.ctx, p); !errors.Is(err, auditstore.ErrInvalid) {
		t.Fatalf("Round-scoped prepare: %v", err)
	}
	p.RoundID = nil
	p.Claim.Epoch++
	if _, _, err := f.store.CreateExecutionIntent(f.ctx, p); !errors.Is(err, auditstore.ErrClaimLost) {
		t.Fatalf("stale intent: %v", err)
	}
	p.Claim = f.claim
	execution, outputs := f.collecting(t, p, "succeeded")
	creation, err := f.store.GetRunCreationIntent(f.ctx, f.claim, execution.ExecutionID)
	if err != nil || creation.Execution.RoundID != nil || !reflect.DeepEqual(creation.Execution.Preparation, p.Preparation) || len(creation.Items) != 0 {
		t.Fatalf("preparation Run authority: %+v %v", creation, err)
	}
	snapshot, err := f.store.GetReconcileSnapshot(f.ctx, f.claim)
	if err != nil || snapshot.Round != nil || len(snapshot.Items) != 0 || len(snapshot.Executions) != 1 || snapshot.Audit.Phase != auditdomain.AuditPhasePreparing {
		t.Fatalf("preparation reconciliation without a Round: %+v %v", snapshot, err)
	}
	params := f.collection(execution, outputs)
	stale := params
	stale.Claim.Epoch++
	if _, _, err := f.store.Collect(f.ctx, stale); !errors.Is(err, auditstore.ErrClaimLost) {
		t.Fatalf("stale collection: %v", err)
	}
	partial := params
	partial.PreparationOutputs = outputs[:1]
	before := f.current(t)
	if _, _, err := f.store.Collect(f.ctx, partial); !errors.Is(err, auditstore.ErrPrecondition) {
		t.Fatalf("partial output acceptance: %v", err)
	}
	if after := f.current(t); after.Revision != before.Revision || after.RetainedEvidenceBytes != 0 {
		t.Fatal("partial collection changed Audit")
	}
	if _, err := f.store.GetPreparationOutput(f.ctx, f.audit.AuditID, "generate-api", "api"); err == nil {
		t.Fatal("uncommitted output resolved")
	}
	if _, created, err := f.store.Collect(f.ctx, params); err != nil || !created {
		t.Fatalf("collect: %t %v", created, err)
	}
	if roles, err := f.store.ListPreparationRoles(f.ctx, f.audit.AuditID); err != nil || len(roles) != 1 || roles[0].Status != auditdomain.PreparationAccepted || roles[0].Attempts != 1 {
		t.Fatalf("accepted preparation projection: %+v %v", roles, err)
	}
	if _, created, err := f.store.Collect(f.ctx, stale); err != nil || created {
		t.Fatalf("durable replay after stale claim: %t %v", created, err)
	}
	conflict := params
	conflict.RequestDigest = prepDigest("different")
	if _, _, err := f.store.Collect(f.ctx, conflict); !errors.Is(err, auditstore.ErrConflict) {
		t.Fatalf("conflicting receipt: %v", err)
	}
	if _, _, err := f.store.CreateExecutionIntent(f.ctx, f.intent("generate-api", 2)); !errors.Is(err, auditstore.ErrPrecondition) {
		t.Fatalf("accepted role ran twice: %v", err)
	}
	for _, query := range []string{
		`UPDATE audits SET baseline_snapshot=baseline_snapshot || '{"changed":true}'`,
		`UPDATE audit_executions SET preparation_outputs='[]'`,
		`UPDATE audit_executions SET workflow_role='another-role'`,
		`UPDATE audit_executions SET preparation_snapshot='{"inputs":{},"parameters":{}}'`,
	} {
		if _, err := f.pool.Exec(f.ctx, query); persistencepostgres.SQLState(err) != "23514" {
			t.Fatalf("immutable data changed: %s: %v", query, err)
		}
	}
	current := f.current(t)
	if _, err := f.store.CompletePreparation(f.ctx, f.claim, current.Revision-1); !errors.Is(err, auditstore.ErrPrecondition) {
		t.Fatalf("stale preparation revision: %v", err)
	}
	audit, err := f.store.CompletePreparation(f.ctx, f.claim, current.Revision)
	if err != nil || audit.Phase != auditdomain.AuditPhaseInventory || audit.CurrentRoundID != nil || !reflect.DeepEqual(audit.BaselineSnapshot, current.BaselineSnapshot) {
		t.Fatalf("inventory transition: %+v %v", audit, err)
	}
}

func TestPostgresAuditPreparationConcurrentIntentAndReceipt(t *testing.T) {
	f := newPreparationFixture(t, false)
	p := f.intent("generate-api", 1)
	var wg sync.WaitGroup
	created := make(chan bool, 8)
	failures := make(chan error, 8)
	for range 8 {
		wg.Go(func() { _, fresh, err := f.store.CreateExecutionIntent(f.ctx, p); created <- fresh; failures <- err })
	}
	wg.Wait()
	close(created)
	close(failures)
	count := 0
	for fresh := range created {
		if fresh {
			count++
		}
	}
	for err := range failures {
		if err != nil {
			t.Fatal(err)
		}
	}
	if count != 1 {
		t.Fatalf("intent winners: %d", count)
	}
	execution, outputs := f.collecting(t, p, "succeeded")
	params := f.collection(execution, outputs)
	created = make(chan bool, 8)
	failures = make(chan error, 8)
	for range 8 {
		wg.Go(func() { _, fresh, err := f.store.Collect(f.ctx, params); created <- fresh; failures <- err })
	}
	wg.Wait()
	close(created)
	close(failures)
	count = 0
	for fresh := range created {
		if fresh {
			count++
		}
	}
	for err := range failures {
		if err != nil {
			t.Fatal(err)
		}
	}
	if count != 1 {
		t.Fatalf("receipt winners: %d", count)
	}
	audit := f.current(t)
	if audit.ReservedRunCount != 1 || audit.SubmittedRunCount != 1 || audit.OutstandingRunCount != 0 || audit.RetainedEvidenceBytes != outputs[0].Retained.SizeBytes+outputs[1].Retained.SizeBytes {
		t.Fatalf("duplicate accounting: %+v", audit)
	}
}

func TestPostgresAuditPreparationDependenciesAndRetryBounds(t *testing.T) {
	f := newPreparationFixture(t, true)
	dependent := f.intent("refine-api", 1)
	if _, _, err := f.store.CreateExecutionIntent(f.ctx, dependent); !errors.Is(err, auditstore.ErrPrecondition) {
		t.Fatalf("unaccepted dependency: %v", err)
	}
	execution, _ := f.collecting(t, f.intent("generate-api", 1), "failed")
	if _, _, err := f.store.CreateExecutionIntent(f.ctx, f.intent("generate-api", 2)); !errors.Is(err, auditstore.ErrPrecondition) {
		t.Fatalf("uncollected retry: %v", err)
	}
	if _, _, err := f.store.Collect(f.ctx, f.collection(execution, nil)); err != nil {
		t.Fatal(err)
	}
	if _, _, err := f.store.CreateExecutionIntent(f.ctx, f.intent("generate-api", 3)); !errors.Is(err, auditstore.ErrPrecondition) {
		t.Fatalf("attempt budget: %v", err)
	}
	execution, outputs := f.collecting(t, f.intent("generate-api", 2), "succeeded")
	if _, _, err := f.store.Collect(f.ctx, f.collection(execution, outputs)); err != nil {
		t.Fatal(err)
	}
	accepted, err := f.store.GetPreparationOutput(f.ctx, f.audit.AuditID, "generate-api", "api")
	if err != nil || accepted.RoleAttempt != 2 || accepted.ExecutionID != execution.ExecutionID {
		t.Fatalf("accepted attempt: %+v %v", accepted, err)
	}
	dependent.Preparation.Inputs["existing_openapi"] = accepted.Output.Source
	if _, _, err := f.store.CreateExecutionIntent(f.ctx, dependent); !errors.Is(err, auditstore.ErrPrecondition) {
		t.Fatalf("RunScope instead of retained dependency: %v", err)
	}
	dependent.Preparation.Inputs["existing_openapi"] = accepted.Output.Retained
	if _, _, err := f.store.CreateExecutionIntent(f.ctx, dependent); err != nil {
		t.Fatal(err)
	}
	if _, err := f.store.CompletePreparation(f.ctx, f.claim, f.current(t).Revision); !errors.Is(err, auditstore.ErrPrecondition) {
		t.Fatalf("incomplete roles: %v", err)
	}
}

func TestPostgresAuditPreparationRetentionAndLifecycleWithoutRound(t *testing.T) {
	f := newPreparationFixture(t, false)
	execution, outputs := f.collecting(t, f.intent("generate-api", 1), "succeeded")
	audit := f.current(t)
	paused, _, err := f.store.Transition(f.ctx, auditstore.TransitionParams{OwnerID: audit.OwnerID, AuditID: audit.AuditID,
		ExpectedRevision: audit.Revision, ExpectedState: audit.State, TargetState: auditstore.AuditPaused, IdempotencyKey: "pause", RequestDigest: prepDigest("pause")})
	if err != nil || paused.Phase != auditdomain.AuditPhasePreparing || paused.Hold != auditstore.HoldHeld {
		t.Fatalf("pause: %+v %v", paused, err)
	}
	if _, _, err := f.store.Collect(f.ctx, f.collection(execution, outputs)); err != nil {
		t.Fatal(err)
	}
	if _, err := f.store.CompletePreparation(f.ctx, f.claim, f.current(t).Revision); !errors.Is(err, auditstore.ErrPrecondition) {
		t.Fatalf("paused inventory acceptance: %v", err)
	}
	if ids, err := f.store.ListHeldAuditIDsByLLMCredential(f.ctx, "preparation-credential", 10); err != nil || len(ids) != 1 {
		t.Fatalf("preparation credential hold: %v %v", ids, err)
	}
	beforeDelete := f.current(t)
	if err := runstore.NewPostgresStore(f.pool).DeleteReleasedTerminalRun(f.ctx, f.audit.OwnerID, *execution.RunID); err != nil {
		t.Fatalf("delete collected source Run: %v", err)
	}
	if f.current(t).Revision <= beforeDelete.Revision {
		t.Fatal("Run deletion did not invalidate Audit projection")
	}
	for _, output := range outputs {
		accepted, err := f.store.GetPreparationOutput(f.ctx, f.audit.AuditID, "generate-api", output.LogicalName)
		if err != nil || !reflect.DeepEqual(accepted.Output, output) {
			t.Fatalf("retained output after source deletion: %+v %v", accepted, err)
		}
		read, err := NewPostgresRepository(f.pool).Read(f.ctx, mustProjectScope(t, f.audit.ProjectID), output.Retained.Ref)
		if err != nil || auditdomain.DigestBytes(read.Payload.Data) != output.Retained.Digest {
			t.Fatalf("retained content: %v", err)
		}
		project, _ := f.artifacts.Project(f.audit.ProjectID)
		mutable := output.Retained.Ref
		mutable.Revision = nil
		if _, err := project.Write(f.ctx, mutable, Payload{MediaType: "application/json", Data: []byte("changed")}, output.Retained.Ref.Revision); !errors.Is(err, ErrArtifactFrozen) {
			t.Fatalf("retained binding was mutable: %v", err)
		}
	}
	audit = f.current(t)
	resumed, _, err := f.store.Resume(f.ctx, auditstore.ResumeParams{OwnerID: audit.OwnerID, AuditID: audit.AuditID,
		ExpectedRevision: audit.Revision, DeadlineAt: audit.DeadlineAt, IdempotencyKey: "resume", RequestDigest: prepDigest("resume")})
	if err != nil || resumed.Phase != auditdomain.AuditPhasePreparing {
		t.Fatalf("resume: %+v %v", resumed, err)
	}
	cancelling, _, err := f.store.Transition(f.ctx, auditstore.TransitionParams{OwnerID: resumed.OwnerID, AuditID: resumed.AuditID,
		ExpectedRevision: resumed.Revision, ExpectedState: resumed.State, TargetState: auditstore.AuditCancelling, IdempotencyKey: "cancel", RequestDigest: prepDigest("cancel")})
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := f.store.CreateExecutionIntent(f.ctx, f.intent("generate-api", 2)); !errors.Is(err, auditstore.ErrPrecondition) {
		t.Fatalf("cancelled admission: %v", err)
	}
	if _, released, err := f.store.ReleaseDispatchHold(f.ctx, f.claim); err != nil || !released {
		t.Fatalf("release preparation hold: %t %v", released, err)
	}
	if ids, err := f.store.ListHeldAuditIDsByLLMCredential(f.ctx, "preparation-credential", 10); err != nil || len(ids) != 0 {
		t.Fatalf("leaked credential hold: %v %v", ids, err)
	}
	cancelling = f.current(t)
	cancelled, err := f.store.TransitionClaimed(f.ctx, auditstore.ClaimedTransitionParams{Claim: f.claim, ExpectedRevision: cancelling.Revision,
		ExpectedState: auditstore.AuditCancelling, TargetState: auditstore.AuditCancelled})
	if err != nil {
		t.Fatal(err)
	}
	deleting, _, err := f.store.RequestDelete(f.ctx, auditstore.DeleteParams{OwnerID: cancelled.OwnerID, AuditID: cancelled.AuditID,
		ExpectedRevision: cancelled.Revision, IdempotencyKey: "delete", RequestDigest: prepDigest("delete")})
	if err != nil || deleting.Phase != auditdomain.AuditPhasePreparing {
		t.Fatalf("delete: %+v %v", deleting, err)
	}
	if err := f.store.PurgeClaimed(f.ctx, f.claim, auditdomain.ArtifactNamespace(f.audit.AuditID)); err != nil {
		t.Fatal(err)
	}
	var count int
	if err := f.pool.QueryRow(f.ctx, `SELECT count(*) FROM artifact_bindings WHERE namespace=$1`, auditdomain.ArtifactNamespace(f.audit.AuditID)).Scan(&count); err != nil || count != 0 {
		t.Fatalf("leaked preparation bindings: %d %v", count, err)
	}
}

func TestPostgresAuditPreparationDeleteFencesAnUndispatchedIntent(t *testing.T) {
	f := newPreparationFixture(t, false)
	execution, _, err := f.store.CreateExecutionIntent(f.ctx, f.intent("generate-api", 1))
	if err != nil {
		t.Fatal(err)
	}
	audit := f.current(t)
	deleting, _, err := f.store.RequestDelete(f.ctx, auditstore.DeleteParams{OwnerID: audit.OwnerID, AuditID: audit.AuditID,
		ExpectedRevision: audit.Revision, IdempotencyKey: "delete", RequestDigest: prepDigest("delete")})
	if err != nil || deleting.State != auditstore.AuditCancelling || deleting.Phase != auditdomain.AuditPhasePreparing {
		t.Fatalf("delete fence: %+v %v", deleting, err)
	}
	if audit, released, err := f.store.ReleaseDispatchHold(f.ctx, f.claim); err != nil || released || audit.Hold != auditstore.HoldHeld {
		t.Fatalf("unresolved intent lost hold: %+v %t %v", audit, released, err)
	}
	if _, _, err := f.store.CreateExecutionIntent(f.ctx, f.intent("generate-api", 2)); !errors.Is(err, auditstore.ErrPrecondition) {
		t.Fatalf("admitted through delete fence: %v", err)
	}
	execution, err = f.store.ObserveSubmissionFailure(f.ctx, auditstore.ObserveSubmissionFailureParams{Claim: f.claim, ExecutionID: execution.ExecutionID})
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := f.store.Collect(f.ctx, f.collection(execution, nil)); err != nil {
		t.Fatal(err)
	}
	if _, released, err := f.store.ReleaseDispatchHold(f.ctx, f.claim); err != nil || !released {
		t.Fatalf("drained hold: %t %v", released, err)
	}
	if err := f.store.PurgeClaimed(f.ctx, f.claim, auditdomain.ArtifactNamespace(f.audit.AuditID)); !errors.Is(err, auditstore.ErrPrecondition) {
		t.Fatalf("purged before lifecycle transition: %v", err)
	}
	if _, err := f.store.TransitionClaimed(f.ctx, auditstore.ClaimedTransitionParams{Claim: f.claim, ExpectedRevision: f.current(t).Revision,
		ExpectedState: auditstore.AuditCancelling, TargetState: auditstore.AuditDeleting}); err != nil {
		t.Fatal(err)
	}
	if err := f.store.PurgeClaimed(f.ctx, f.claim, auditdomain.ArtifactNamespace(f.audit.AuditID)); err != nil {
		t.Fatal(err)
	}
}

func TestPostgresAuditPreparationRejectsObsoleteSnapshotsAndInventoryBaseline(t *testing.T) {
	f := newPreparationFixture(t, false)
	current, err := os.ReadFile("../persistence/postgres/testdata/audit-profile-current.json")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := config.DecodeResolvedAuditProfileSnapshot(current); err != nil {
		t.Fatalf("migration fixture is not a current-schema closure: %v", err)
	}
	for _, obsolete := range []bool{false, true} {
		raw := f.audit.ProfileSnapshot
		id := "invalid-baseline"
		if obsolete {
			raw = json.RawMessage(`{"name":"old-profile","workflows":{"generate-api":{"ref":"openapi-from-workspace@7"}}}`)
			id = "obsolete-profile"
		}
		audit, _, err := f.store.CreateDraft(f.ctx, auditstore.CreateDraftParams{AuditID: id, OwnerID: f.audit.OwnerID, ProjectID: f.audit.ProjectID,
			Profile: f.audit.Profile, ProfileSnapshot: raw, InputSelection: json.RawMessage(`{}`), Limits: f.audit.Limits,
			IdempotencyKey: id, RequestDigest: prepDigest(id)})
		if err != nil {
			t.Fatal(err)
		}
		p := f.start
		p.AuditID = id
		p.ExpectedRevision = audit.Revision
		p.IdempotencyKey = id
		if !obsolete {
			var baseline map[string]any
			_ = json.Unmarshal(p.BaselineSnapshot, &baseline)
			baseline["inventory"] = map[string]any{}
			p.BaselineSnapshot, _ = json.Marshal(baseline)
		}
		if _, _, err := f.store.StartPreparation(f.ctx, p); !errors.Is(err, auditstore.ErrInvalid) {
			t.Fatalf("invalid preparation %s admitted: %v", id, err)
		}
		unchanged, err := f.store.Get(f.ctx, f.audit.OwnerID, id)
		if err != nil || unchanged.State != auditstore.AuditDraft || unchanged.Phase != auditdomain.AuditPhaseNotStarted || unchanged.BaselineSnapshot != nil || !reflect.DeepEqual(unchanged.ProfileSnapshot, audit.ProfileSnapshot) {
			t.Fatalf("invalid snapshot was converted: %+v %v", unchanged, err)
		}
	}
}
