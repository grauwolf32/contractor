package runservice

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/agentskills"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/artifactstest"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/auditbaseline"
	"github.com/grauwolf32/contractor/internal/auditimport"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
	"github.com/grauwolf32/contractor/internal/auditstore/hookstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/configtest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/persistence/postgrestest"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/rundeletion"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/jackc/pgx/v5/pgxpool"
)

type taskRunFixture struct {
	pool    *pgxpool.Pool
	audits  *auditstoretest.Store
	service *Service
	audit   auditstore.Audit
	params  AuditCreateParams
	source  auditstore.ExactArtifact
}

type taskCollectionReadFault struct {
	auditimport.ArtifactAccess
	missingContext bool
	transient      bool
}

func (a *taskCollectionReadFault) WithClaim(claim auditstore.ControllerClaim) auditimport.ArtifactAccess {
	a.ArtifactAccess = a.ArtifactAccess.WithClaim(claim)
	return a
}

func (a *taskCollectionReadFault) DescribeRunBinding(ctx context.Context, runID string, ref contracts.ArtifactRef) (auditstore.ExactArtifact, bool, error) {
	if a.missingContext && ref.Namespace == "inputs" && ref.Name == "context" {
		return auditstore.ExactArtifact{}, false, artifacts.ErrArtifactNotFound
	}
	if a.transient {
		a.transient = false
		return auditstore.ExactArtifact{}, false, io.ErrUnexpectedEOF
	}
	return a.ArtifactAccess.DescribeRunBinding(ctx, runID, ref)
}

func TestSharedPostgresCommonTaskCollectionFailureOutcomes(t *testing.T) {
	for _, test := range []struct {
		name                                                string
		semantic, output, failed, missingContext, transient bool
		gap                                                 string
		state                                               auditstore.TaskState
	}{
		{name: "invalid-conclusions", semantic: true, output: true, gap: "invalid-result", state: auditstore.TaskSettled},
		{name: "missing-trusted-context", semantic: true, output: true, missingContext: true, gap: "collection-contract-invalid", state: auditstore.TaskSettled},
		{name: "missing-output", gap: "missing-output", state: auditstore.TaskSettled},
		{name: "failed-with-partial-output", output: true, failed: true, state: auditstore.TaskReady},
		{name: "ambiguous-read", output: true, transient: true, state: auditstore.TaskSettled},
	} {
		t.Run(test.name, func(t *testing.T) {
			f := newCommonTaskRunFixture(t, test.semantic)
			ctx := t.Context()
			created, err := f.service.CreateAudit(ctx, f.params)
			if err != nil {
				t.Fatal(err)
			}
			art := artifacts.NewService(artifactpgtest.NewStorage().Repository(f.pool))
			run, _ := art.Run(created.Run.RunID)
			if test.output {
				payload := artifacts.Payload{MediaType: "text/plain", Data: []byte("partial support")}
				if test.semantic {
					payload = artifacts.Payload{MediaType: "application/json", Data: []byte(`{"schema":"unknown"}`)}
				}
				written, err := run.Write(ctx, contracts.ArtifactRef{Namespace: "builder", Name: "result"}, payload, nil)
				if err != nil {
					t.Fatal(err)
				}
				if _, err := artifactstest.BindOutput(ctx, art, created.Run.RunID, "result", written.Ref); err != nil {
					t.Fatal(err)
				}
				if err := artifactstest.FreezeOutputs(ctx, art, created.Run.RunID); err != nil {
					t.Fatal(err)
				}
			}
			runs := runstore.NewPostgresStore(f.pool, artifactpgtest.NewStorage())
			terminal := runstore.RunSucceeded
			if test.failed {
				terminal = runstore.RunFailed
			}
			for _, transition := range [][2]runstore.WorkflowRunState{{runstore.RunPending, runstore.RunRunning}, {runstore.RunRunning, terminal}} {
				if _, err := runs.TransitionRun(ctx, created.Run.RunID, transition[0], transition[1], runstore.Reason{Code: "fixture"}); err != nil {
					t.Fatal(err)
				}
			}
			cursor, err := runs.GetRunEventCursor(ctx, created.Run.RunID)
			if err != nil {
				t.Fatal(err)
			}
			execution, err := f.audits.ObserveTerminal(ctx, auditstore.ObserveTerminalParams{Claim: f.params.Claim, ExecutionID: f.params.ExecutionID, RunID: created.Run.RunID, Generation: cursor.Generation, Sequence: uint64(cursor.Sequence)})
			if err != nil {
				t.Fatal(err)
			}
			audit, err := f.audits.Get(ctx, "owner", f.audit.AuditID)
			if err != nil {
				t.Fatal(err)
			}
			access, err := auditimport.NewArtifactAccess(art, hookstore.NewClaimedArtifacts(f.pool, artifactpgtest.NewStorage()))
			if err != nil {
				t.Fatal(err)
			}
			collector, err := auditimport.New(f.audits, f.audits, runs, &taskCollectionReadFault{ArtifactAccess: access, missingContext: test.missingContext, transient: test.transient})
			if err != nil {
				t.Fatal(err)
			}
			snapshot := auditstore.ReconcileSnapshot{Audit: audit}
			if test.transient {
				if changed, err := collector.Collect(ctx, f.params.Claim, snapshot, execution); changed || !errors.Is(err, io.ErrUnexpectedEOF) {
					t.Fatalf("read ambiguity became a semantic gap: %v %v", changed, err)
				}
				if _, err := f.audits.CollectionReceipt(ctx, audit.AuditID, execution.ExecutionID); !errors.Is(err, auditstore.ErrNotFound) {
					t.Fatalf("ambiguous read committed a receipt: %v", err)
				}
			}
			if changed, err := collector.Collect(ctx, f.params.Claim, snapshot, execution); err != nil || !changed {
				t.Fatalf("collection: %v %v", changed, err)
			}
			task, err := f.audits.GetTask(ctx, "owner", audit.AuditID, "ordinary-task")
			if err != nil || task.State != test.state || test.gap != "" && (task.GapCode == nil || *task.GapCode != test.gap) {
				t.Fatalf("failure outcome: %+v %v", task, err)
			}
			if test.failed {
				link, err := f.audits.GetArtifactLink(ctx, audit.AuditID, "execution/"+execution.ExecutionID+"/output/material")
				if err != nil || link.Artifact.Digest != contentdigest.Bytes([]byte("partial support")) {
					t.Fatalf("partial output was lost: %+v %v", link, err)
				}
			}
			if changed, err := collector.Collect(ctx, f.params.Claim, snapshot, execution); err != nil || changed {
				t.Fatalf("permanent outcome was retried: %v %v", changed, err)
			}
		})
	}
}

func TestSharedPostgresCommonTaskCollectsOrdinaryOutputWhilePaused(t *testing.T) {
	for _, semantic := range []bool{false, true} {
		t.Run(map[bool]string{false: "supporting-material", true: "observation-result"}[semantic], func(t *testing.T) {
			f := newCommonTaskRunFixture(t, semantic)
			ctx := t.Context()
			created, err := f.service.CreateAudit(ctx, f.params)
			if err != nil {
				t.Fatal(err)
			}
			art := artifacts.NewService(artifactpgtest.NewStorage().Repository(f.pool))
			run, _ := art.Run(created.Run.RunID)
			payload := artifacts.Payload{MediaType: "text/plain", Data: []byte("ordinary output")}
			if semantic {
				input, err := run.Read(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "context"})
				if err != nil {
					t.Fatal(err)
				}
				var trustedContext struct {
					Items []struct {
						Evidence []auditcontract.Material `json:"evidence"`
					}
				}
				if err := json.Unmarshal(input.Payload.Data, &trustedContext); err != nil {
					t.Fatal(err)
				}
				data, err := json.Marshal(map[string]any{"schema": "contractor.audit.conclusions.v1", "items": []any{map[string]any{"item_key": "item-00", "rationale": "The selected source was inspected", "tested_conditions": []string{"Exact selected source bytes"}, "limitations": []string{}, "evidence": trustedContext.Items[0].Evidence}}})
				if err != nil {
					t.Fatal(err)
				}
				if _, err := auditcontract.Decode(data, input.Payload.Data); err != nil {
					t.Fatalf("invalid authored fixture: %v", err)
				}
				payload = artifacts.Payload{MediaType: "application/json", Data: data}
			}
			written, err := run.Write(ctx, contracts.ArtifactRef{Namespace: "builder", Name: "result"}, payload, nil)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := artifactstest.BindOutput(ctx, art, created.Run.RunID, "result", written.Ref); err != nil {
				t.Fatal(err)
			}
			if err := artifactstest.FreezeOutputs(ctx, art, created.Run.RunID); err != nil {
				t.Fatal(err)
			}
			audit, err := f.audits.Get(ctx, "owner", f.audit.AuditID)
			if err != nil {
				t.Fatal(err)
			}
			if _, _, err := f.audits.Transition(ctx, auditstore.TransitionParams{OwnerID: "owner", AuditID: audit.AuditID, ExpectedRevision: audit.Revision, ExpectedState: auditstore.AuditActive, TargetState: auditstore.AuditPaused, IdempotencyKey: "pause", RequestDigest: contentdigest.Bytes([]byte("pause"))}); err != nil {
				t.Fatal(err)
			}
			runs := runstore.NewPostgresStore(f.pool, artifactpgtest.NewStorage())
			if _, err := runs.TransitionRun(ctx, created.Run.RunID, runstore.RunPending, runstore.RunRunning, runstore.Reason{Code: "admitted"}); err != nil {
				t.Fatal(err)
			}
			if _, err := runs.TransitionRun(ctx, created.Run.RunID, runstore.RunRunning, runstore.RunSucceeded, runstore.Reason{Code: "completed"}); err != nil {
				t.Fatal(err)
			}
			cursor, err := runs.GetRunEventCursor(ctx, created.Run.RunID)
			if err != nil {
				t.Fatal(err)
			}
			execution, err := f.audits.ObserveTerminal(ctx, auditstore.ObserveTerminalParams{Claim: f.params.Claim, ExecutionID: f.params.ExecutionID, RunID: created.Run.RunID, Generation: cursor.Generation, Sequence: uint64(cursor.Sequence)})
			if err != nil {
				t.Fatal(err)
			}
			audit, err = f.audits.Get(ctx, "owner", f.audit.AuditID)
			if err != nil {
				t.Fatal(err)
			}
			access, err := auditimport.NewArtifactAccess(art, hookstore.NewClaimedArtifacts(f.pool, artifactpgtest.NewStorage()))
			if err != nil {
				t.Fatal(err)
			}
			importer, err := auditimport.New(f.audits, f.audits, runs, access)
			if err != nil {
				t.Fatal(err)
			}
			snapshot := auditstore.ReconcileSnapshot{Audit: audit}
			if changed, err := importer.Collect(ctx, f.params.Claim, snapshot, execution); err != nil || !changed {
				t.Fatalf("paused collection: %v %v", changed, err)
			}
			task, err := f.audits.GetTask(ctx, "owner", audit.AuditID, "ordinary-task")
			if err != nil || task.State != auditstore.TaskSettled || task.Result == nil || task.GapCode != nil {
				t.Fatalf("retained task: %+v %v", task, err)
			}
			scope, _ := artifacts.AuditScope(audit.AuditID)
			retained, _ := art.Scoped(scope)
			result, err := retained.Read(ctx, task.Result.Ref)
			if err != nil {
				t.Fatal(err)
			}
			if semantic {
				var doc struct {
					Schema string
					Items  []struct {
						Origin struct {
							AuditID string `json:"audit_id"`
							TaskID  string `json:"task_id"`
							RunID   string `json:"run_id"`
						}
						Evidence []auditcontract.Material
					}
				}
				if json.Unmarshal(result.Payload.Data, &doc) != nil || doc.Schema != "contractor.audit.result.v1" || len(doc.Items) != 1 || doc.Items[0].Origin.AuditID != audit.AuditID || doc.Items[0].Origin.TaskID != "ordinary-task" || doc.Items[0].Origin.RunID != created.Run.RunID || len(doc.Items[0].Evidence) != 1 || doc.Items[0].Evidence[0].ScopeKind != "audit" || doc.Items[0].Evidence[0].ScopeID != audit.AuditID {
					t.Fatalf("invalid retained result provenance: %s", result.Payload.Data)
				}
			} else if string(result.Payload.Data) != "ordinary output" {
				t.Fatalf("supporting output was reinterpreted: %s", result.Payload.Data)
			}
			if err := rundeletion.New(f.pool, artifactpgtest.NewStorage()).DeleteReleasedTerminalRun(ctx, "owner", created.Run.RunID); err != nil {
				t.Fatal(err)
			}
			// A lost acknowledgement is resolved before touching the purged Run.
			if changed, err := importer.Collect(ctx, f.params.Claim, snapshot, execution); err != nil || changed {
				t.Fatalf("collection replay after source purge: %v %v", changed, err)
			}
			if _, err := retained.Read(ctx, task.Result.Ref); err != nil {
				t.Fatalf("retained result lost its bytes: %v", err)
			}
			after, err := f.audits.Get(ctx, "owner", audit.AuditID)
			if err != nil || after.State != auditstore.AuditPaused || after.SubmittedRunCount != 1 || after.OutstandingRunCount != 0 {
				t.Fatalf("paused outcome changed: %+v %v", after, err)
			}
		})
	}
}

func newCommonTaskRunFixture(t *testing.T, contextInput bool, memberCounts ...int) taskRunFixture {
	t.Helper()
	return newCommonTaskRunFixtureWithWorkflow(t, contextInput, nil, memberCounts...)
}
func newCommonTaskRunFixtureWithWorkflow(t *testing.T, contextInput bool, configure func(*resolved.ResolvedWorkflow), memberCounts ...int) taskRunFixture {
	t.Helper()
	ctx := t.Context()
	pool := postgrestest.NewPool(t, ctx, "", postgrestest.Options{Configure: func(config *pgxpool.Config) { config.MaxConns = 1 }})
	if _, err := postgres.ApplyMigrations(ctx, pool, postgres.MigrationBudgets{}); err != nil {
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
	var fixture struct{ Cases []struct{ Document string } }
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "audit-common", "programs.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	program, err := auditcontract.DecodeProgram([]byte(fixture.Cases[0].Document))
	if err != nil {
		t.Fatal(err)
	}
	if contextInput {
		binding := program.Workflows["inspect"]
		binding.Ref = "audit-contract-check@1"
		binding.Inputs["context"] = auditcontract.InputMapping{Source: "task-context"}
		binding.Outputs = map[string]string{"result": "result"}
		program.Workflows["inspect"] = binding
	}
	if configure != nil {
		workflow, err := catalog.Workflow(program.Workflows["inspect"].Ref)
		if err != nil {
			t.Fatal(err)
		}
		configure(&workflow)
		if workflow.Inputs["source"].MaterialSet {
			slot := program.Inputs["source"]
			slot.MediaTypes = []string{"application/json"}
			program.Inputs["source"] = slot
		}
	}
	data, _ := json.Marshal(program)
	pinned, err := auditprofile.PinProgram(resolved.Selector{ID: "common-run", Version: "1"}, data, func(ref string) (resolved.ResolvedWorkflow, error) {
		workflow, err := catalog.Workflow(ref)
		if err == nil && configure != nil {
			configure(&workflow)
		}
		return workflow, err
	})
	if err != nil {
		t.Fatal(err)
	}
	project, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{ProjectID: "task-run-project", OwnerID: "owner", Kind: projectstore.KindProject, Name: "Common Run", IdempotencyKey: "project", RequestDigest: contentdigest.Bytes([]byte("project"))})
	if err != nil {
		t.Fatal(err)
	}
	art := artifacts.NewService(artifactpgtest.NewStorage().Repository(pool))
	projectArtifacts, _ := art.Project(project.ProjectID)
	source := writeExact(t, ctx, projectArtifacts, "inputs", "source", "text/plain", []byte("exact selected bytes"))
	if pinned.Workflows["inspect"].Inputs["source"].MaterialSet {
		raw, err := json.Marshal(contracts.MaterialSet{Schema: contracts.MaterialSetSchema, Materials: []contracts.MaterialDescriptor{{ScopeKind: "project", ScopeID: project.ProjectID, Ref: source.Ref, Digest: source.Digest, MediaType: source.MediaType, SizeBytes: source.SizeBytes}}})
		if err != nil {
			t.Fatal(err)
		}
		source = writeExact(t, ctx, projectArtifacts, "inputs", "selection", "application/json", raw)
	}

	profile, _ := json.Marshal(pinned)
	limits := program.Limits
	audits := auditstoretest.NewStore(pool, artifactpgtest.NewStorage())
	audit, _, err := audits.CreateDraft(ctx, auditstore.CreateDraftParams{AuditID: "common-run-audit", OwnerID: "owner", ProjectID: project.ProjectID, Profile: auditstore.ProfileIdentity{Name: pinned.Ref.Name, Version: pinned.Ref.Version, Digest: pinned.Ref.Digest}, ProfileSnapshot: profile, InputSelection: json.RawMessage(`{}`), Limits: auditstore.Limits{MaxRounds: limits.MaxIterations, BatchSize: limits.BatchSize, MaxItemsPerRound: limits.MaxTasksPerIteration, MaxItemsTotal: limits.MaxTasks, MaxSubmittedRuns: limits.MaxSubmittedRuns, MaxItemRunAttempts: limits.MaxAttempts, MaxEvidenceBytes: limits.MaxEvidenceBytes}, IdempotencyKey: "draft", RequestDigest: contentdigest.Bytes([]byte("draft"))})
	if err != nil {
		t.Fatal(err)
	}
	baseline, _ := json.Marshal(auditbaseline.BaselineSnapshot{Schema: auditbaseline.Schema, Labels: map[string]string{}, RequiredPlacementLabels: []string{}, Inputs: map[string]auditstore.ExactArtifact{"source": source}, ActiveChecks: contracts.ActiveChecksProhibited, RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot(), RuntimeLabels: []string{}, Skills: []agentskills.RunSkillSnapshot{}, LLMCredentialIDs: []string{}, RuntimeCredentialIDs: []string{}, Standards: []auditstandards.PinnedPackage{}})
	if _, err := pool.Exec(ctx, `UPDATE audits SET state='active',phase='rounds',started_at=clock_timestamp(),baseline_snapshot=$2 WHERE audit_id=$1`, audit.AuditID, baseline); err != nil {
		t.Fatal(err)
	}
	projectScope, _ := artifacts.ProjectScope(project.ProjectID)
	auditScope, _ := artifacts.AuditScope(audit.AuditID)
	fork, err := art.Fork(ctx, artifacts.Fork{Kind: artifacts.LineageAuditInput, Source: projectScope, SourceRef: source.Ref, Target: auditScope, TargetRef: contracts.ArtifactRef{Namespace: "inputs", Name: "selected"}, Frozen: true})
	if err != nil {
		t.Fatal(err)
	}
	material := auditcontract.Material{ScopeKind: "audit", ScopeID: audit.AuditID, Ref: fork.TargetRef, Digest: source.Digest, MediaType: source.MediaType, SizeBytes: source.SizeBytes}
	document := auditcontract.Task{Schema: auditcontract.TaskSchema, AuditID: audit.AuditID, TaskID: "ordinary-task", Workflow: "inspect", Inputs: map[string]auditcontract.Material{"source": material}, Parameters: map[string]string{}, Objective: "Inspect exact source"}
	members := []auditstore.AcceptedTask{{Document: document, Source: auditstore.TaskSource{Kind: "manual", Key: "first"}}}
	if len(memberCounts) > 0 {
		for index := 1; index < memberCounts[0]; index++ {
			next := document
			next.TaskID, next.Objective = fmt.Sprintf("ordinary-task-%d", index), fmt.Sprintf("Inspect condition %d", index)
			members = append(members, auditstore.AcceptedTask{Document: next, Source: auditstore.TaskSource{Kind: "manual", Key: fmt.Sprintf("member-%d", index)}})
		}
	}
	accepted, _, err := audits.AcceptTasks(ctx, auditstore.AcceptTasksParams{OwnerID: "owner", AuditID: audit.AuditID, ExpectedRevision: audit.Revision, IdempotencyKey: "accept", RequestDigest: contentdigest.Bytes([]byte("accept")), Tasks: members})
	if err != nil {
		t.Fatal(err)
	}
	claimed, err := audits.Claim(ctx, auditstore.ClaimParams{HolderID: "task-run-controller", Lease: time.Minute, Limit: 1})
	if err != nil || len(claimed) != 1 {
		t.Fatalf("claim: %+v %v", claimed, err)
	}
	execution, _, err := audits.SelectTasks(ctx, auditstore.SelectTasksParams{Claim: claimed[0], TaskIDs: accepted.TaskIDs})
	if err != nil {
		t.Fatal(err)
	}
	service := &Service{auditTransaction: postgresAuditTransaction(pool), credentialGuard: openCredentialGuard{}}
	params := AuditCreateParams{Claim: claimed[0], ExecutionID: execution.ExecutionID, TaskIDs: accepted.TaskIDs, RequestDigest: execution.RequestDigest, NewRunID: func() (string, error) { return "ordinary-task-run", nil }}
	return taskRunFixture{pool: pool, audits: audits, service: service, audit: audit, params: params, source: source}
}

func TestSharedPostgresCommonTaskCreatesOrdinaryRunAndReplays(t *testing.T) {
	for _, withContext := range []bool{false, true} {
		t.Run(map[bool]string{false: "ordinary-material", true: "trusted-context"}[withContext], func(t *testing.T) {
			f := newCommonTaskRunFixture(t, withContext)
			created, err := f.service.CreateAudit(t.Context(), f.params)
			if err != nil || !created.Created || created.Run.State != runstore.RunPending {
				t.Fatalf("create ordinary Run: %+v %v", created, err)
			}
			f.params.NewRunID = func() (string, error) { t.Fatal("replay minted a Run identity"); return "", nil }
			replay, err := f.service.CreateAudit(t.Context(), f.params)
			if err != nil || !replay.Replayed || replay.Run.RunID != created.Run.RunID {
				t.Fatalf("replay: %+v %v", replay, err)
			}
			art := artifacts.NewService(artifactpgtest.NewStorage().Repository(f.pool))
			run, _ := art.Run(created.Run.RunID)
			input, err := run.Read(t.Context(), contracts.ArtifactRef{Namespace: "inputs", Name: "source"})
			if err != nil || string(input.Payload.Data) != "exact selected bytes" {
				t.Fatalf("exact fork: %+v %v", input, err)
			}
			if withContext {
				trustedContext, err := run.Read(t.Context(), contracts.ArtifactRef{Namespace: "inputs", Name: "context"})
				if err != nil {
					t.Fatal(err)
				}
				if _, err := auditcontract.Decode(trustedContext.Payload.Data, nil); err != nil {
					t.Fatal(err)
				}
				var value struct {
					AuditID string `json:"audit_id"`
					RunID   string `json:"run_id"`
					Items   []struct {
						TaskID   string                   `json:"task_id"`
						Evidence []auditcontract.Material `json:"evidence"`
					}
				}
				if json.Unmarshal(trustedContext.Payload.Data, &value) != nil || value.AuditID != f.audit.AuditID || value.RunID != created.Run.RunID || len(value.Items) != 1 || value.Items[0].TaskID != "ordinary-task" || len(value.Items[0].Evidence) != 1 || value.Items[0].Evidence[0].ScopeKind != "run" || value.Items[0].Evidence[0].ScopeID != created.Run.RunID || !value.Items[0].Evidence[0].Ref.SameExact(input.Ref) {
					t.Fatalf("wrong trusted context: %s", trustedContext.Payload.Data)
				}
			}
			audit, err := f.audits.Get(t.Context(), "owner", f.audit.AuditID)
			if err != nil || audit.ReservedRunCount != 1 || audit.SubmittedRunCount != 1 {
				t.Fatalf("duplicate reservation or submission: %+v %v", audit, err)
			}
			var rounds int
			if err := f.pool.QueryRow(t.Context(), `SELECT count(*) FROM audit_rounds WHERE audit_id=$1`, f.audit.AuditID).Scan(&rounds); err != nil || rounds != 0 {
				t.Fatalf("invented legacy Round: %d %v", rounds, err)
			}
		})
	}
}

func TestSharedPostgresCommonTaskBatchKeepsMembershipAndChargesEvidenceOnce(t *testing.T) {
	for _, native := range []bool{false, true} {
		t.Run(fmt.Sprint("material-set=", native), func(t *testing.T) {
			f := newCommonTaskRunFixtureWithWorkflow(t, true, func(workflow *resolved.ResolvedWorkflow) {
				if native {
					slot := workflow.Inputs["source"]
					slot.MaterialSet = true
					workflow.Inputs["source"] = slot
				}
			}, 2)
			ctx := t.Context()
			created, err := f.service.CreateAudit(ctx, f.params)
			if err != nil {
				t.Fatal(err)
			}
			art := artifacts.NewService(artifactpgtest.NewStorage().Repository(f.pool))
			run, _ := art.Run(created.Run.RunID)
			trusted, err := run.Read(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "context"})
			if err != nil {
				t.Fatal(err)
			}
			var value struct {
				Items []struct {
					ItemKey  string                   `json:"item_key"`
					TaskID   string                   `json:"task_id"`
					Evidence []auditcontract.Material `json:"evidence"`
				}
			}
			if json.Unmarshal(trusted.Payload.Data, &value) != nil || len(value.Items) != 2 || value.Items[0].TaskID == value.Items[1].TaskID {
				t.Fatalf("lost batch context membership: %s", trusted.Payload.Data)
			}
			for _, member := range value.Items {
				for _, evidence := range member.Evidence {
					actual, err := run.ReadExact(ctx, evidence.Exact())
					if err != nil || evidence.Exact().Verify(actual.Exact()) != nil {
						t.Fatalf("context metadata does not describe local bytes: %+v actual=%+v err=%v", evidence, actual, err)
					}
					if native && evidence.Digest == f.source.Digest {
						t.Fatal("material set was not transformed")
					}
				}
			}
			items := []any{}
			for index, member := range value.Items {
				items = append(items, map[string]any{"item_key": member.ItemKey, "rationale": fmt.Sprintf("Inspected condition %d", index), "tested_conditions": []string{fmt.Sprintf("Condition %d", index)}, "limitations": []string{}, "evidence": member.Evidence})
			}
			data, _ := json.Marshal(map[string]any{"schema": "contractor.audit.conclusions.v1", "items": items})
			written, err := run.Write(ctx, contracts.ArtifactRef{Namespace: "builder", Name: "result"}, artifacts.Payload{MediaType: "application/json", Data: data}, nil)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := artifactstest.BindOutput(ctx, art, created.Run.RunID, "result", written.Ref); err != nil {
				t.Fatal(err)
			}
			if err := artifactstest.FreezeOutputs(ctx, art, created.Run.RunID); err != nil {
				t.Fatal(err)
			}
			runs := runstore.NewPostgresStore(f.pool, artifactpgtest.NewStorage())
			for _, transition := range [][2]runstore.WorkflowRunState{{runstore.RunPending, runstore.RunRunning}, {runstore.RunRunning, runstore.RunSucceeded}} {
				if _, err := runs.TransitionRun(ctx, created.Run.RunID, transition[0], transition[1], runstore.Reason{Code: "fixture"}); err != nil {
					t.Fatal(err)
				}
			}
			cursor, err := runs.GetRunEventCursor(ctx, created.Run.RunID)
			if err != nil {
				t.Fatal(err)
			}
			execution, err := f.audits.ObserveTerminal(ctx, auditstore.ObserveTerminalParams{Claim: f.params.Claim, ExecutionID: f.params.ExecutionID, RunID: created.Run.RunID, Generation: cursor.Generation, Sequence: uint64(cursor.Sequence)})
			if err != nil {
				t.Fatal(err)
			}
			audit, err := f.audits.Get(ctx, "owner", f.audit.AuditID)
			if err != nil {
				t.Fatal(err)
			}
			access, err := auditimport.NewArtifactAccess(art, hookstore.NewClaimedArtifacts(f.pool, artifactpgtest.NewStorage()))
			if err != nil {
				t.Fatal(err)
			}
			collector, err := auditimport.New(f.audits, f.audits, runs, access)
			if err != nil {
				t.Fatal(err)
			}
			if changed, err := collector.Collect(ctx, f.params.Claim, auditstore.ReconcileSnapshot{Audit: audit}, execution); err != nil || !changed {
				t.Fatalf("batch collection: %v %v", changed, err)
			}
			page, err := f.audits.ListTasks(ctx, auditstore.ListTasksParams{OwnerID: "owner", AuditID: audit.AuditID, Limit: 10})
			if err != nil || len(page.Tasks) != 2 {
				t.Fatalf("batch Tasks: %+v %v", page, err)
			}
			for _, task := range page.Tasks {
				if task.State != auditstore.TaskSettled || task.Result == nil || task.GapCode != nil {
					t.Fatalf("invalid batch outcome: %+v", task)
				}
			}
			if !page.Tasks[0].Result.Ref.SameExact(page.Tasks[1].Result.Ref) {
				t.Fatal("one batch produced different retained output identities")
			}
			scope, _ := artifacts.AuditScope(audit.AuditID)
			retained, _ := art.Scoped(scope)
			result, err := retained.Read(ctx, page.Tasks[0].Result.Ref)
			if err != nil {
				t.Fatal(err)
			}
			var normalized struct {
				Items []struct {
					Origin struct {
						TaskID string `json:"task_id"`
					}
					Rationale string
				}
			}
			if json.Unmarshal(result.Payload.Data, &normalized) != nil || len(normalized.Items) != 2 || normalized.Items[0].Origin.TaskID != value.Items[0].TaskID || normalized.Items[1].Origin.TaskID != value.Items[1].TaskID || normalized.Items[0].Rationale == normalized.Items[1].Rationale {
				t.Fatalf("conclusion copied between members: %s", result.Payload.Data)
			}
			after, err := f.audits.Get(ctx, "owner", audit.AuditID)
			wantBytes := int64(len(data)+len(result.Payload.Data)) + f.source.SizeBytes
			if err != nil || after.RetainedEvidenceBytes != wantBytes {
				t.Fatalf("shared evidence counted incorrectly: %d want %d: %v", after.RetainedEvidenceBytes, wantBytes, err)
			}
		})
	}
}

func TestSharedPostgresCommonTaskRunCreationRespectsPause(t *testing.T) {
	f := newCommonTaskRunFixture(t, false)
	audit, err := f.audits.Get(t.Context(), "owner", f.audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := f.audits.Transition(t.Context(), auditstore.TransitionParams{OwnerID: "owner", AuditID: audit.AuditID, ExpectedRevision: audit.Revision, ExpectedState: auditstore.AuditActive, TargetState: auditstore.AuditPaused, IdempotencyKey: "pause", RequestDigest: contentdigest.Bytes([]byte("pause"))}); err != nil {
		t.Fatal(err)
	}
	if _, err := f.service.CreateAudit(t.Context(), f.params); !errors.Is(err, auditstore.ErrPrecondition) {
		t.Fatalf("paused submission: %v", err)
	}
	if _, err := runstore.NewPostgresStore(f.pool, artifactpgtest.NewStorage()).GetRun(t.Context(), "ordinary-task-run"); !errors.Is(err, runstore.ErrNotFound) {
		t.Fatalf("partial Run committed: %v", err)
	}
}
