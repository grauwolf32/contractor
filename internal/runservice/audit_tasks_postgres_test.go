package runservice

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/agentskills"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/artifactstest"
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

func TestSharedPostgresCommonTaskCollectsOrdinaryOutputWhilePaused(t *testing.T) {
	for _, semantic := range []bool{false, true} {
		t.Run(map[bool]string{false: "supporting-material", true: "observation-result"}[semantic], func(t *testing.T) {
			f := newCommonTaskRunFixture(t, semantic)
			ctx := t.Context()
			created, err := f.service.CreateAudit(ctx, f.params)
			if err != nil {
				t.Fatal(err)
			}
			art := artifacts.NewService(artifacts.NewPostgresRepository(f.pool))
			run, _ := art.Run(created.Run.RunID)
			payload := artifacts.Payload{MediaType: "text/plain", Data: []byte("ordinary output")}
			if semantic {
				input, err := run.Read(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "context"})
				if err != nil {
					t.Fatal(err)
				}
				var context struct {
					Items []struct {
						Evidence []auditcontract.Material `json:"evidence"`
					}
				}
				if err := json.Unmarshal(input.Payload.Data, &context); err != nil {
					t.Fatal(err)
				}
				data, err := json.Marshal(map[string]any{"schema": "contractor.audit.conclusions.v1", "items": []any{map[string]any{"item_key": "item-00", "rationale": "The selected source was inspected", "tested_conditions": []string{"Exact selected source bytes"}, "limitations": []string{}, "evidence": context.Items[0].Evidence}}})
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
			runs := runstore.NewPostgresStore(f.pool)
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
			access, err := auditimport.NewArtifactAccess(art, hookstore.NewClaimedArtifacts(f.pool))
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
			if err := rundeletion.New(f.pool).DeleteReleasedTerminalRun(ctx, "owner", created.Run.RunID); err != nil {
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

func newCommonTaskRunFixture(t *testing.T, contextInput bool) taskRunFixture {
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
	data, _ := json.Marshal(program)
	pinned, err := auditprofile.PinProgram(resolved.Selector{ID: "common-run", Version: "1"}, data, catalog.Workflow)
	if err != nil {
		t.Fatal(err)
	}
	project, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{ProjectID: "task-run-project", OwnerID: "owner", Kind: projectstore.KindProject, Name: "Common Run", IdempotencyKey: "project", RequestDigest: contentdigest.Bytes([]byte("project"))})
	if err != nil {
		t.Fatal(err)
	}
	art := artifacts.NewService(artifacts.NewPostgresRepository(pool))
	projectArtifacts, _ := art.Project(project.ProjectID)
	source := writeExact(t, ctx, projectArtifacts, "inputs", "source", "text/plain", []byte("exact selected bytes"))
	profile, _ := json.Marshal(pinned)
	limits := program.Limits
	audits := auditstoretest.NewStore(pool)
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
	accepted, _, err := audits.AcceptTasks(ctx, auditstore.AcceptTasksParams{OwnerID: "owner", AuditID: audit.AuditID, ExpectedRevision: audit.Revision, IdempotencyKey: "accept", RequestDigest: contentdigest.Bytes([]byte("accept")), Tasks: []auditstore.AcceptedTask{{Document: document, Source: auditstore.TaskSource{Kind: "manual", Key: "first"}}}})
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
			art := artifacts.NewService(artifacts.NewPostgresRepository(f.pool))
			run, _ := art.Run(created.Run.RunID)
			input, err := run.Read(t.Context(), contracts.ArtifactRef{Namespace: "inputs", Name: "source"})
			if err != nil || string(input.Payload.Data) != "exact selected bytes" {
				t.Fatalf("exact fork: %+v %v", input, err)
			}
			if withContext {
				context, err := run.Read(t.Context(), contracts.ArtifactRef{Namespace: "inputs", Name: "context"})
				if err != nil {
					t.Fatal(err)
				}
				if _, err := auditcontract.Decode(context.Payload.Data, nil); err != nil {
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
				if json.Unmarshal(context.Payload.Data, &value) != nil || value.AuditID != f.audit.AuditID || value.RunID != created.Run.RunID || len(value.Items) != 1 || value.Items[0].TaskID != "ordinary-task" || len(value.Items[0].Evidence) != 1 || value.Items[0].Evidence[0].ScopeKind != "run" || value.Items[0].Evidence[0].ScopeID != created.Run.RunID || !value.Items[0].Evidence[0].Ref.SameExact(input.Ref) {
					t.Fatalf("wrong trusted context: %s", context.Payload.Data)
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
	if _, err := runstore.NewPostgresStore(f.pool).GetRun(t.Context(), "ordinary-task-run"); !errors.Is(err, runstore.ErrNotFound) {
		t.Fatalf("partial Run committed: %v", err)
	}
}
