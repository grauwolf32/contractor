package auditservice

import (
	"encoding/json"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/artifactstest"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/jackc/pgx/v5"
)

func TestSharedCommonInitialAndPreparedTaskIntake(t *testing.T) {
	for _, cancelAfterFirstSlice := range []bool{false, true} {
		t.Run(fmt.Sprint("cancel=", cancelAfterFirstSlice), func(t *testing.T) {
			f := newCommonTaskServiceFixture(t, func(program *auditcontract.Program) {
				producer := program.Workflows["inspect"]
				producer.Ref = "audit-contract-check@1"
				producer.Inputs["context"] = auditcontract.InputMapping{Source: "task-context"}
				producer.Outputs = map[string]string{"work": "result"}
				program.Workflows["inspect"] = producer
				program.Workflows["check"] = auditcontract.ProgramWorkflow{Ref: "audit-prepared-check@1", Inputs: map[string]auditcontract.InputMapping{"operation": {Source: "task-input"}, "openapi": {Source: "task-input"}, "settings": {Source: "task-input"}}, Parameters: map[string]string{}, Outputs: map[string]string{"material": "result"}}
			})
			ctx := t.Context()
			store := auditstoretest.NewStore(f.pool, artifactpgtest.NewStorage())
			if _, err := f.pool.Exec(ctx, `UPDATE audits SET state='active',dispatch_state='open' WHERE audit_id=$1`, f.audit.AuditID); err != nil {
				t.Fatal(err)
			}
			claims, err := store.Claim(ctx, auditstore.ClaimParams{HolderID: "initial-controller", Lease: time.Minute, Limit: 1})
			if err != nil || len(claims) != 1 {
				t.Fatalf("claim: %+v %v", claims, err)
			}
			claim := claims[0]
			if _, err := f.pool.Exec(ctx, `UPDATE audits SET state='paused',dispatch_state='closed' WHERE audit_id=$1`, f.audit.AuditID); err != nil {
				t.Fatal(err)
			}
			snapshot := auditstore.ReconcileSnapshot{Audit: f.audit}
			initial, err := f.service.AcceptInitialTasks(ctx, claim, snapshot)
			if err != nil || !initial.Complete || !initial.Changed {
				t.Fatalf("initial: %+v %v", initial, err)
			}
			// Lost acknowledgement reuses the original slice identity and receipt.
			initial, err = f.service.AcceptInitialTasks(ctx, claim, snapshot)
			if err != nil || !initial.Complete || initial.Changed {
				t.Fatalf("initial replay: %+v %v", initial, err)
			}
			page, err := f.service.Tasks(ctx, auditstore.ListTasksParams{OwnerID: "owner", AuditID: f.audit.AuditID, Limit: 10})
			if err != nil || len(page.Tasks) != 1 {
				t.Fatalf("initial tasks: %+v %v", page, err)
			}
			history := 0
			if !cancelAfterFirstSlice {
				history = 90000
				for start := 2; start < history+2; start += 1000 {
					rows := make([][]any, 0, 1000)
					for ordinal := start; ordinal < min(start+1000, history+2); ordinal++ {
						document := page.Tasks[0].Document
						document.TaskID = fmt.Sprintf("history-%06d", ordinal)
						source := auditstore.TaskSource{Kind: "manual", Key: document.TaskID}
						encoded, err := strictjson.Canonical(document)
						if err != nil {
							t.Fatal(err)
						}
						origin, err := strictjson.Canonical(source)
						if err != nil {
							t.Fatal(err)
						}
						rows = append(rows, []any{document.TaskID, f.audit.AuditID, contentdigest.Bytes(origin), origin, encoded, page.Tasks[0].BatchDigest, ordinal, (ordinal-1)/10000 + 1, "settled", "fixture-complete"})
					}
					if _, err := f.pool.CopyFrom(ctx, pgx.Identifier{"audit_tasks"}, []string{"task_id", "audit_id", "source_key", "source", "document", "batch_digest", "ordinal", "iteration", "state", "gap_code"}, pgx.CopyFromRows(rows)); err != nil {
						t.Fatal(err)
					}
				}
				if _, err := f.pool.Exec(ctx, `UPDATE audits SET task_count=90001,task_iteration=10,iteration_task_count=1 WHERE audit_id=$1`, f.audit.AuditID); err != nil {
					t.Fatal(err)
				}
				if _, err := f.pool.Exec(ctx, `ANALYZE audit_tasks`); err != nil {
					t.Fatal(err)
				}
			}
			if _, err := f.pool.Exec(ctx, `UPDATE audits SET state='active',dispatch_state='open' WHERE audit_id=$1`, f.audit.AuditID); err != nil {
				t.Fatal(err)
			}
			execution, _, err := store.SelectTasks(ctx, auditstore.SelectTasksParams{Claim: claim, TaskIDs: []string{page.Tasks[0].Document.TaskID}})
			if err != nil {
				t.Fatal(err)
			}
			// This test isolates intake over actual Run/Artifact owner stores. The
			// RunService creation/collection integration has its own real-store gate.
			runID := "prepared-producer"
			workflow := f.pinned.Workflows["inspect"]
			workflowData, err := resolved.EncodeResolvedWorkflowSnapshot(workflow)
			if err != nil {
				t.Fatal(err)
			}
			err = postgres.InTx(ctx, f.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
				runs := runstore.NewPostgresStore(tx, artifactpgtest.NewStorage())
				if _, err := runs.CreateAuditRun(ctx, runstore.CreateAuditRunParams{CreateRunParams: runstore.CreateRunParams{RunID: runID, OwnerID: "owner", ProjectID: &f.audit.ProjectID, WorkflowName: workflow.Ref.Name, WorkflowVersion: workflow.Ref.Version, WorkflowSchemaVersion: "1", WorkflowSnapshot: workflowData, Parameters: map[string]string{}, RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot(), LaneClass: resolved.EntryLaneClass(workflow)}, AuditExecutionID: execution.ExecutionID, AuditSubmissionKey: execution.SubmissionKey}); err != nil {
					return err
				}
				_, err := auditstoretest.NewStore(tx, artifactpgtest.NewStorage()).BindRun(ctx, auditstore.BindRunParams{Claim: claim, ExecutionID: execution.ExecutionID, RunID: runID})
				return err
			})
			if err != nil {
				t.Fatal(err)
			}
			art := artifacts.NewService(artifactpgtest.NewStorage().Repository(f.pool))
			run, _ := art.Run(runID)
			write := func(name string, data []byte) auditcontract.Material {
				t.Helper()
				written, err := run.Write(ctx, contracts.ArtifactRef{Namespace: "builder", Name: name}, artifacts.Payload{MediaType: "application/json", Data: data}, nil)
				if err != nil {
					t.Fatal(err)
				}
				metadata, err := run.Metadata(ctx, written.Ref)
				if err != nil {
					t.Fatal(err)
				}
				return auditcontract.Material{ScopeKind: "run", ScopeID: runID, Ref: metadata.Ref, Digest: metadata.Digest, MediaType: metadata.MediaType, SizeBytes: metadata.Size}
			}
			openapi := write("openapi", []byte(`{"openapi":"3.1.0","paths":{}}`))
			work := auditcontract.PreparedWork{Schema: auditcontract.PreparedWorkSchema, Work: []auditcontract.PreparedSelection{}, Gaps: []string{}}
			for index := range 256 {
				operation := write(fmt.Sprintf("operation-%03d", index), []byte(fmt.Sprintf(`{"operation":%d}`, index)))
				work.Work = append(work.Work, auditcontract.PreparedSelection{Key: fmt.Sprint("operation-", index), Workflow: "check", Objective: "Check selected operation", Inputs: map[string]auditcontract.Material{"operation": operation, "openapi": openapi}})
			}
			data, _ := json.Marshal(work)
			output := write("work", data)
			if _, err := artifactstest.BindOutput(ctx, art, runID, "result", output.Ref); err != nil {
				t.Fatal(err)
			}
			scope, _ := artifacts.RunScope(runID)
			if err := art.Freeze(ctx, scope, "builder"); err != nil {
				t.Fatal(err)
			}
			if err := artifactstest.FreezeOutputs(ctx, art, runID); err != nil {
				t.Fatal(err)
			}
			bound, err := run.Metadata(ctx, contracts.ArtifactRef{Namespace: "outputs", Name: "result"})
			if err != nil {
				t.Fatal(err)
			}
			output.Ref = bound.Ref
			runs := runstore.NewPostgresStore(f.pool, artifactpgtest.NewStorage())
			for _, transition := range [][2]runstore.WorkflowRunState{{runstore.RunInitializing, runstore.RunPending}, {runstore.RunPending, runstore.RunRunning}, {runstore.RunRunning, runstore.RunSucceeded}} {
				if _, err := runs.TransitionRun(ctx, runID, transition[0], transition[1], runstore.Reason{Code: "fixture"}); err != nil {
					t.Fatal(err)
				}
			}
			cursor, err := runs.GetRunEventCursor(ctx, runID)
			if err != nil {
				t.Fatal(err)
			}
			execution, err = store.ObserveTerminal(ctx, auditstore.ObserveTerminalParams{Claim: claim, ExecutionID: execution.ExecutionID, RunID: runID, Generation: cursor.Generation, Sequence: uint64(cursor.Sequence)})
			if err != nil {
				t.Fatal(err)
			}
			if _, err := f.pool.Exec(ctx, `UPDATE audits SET state='paused',dispatch_state='closed' WHERE audit_id=$1`, f.audit.AuditID); err != nil {
				t.Fatal(err)
			}
			forged := append(append([]byte{}, data...), ' ')
			forgedSource := output
			forgedSource.Digest, forgedSource.SizeBytes = contentdigest.Bytes(forged), int64(len(forged))
			if _, err := f.service.AcceptPreparedWork(ctx, claim, auditstore.ReconcileSnapshot{Audit: f.audit}, execution, forgedSource, forged); !errors.Is(err, ErrInvalid) {
				t.Fatalf("forged list metadata accepted: %v", err)
			}
			for iteration := range 20 {
				current, err := f.service.Get(ctx, "owner", f.audit.AuditID)
				if err != nil {
					t.Fatal(err)
				}
				before := current.TaskCount
				result, err := f.service.AcceptPreparedWork(ctx, claim, auditstore.ReconcileSnapshot{Audit: current}, execution, output, data)
				if err != nil {
					t.Fatalf("slice %d: %v", iteration, err)
				}
				after, err := f.service.Get(ctx, "owner", f.audit.AuditID)
				if err != nil {
					t.Fatal(err)
				}
				if after.TaskCount-before > auditcontract.MaxMutationRecords || after.State != auditstore.AuditPaused {
					t.Fatalf("unbounded or dispatched intake: %+v", after)
				}
				if iteration == 0 {
					if result.Complete || !result.Changed || after.TaskCount <= history+1 || after.TaskCount >= history+257 {
						t.Fatalf("not a finite partial intake: %+v count=%d", result, after.TaskCount)
					}
					for _, statement := range []string{
						`UPDATE audit_executions SET prepared_work_offset=0 WHERE execution_id=$1`,
						`UPDATE audit_executions SET prepared_work_count=255 WHERE execution_id=$1`,
						`UPDATE audit_executions SET prepared_work_source=NULL,prepared_work_count=NULL,prepared_work_offset=0 WHERE execution_id=$1`,
					} {
						if _, err := f.pool.Exec(ctx, statement, execution.ExecutionID); err == nil {
							t.Fatal("committed producer source/progress was mutable")
						}
					}
					if err := store.ReleaseClaim(ctx, claim); err != nil {
						t.Fatal(err)
					}
					next, err := store.Claim(ctx, auditstore.ClaimParams{HolderID: "restarted-controller", Lease: time.Minute, Limit: 1})
					if err != nil || len(next) != 1 {
						t.Fatalf("takeover: %+v %v", next, err)
					}
					if _, err := f.service.AcceptPreparedWork(ctx, claim, auditstore.ReconcileSnapshot{Audit: current}, execution, output, data); !errors.Is(err, auditstore.ErrClaimLost) {
						t.Fatalf("stale claim: %v", err)
					}
					claim = next[0]
					if cancelAfterFirstSlice {
						if _, err := f.pool.Exec(ctx, `UPDATE audits SET state='cancelling' WHERE audit_id=$1`, f.audit.AuditID); err != nil {
							t.Fatal(err)
						}
						closed, err := f.service.Get(ctx, "owner", f.audit.AuditID)
						if err != nil {
							t.Fatal(err)
						}
						result, err := f.service.AcceptPreparedWork(ctx, claim, auditstore.ReconcileSnapshot{Audit: closed}, execution, output, data)
						if err != nil || !result.Complete || result.Changed {
							t.Fatalf("closed intake: %+v %v", result, err)
						}
						if latest, err := f.service.Get(ctx, "owner", f.audit.AuditID); err != nil || latest.TaskCount != after.TaskCount {
							t.Fatalf("post-cancel work admitted: %+v %v", latest, err)
						}
						return
					}
				}
				if result.Complete {
					if after.TaskCount != history+257 {
						t.Fatalf("lost work: %d", after.TaskCount)
					}
					var offset, count int
					if err := f.pool.QueryRow(ctx, `SELECT prepared_work_offset,prepared_work_count FROM audit_executions WHERE execution_id=$1`, execution.ExecutionID).Scan(&offset, &count); err != nil || offset != 256 || count != 256 {
						t.Fatalf("progress=%d/%d: %v", offset, count, err)
					}
					replay, err := f.service.AcceptPreparedWork(ctx, claim, auditstore.ReconcileSnapshot{Audit: current}, execution, output, data)
					if err != nil || !replay.Complete || replay.Changed {
						t.Fatalf("completed replay: %+v %v", replay, err)
					}
					return
				}
			}
			t.Fatal("bounded intake did not drain")
		})
	}
}
