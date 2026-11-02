package runstore

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/agentskills"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/stageresult"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// unguardTable drops every CHECK and foreign-key constraint of table and
// disables its user triggers in the test's isolated schema, so that a raw
// UPDATE can store a document that only a broken writer would.
func unguardTable(t *testing.T, ctx context.Context, pool *pgxpool.Pool, table string) {
	t.Helper()
	rows, err := pool.Query(ctx,
		`SELECT conname FROM pg_constraint WHERE conrelid = $1::regclass AND contype IN ('c', 'f')`, table)
	if err != nil {
		t.Fatal(err)
	}
	names, err := pgx.CollectRows(rows, pgx.RowTo[string])
	if err != nil {
		t.Fatal(err)
	}
	identifier := pgx.Identifier{table}.Sanitize()
	for _, name := range names {
		if _, err := pool.Exec(ctx, `ALTER TABLE `+identifier+` DROP CONSTRAINT `+pgx.Identifier{name}.Sanitize()); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := pool.Exec(ctx, `ALTER TABLE `+identifier+` DISABLE TRIGGER USER`); err != nil {
		t.Fatal(err)
	}
}

// strictVariants returns document with an unknown member added and with
// member key spelled with a capital first letter, which encoding/json would
// both accept. In an array the first element changes.
func strictVariants(t *testing.T, document []byte, key string) map[string][]byte {
	t.Helper()
	variant := func(change func(map[string]any)) []byte {
		decoder := json.NewDecoder(bytes.NewReader(document))
		decoder.UseNumber()
		var value any
		if err := decoder.Decode(&value); err != nil {
			t.Fatal(err)
		}
		object, ok := value.(map[string]any)
		if list, isList := value.([]any); isList && len(list) > 0 {
			object, ok = list[0].(map[string]any)
		}
		if !ok {
			t.Fatalf("document %s has no object to change", document)
		}
		if _, present := object[key]; !present {
			t.Fatalf("document %s has no member %q", document, key)
		}
		change(object)
		encoded, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		return encoded
	}
	return map[string][]byte{
		"unknown member": variant(func(object map[string]any) { object["unexpectedMember"] = true }),
		"case-variant member": variant(func(object map[string]any) {
			object[strings.ToUpper(key[:1])+key[1:]] = object[key]
			delete(object, key)
		}),
	}
}

// strictReadCase is one struct-shaped document column: the row it belongs
// to, a member the document has, a valid document and the store read that
// decodes it.
type strictReadCase struct {
	table, column, keyColumn, id, member string
	valid                                func(t *testing.T) []byte
	read                                 func(ctx context.Context) error
}

func storedDocument(t *testing.T, ctx context.Context, pool *pgxpool.Pool, test strictReadCase) []byte {
	t.Helper()
	var document []byte
	if err := pool.QueryRow(ctx,
		`SELECT `+pgx.Identifier{test.column}.Sanitize()+`::text FROM `+pgx.Identifier{test.table}.Sanitize()+
			` WHERE `+pgx.Identifier{test.keyColumn}.Sanitize()+` = $1`, test.id,
	).Scan(&document); err != nil {
		t.Fatal(err)
	}
	return document
}

// TestPostgresStoredDocumentsDecodeStrictly stores, past every schema guard,
// each struct-shaped document of a Run, a StageExecution and an allocation
// with an unknown member and with a member spelled in another case. The
// valid document reads; each variant fails the read, as encoding/json would
// not have noticed (AGENTS.md section 6).
func TestPostgresStoredDocumentsDecodeStrictly(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	store := NewPostgresStore(pool, artifactpgtest.NewStorage())
	run := createTestRun(t, ctx, store, "run-strict")
	if _, err := admitTestRun(ctx, store, run.RunID); err != nil {
		t.Fatal(err)
	}
	audited := createTestRun(t, ctx, store, "run-strict-audit")
	stage := testActiveStageParams(run.RunID, "stage-strict", "copy")
	stage.StageContext = StageContextSnapshot{
		Parameters: map[string]string{"objective": "copy"}, Artifacts: map[string]PinnedContextArtifact{},
	}
	if _, err := store.CreateStageExecution(ctx, stage); err != nil {
		t.Fatal(err)
	}
	allocation := testStageAllocation(stage.StageExecutionID, "allocation-strict")
	if err := store.RecordStageAllocation(ctx, allocation); err != nil {
		t.Fatal(err)
	}
	for _, table := range []string{"workflow_runs", "stage_executions", "stage_allocations"} {
		unguardTable(t, ctx, pool, table)
	}
	// An Audit-managed Run carries the authority its completion requires.
	if _, err := pool.Exec(ctx, `
UPDATE workflow_runs
SET publication_mode = 'audit-managed', audit_execution_id = 'audit-strict', audit_submission_key = 'submission-strict'
WHERE run_id = $1`, audited.RunID); err != nil {
		t.Fatal(err)
	}

	marshal := func(value any) func(*testing.T) []byte {
		return func(t *testing.T) []byte {
			encoded, err := json.Marshal(value)
			if err != nil {
				t.Fatal(err)
			}
			return encoded
		}
	}
	readRun := func(id string) func(context.Context) error {
		return func(ctx context.Context) error { _, err := store.GetRun(ctx, id); return err }
	}
	readStage := func(ctx context.Context) error {
		_, err := store.GetStageExecution(ctx, stage.StageExecutionID)
		return err
	}
	readAllocations := func(ctx context.Context) error {
		_, err := store.ListStageAllocations(ctx, stage.StageExecutionID)
		return err
	}
	revision := "revision-1"
	result := stageresult.StageContentResult{
		APIVersion: contracts.APIVersion, Outcome: stageresult.StageSucceeded, Summary: "copied",
		Artifacts: map[string]contracts.ArtifactRef{},
	}
	completion := contracts.WorkerCompletionContract{
		Kind:              contracts.AuditCheckResultsV1,
		Task:              contracts.ArtifactRef{Namespace: "inputs", Name: "task", Revision: &revision},
		ExecutionManifest: contracts.ArtifactRef{Namespace: "inputs", Name: "manifest", Revision: &revision},
		ResultArtifact:    contracts.ArtifactRef{Namespace: "check", Name: "result"},
	}
	allocationCompletion := completion
	allocationCompletion.ResultArtifact.Namespace = allocation.Namespace
	runRow := func(column, member string, valid func(*testing.T) []byte) strictReadCase {
		return strictReadCase{
			table: "workflow_runs", column: column, keyColumn: "run_id", id: run.RunID, member: member,
			valid: valid, read: readRun(run.RunID),
		}
	}
	stageRow := func(column, member string, valid func(*testing.T) []byte) strictReadCase {
		return strictReadCase{
			table: "stage_executions", column: column, keyColumn: "stage_execution_id", id: stage.StageExecutionID,
			member: member, valid: valid, read: readStage,
		}
	}
	allocationRow := func(column, member string, valid func(*testing.T) []byte) strictReadCase {
		return strictReadCase{
			table: "stage_allocations", column: column, keyColumn: "allocation_id", id: allocation.AllocationID,
			member: member, valid: valid, read: readAllocations,
		}
	}
	var cases map[string]strictReadCase
	stored := func(name string) func(*testing.T) []byte {
		return func(t *testing.T) []byte { return storedDocument(t, ctx, pool, cases[name]) }
	}
	cases = map[string]strictReadCase{
		"Run RuntimeConfig snapshot": runRow("runtime_config_snapshot", "default", stored("Run RuntimeConfig snapshot")),
		"Run cancellation": runRow("run_cancellation", "requestedAt", marshal(WorkflowRunCancellation{
			Code: CancellationUserRequested, RequestedAt: time.Now().UTC(),
		})),
		"Run Project HTTP target": runRow("project_http_target_snapshot", "url", marshal(runtimesettings.HTTPOriginTargetRef{
			URL: "https://app.example.test/api",
		})),
		"Run Skill snapshot": runRow("skill_snapshot", "name", marshal([]agentskills.RunSkillSnapshot{{
			Name: "review",
			Source: &contracts.ArtifactRef{
				Namespace: contracts.AgentSkillNamespace, Name: "review", Revision: &revision,
			},
			SourceDigest: "sha256:" + strings.Repeat("a", 64), SourceSize: 1024,
		}})),
		"Run Audit completion": {
			table: "workflow_runs", column: "audit_completion", keyColumn: "run_id", id: audited.RunID, member: "stage",
			valid: marshal(AuditCompletionSnapshot{Stage: "check", Agent: "checker", Contract: completion}),
			read:  readRun(audited.RunID),
		},
		"StageContext":          stageRow("stage_context_snapshot", "parameters", stored("StageContext")),
		"candidate StageResult": stageRow("candidate_stage_result", "summary", marshal(result)),
		"accepted StageResult":  stageRow("accepted_stage_result", "summary", marshal(result)),
		"StageTermination": stageRow("stage_termination", "code", marshal(StageTermination{
			Outcome: TerminationInterrupted, Code: "runtime_unavailable", Message: "Runtime Agent became unavailable",
			Retryable: true, Phase: TerminationPreparing, OccurredAt: time.Now().UTC(),
		})),
		"allocation completion contract": allocationRow("completion_contract", "kind", marshal(allocationCompletion)),
		"allocation AgentTemplate ref":   allocationRow("agent_template_ref", "templateId", stored("allocation AgentTemplate ref")),
		"allocation WorkerRuntime ref":   allocationRow("worker_runtime_ref", "runtimeId", stored("allocation WorkerRuntime ref")),
		"allocation Runtime configuration": allocationRow(
			"runtime_configuration", "origins", stored("allocation Runtime configuration"),
		),
	}
	for name, test := range cases {
		t.Run(name, func(t *testing.T) {
			write := func(document []byte) {
				t.Helper()
				if _, err := pool.Exec(ctx,
					`UPDATE `+pgx.Identifier{test.table}.Sanitize()+` SET `+pgx.Identifier{test.column}.Sanitize()+
						` = $1::jsonb WHERE `+pgx.Identifier{test.keyColumn}.Sanitize()+` = $2`, string(document), test.id,
				); err != nil {
					t.Fatal(err)
				}
			}
			valid := test.valid(t)
			write(valid)
			if err := test.read(ctx); err != nil {
				t.Fatalf("read with the valid %s = %v", test.column, err)
			}
			for variant, document := range strictVariants(t, valid, test.member) {
				write(document)
				if err := test.read(ctx); err == nil {
					t.Errorf("read with %s %s succeeded: %s", test.column, variant, document)
				}
			}
			// Later cases read the same row.
			write(valid)
		})
	}
}
