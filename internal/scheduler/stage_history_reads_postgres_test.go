package scheduler

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// stageSnapshotColumns are the StageExecution columns that hold snapshots; a
// read of the Run's Stage history selects none of them (SCH-R164).
var stageSnapshotColumns = []string{
	"stage_spec_snapshot", "stage_context_snapshot", "candidate_stage_result",
	"accepted_stage_result", "stage_termination",
}

// stageHistoryPadding pads the StageContext snapshot of every historical
// attempt, so a read that selected the snapshots would move and decode tens
// of MiB for one decision.
const stageHistoryPadding = 64 << 10

// tracedStatement is one statement a traced pool ran and the rows its command
// tag reports.
type tracedStatement struct {
	sql  string
	rows int64
}

type tracedStatementKey struct{}

// statementTrace records every statement of a pool with its row count.
type statementTrace struct {
	mu         sync.Mutex
	statements []tracedStatement
}

func (trace *statementTrace) TraceQueryStart(ctx context.Context, _ *pgx.Conn, data pgx.TraceQueryStartData) context.Context {
	return context.WithValue(ctx, tracedStatementKey{}, data.SQL)
}

func (trace *statementTrace) TraceQueryEnd(ctx context.Context, _ *pgx.Conn, data pgx.TraceQueryEndData) {
	statement, _ := ctx.Value(tracedStatementKey{}).(string)
	trace.mu.Lock()
	defer trace.mu.Unlock()
	trace.statements = append(trace.statements, tracedStatement{sql: statement, rows: data.CommandTag.RowsAffected()})
}

// take returns the statements recorded since the last take.
func (trace *statementTrace) take() []tracedStatement {
	trace.mu.Lock()
	defer trace.mu.Unlock()
	statements := trace.statements
	trace.statements = nil
	return statements
}

// tracedStageStore is a store on its own pool of the harness database whose
// statements trace records.
func tracedStageStore(t *testing.T, ctx context.Context, pool *pgxpool.Pool) (*runstore.PostgresStore, *statementTrace) {
	t.Helper()
	trace := &statementTrace{}
	configuration := pool.Config().Copy()
	configuration.ConnConfig.Tracer = trace
	traced, err := pgxpool.NewWithConfig(ctx, configuration)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(traced.Close)
	return runstore.NewPostgresStore(traced), trace
}

// requireNoSnapshotHistoryRead fails when a statement that selects a
// snapshot column of stage_executions returned more rows than the one active
// Stage read may (LIMIT 2), and returns the statements on stage_executions
// that returned at least history rows.
func requireNoSnapshotHistoryRead(t *testing.T, statements []tracedStatement, history int64) []tracedStatement {
	t.Helper()
	var wide []tracedStatement
	for _, statement := range statements {
		if !strings.Contains(statement.sql, "stage_executions") {
			continue
		}
		for _, column := range stageSnapshotColumns {
			if strings.Contains(statement.sql, column) && statement.rows > 2 {
				t.Fatalf("statement selecting %s returned %d rows of the Stage history:\n%s",
					column, statement.rows, statement.sql)
			}
		}
		if statement.rows >= history {
			wide = append(wide, statement)
		}
	}
	return wide
}

// insertStageHistory stores count interrupted attempts of Stage stageName in
// run-1, chained by previous execution, each with a padded StageContext
// snapshot, in one statement, and returns the ID of the last one.
func insertStageHistory(
	t *testing.T, ctx context.Context, pool *pgxpool.Pool, stageName string, spec []byte,
	stageContext runstore.StageContextSnapshot, count int,
) string {
	t.Helper()
	padded := stageContext
	padded.Parameters = map[string]string{"padding": strings.Repeat("p", stageHistoryPadding)}
	for name, value := range stageContext.Parameters {
		padded.Parameters[name] = value
	}
	encodedContext, err := json.Marshal(padded)
	if err != nil {
		t.Fatal(err)
	}
	termination, err := json.Marshal(runstore.StageTermination{
		Outcome: runstore.TerminationInterrupted, Code: "runtime_unavailable",
		Message: "Runtime Agent became unavailable", Retryable: true,
		Phase: runstore.TerminationPreparing, OccurredAt: time.Now().UTC(),
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `
INSERT INTO stage_executions (
    stage_execution_id, run_id, stage_name, attempt, previous_execution_id,
    stage_spec_schema_version, stage_spec_snapshot,
    stage_context_schema_version, stage_context_snapshot,
    state, state_reason_code,
    termination_schema_version, stage_termination, abort_id, abort_deadline, terminal_at
)
SELECT format('stage-history-%s', lpad(attempt::text, 4, '0')), 'run-1', $1, attempt,
       CASE WHEN attempt = 1 THEN NULL ELSE format('stage-history-%s', lpad((attempt - 1)::text, 4, '0')) END,
       $2, $3::jsonb, $2, $4::jsonb,
       'interrupted', 'stage_interrupted',
       $2, $5::jsonb, format('abort-%s', attempt), clock_timestamp(), clock_timestamp()
FROM generate_series(1, $6::integer) AS attempt`,
		stageName, contracts.APIVersion, spec, encodedContext, termination, count,
	); err != nil {
		t.Fatal(err)
	}
	return fmt.Sprintf("stage-history-%04d", count)
}

// An automatic escalate decision of a Run with 1,024 StageExecutions counts
// the escalation budget from the lineage of the Stage's attempts in one
// statement that selects no snapshot column (SCH-R164), and still escalates.
func TestPostgresEscalationCountReadsOnlyStageLineage(t *testing.T) {
	const history = 1022 // with the escalated attempt and its successor: 1,024 (LIM-RUN-2)
	for _, outcome := range []string{"failed", "interrupted"} {
		t.Run(outcome, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
			defer cancel()
			f := newEscalationFixture(t, ctx, outcome)
			spec, err := json.Marshal(f.stage)
			if err != nil {
				t.Fatal(err)
			}
			last := insertStageHistory(t, ctx, f.pool, f.workflow.EntryStage, spec, f.stageContext, history)
			if _, err := f.persistence.CreateStageWithContext(
				ctx, f.attempt(t, "stage-base", history+1, &last, nil), f.pins,
			); err != nil {
				t.Fatal(err)
			}
			f.endAttempt(t, ctx, "stage-base")
			traced, trace := tracedStageStore(t, ctx, f.pool)
			f.reads.Store = traced

			if worked, err := f.scheduler.RunOnce(ctx); err != nil || !worked {
				t.Fatalf("escalating pass = (%v, %v)", worked, err)
			}
			wide := requireNoSnapshotHistoryRead(t, trace.take(), history+1)
			if f.reads.reads != 1 || len(wide) != 1 || wide[0].rows != history+1 {
				t.Fatalf("escalation count = %d lineage reads and history statements %+v, want one statement of %d rows",
					f.reads.reads, wide, history+1)
			}
			escalated, err := f.store.GetActiveStageExecution(ctx, "run-1")
			if err != nil || escalated.Attempt != history+2 || escalated.EscalationOrdinal == nil ||
				*escalated.EscalationOrdinal != 1 || escalated.PreviousExecutionID == nil ||
				*escalated.PreviousExecutionID != "stage-base" {
				t.Fatalf("escalated attempt = (%+v, %v), want attempt %d with ordinal 1", escalated, err, history+2)
			}
		})
	}
}

// stageHistoryProbes counts the Scheduler's existence probes of a Run's
// Stage history.
type stageHistoryProbes struct {
	Store
	probes int
}

func (s *stageHistoryProbes) HasStageExecutions(ctx context.Context, runID string) (bool, error) {
	s.probes++
	return s.Store.HasStageExecutions(ctx, runID)
}

// A Run without an active StageExecution whose history is not empty is
// durable state the Scheduler cannot progress. The first-tick check proves it
// with one existence probe instead of reading and decoding 1,024 attempts
// (SCH-R164), and ends the Run once.
func TestPostgresFirstTickProbesStageHistory(t *testing.T) {
	const history = 1024
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	h := newDeferredPlacementHarness(t, ctx, nil)
	spec, err := json.Marshal(h.workflow.Stages[h.workflow.EntryStage])
	if err != nil {
		t.Fatal(err)
	}
	insertStageHistory(t, ctx, h.pool, h.workflow.EntryStage, spec, runstore.StageContextSnapshot{
		Parameters: map[string]string{"objective": "copy exactly"},
		Artifacts:  map[string]runstore.PinnedContextArtifact{},
	}, history)
	traced, trace := tracedStageStore(t, ctx, h.pool)
	probes := &stageHistoryProbes{Store: traced}
	h.scheduler.store = probes

	if worked, err := h.scheduler.RunOnce(ctx); err != nil || !worked {
		t.Fatalf("first pass = (%v, %v), want the Run ended", worked, err)
	}
	statements := trace.take()
	if wide := requireNoSnapshotHistoryRead(t, statements, 3); len(wide) != 0 {
		t.Fatalf("first-tick check read the Stage history: %+v", wide)
	}
	existence := 0
	for _, statement := range statements {
		// The claim filters through NOT EXISTS too; the probe is a
		// statement of its own that returns one boolean.
		if strings.HasPrefix(strings.TrimSpace(statement.sql), "SELECT EXISTS") &&
			strings.Contains(statement.sql, "stage_executions") && statement.rows == 1 {
			existence++
		}
	}
	if probes.probes != 1 || existence != 1 {
		t.Fatalf("first-tick check = %d probes and %d existence statements, want one of each", probes.probes, existence)
	}
	run, err := h.store.GetRun(ctx, "run-1")
	if err != nil || run.State != runstore.RunFailed || run.StateReason.Code != "scheduler_state_invalid" {
		t.Fatalf("Run = (%s %+v, %v), want failed as scheduler_state_invalid", run.State, run.StateReason, err)
	}
	if worked, err := h.scheduler.RunOnce(ctx); err != nil || worked {
		t.Fatalf("second pass = (%v, %v), want no work", worked, err)
	}
	if probes.probes != 1 {
		t.Fatalf("first-tick probes = %d after the Run ended, want 1", probes.probes)
	}
}
