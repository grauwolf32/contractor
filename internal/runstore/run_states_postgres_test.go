package runstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
)

// errStatementCaptured stops a store method after it has handed over its SQL.
var errStatementCaptured = errors.New("statement captured")

// statementCapture records the statement a store method sends, so a plan test
// explains exactly the production SQL and arguments.
type statementCapture struct {
	sql  string
	args []any
}

func (c *statementCapture) Exec(_ context.Context, sql string, args ...any) (pgconn.CommandTag, error) {
	c.sql, c.args = sql, args
	return pgconn.CommandTag{}, errStatementCaptured
}

func (c *statementCapture) Query(_ context.Context, sql string, args ...any) (pgx.Rows, error) {
	c.sql, c.args = sql, args
	return nil, errStatementCaptured
}

func (c *statementCapture) QueryRow(_ context.Context, sql string, args ...any) pgx.Row {
	c.sql, c.args = sql, args
	return capturedRow{}
}

type capturedRow struct{}

func (capturedRow) Scan(...any) error { return errStatementCaptured }

func captureStatement(t *testing.T, call func(*PostgresStore) error) statementCapture {
	t.Helper()
	capture := &statementCapture{}
	// Stores may wrap the error without its cause; the recorded SQL is proof.
	if err := call(NewPostgresStore(capture)); err == nil || capture.sql == "" {
		t.Fatalf("store call did not reach the database: %v", err)
	}
	return *capture
}

// The plan-test history has several owners and is mostly terminal: one Run
// in fifty is active. historyRun mirrors populateRunHistorySQL so the tests
// know which Runs each read must return.
const (
	runHistorySize   = 5000
	runHistoryOwners = 7
	runHistoryKeys   = 30
)

type historyRun struct {
	id, owner          string
	state              WorkflowRunState
	llm, workflowModel string
}

func runHistoryEntry(i int) historyRun {
	state := RunSucceeded
	switch {
	case i%50 == 0:
		state = []WorkflowRunState{RunInitializing, RunPending, RunRunning, RunWaiting}[(i/50)%4]
	case i%2 == 1:
		state = RunFailed
	}
	return historyRun{
		id: fmt.Sprintf("run-history-%05d", i), owner: fmt.Sprintf("owner-%d", i%runHistoryOwners),
		state: state, llm: fmt.Sprintf("llm-%d", i%runHistoryKeys),
		workflowModel: fmt.Sprintf("workflow-%d", i%runHistoryKeys),
	}
}

// populateRunHistorySQL loads the history in one statement; see historyRun.
// $1 is a valid RuntimeConfig snapshot whose LLM credential list is replaced
// per Run, $2 the owner count, $3 the credential count and $4 the Run count.
const populateRunHistorySQL = `
INSERT INTO workflow_runs (
    run_id, owner_id, workflow_name, workflow_version, workflow_schema_version,
    workflow_snapshot, parameters, runtime_labels, runtime_config_snapshot,
    state, state_reason_code, created_at, updated_at, started_at, finished_at
)
SELECT 'run-history-' || lpad(i::text, 5, '0'), 'owner-' || (i % $2),
       'history', '1', 'contractor/v1alpha1',
       jsonb_build_object('stages', jsonb_build_array(jsonb_build_object(
           'model', jsonb_build_object('credentialId', 'workflow-' || (i % $3))
       ))),
       '{}'::jsonb, '{}'::text[],
       jsonb_set($1::jsonb, '{llmCredentialIds}', jsonb_build_array('llm-' || (i % $3))),
       run.state, 'history', run.created_at, run.created_at,
       CASE WHEN run.state NOT IN ('initializing', 'pending') THEN run.created_at END,
       CASE WHEN run.state IN ('succeeded', 'failed') THEN run.created_at + interval '1 minute' END
FROM generate_series(1, $4) AS i
CROSS JOIN LATERAL (
    SELECT CASE
               WHEN i % 50 = 0 THEN (ARRAY['initializing', 'pending', 'running', 'waiting'])[(i / 50) % 4 + 1]
               WHEN i % 2 = 1 THEN 'failed'
               ELSE 'succeeded'
           END AS state,
           timestamptz '2026-09-01 00:00:00+00' + i * interval '1 second' AS created_at
) AS run`

// populateRunHistory loads the history and analyzes it, so the planner sees
// the real distribution. The per-row lifecycle-event trigger is off during
// the load: the reads under test do not look at Run events, and appending
// one per Run would dominate the test's run time.
func populateRunHistory(t *testing.T, ctx context.Context, pool *pgxpool.Pool) {
	t.Helper()
	snapshot, err := json.Marshal(runtimeconfig.BuiltInRunSnapshot())
	if err != nil {
		t.Fatal(err)
	}
	tx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tx.Rollback(context.Background()) }()
	for _, statement := range []struct {
		sql  string
		args []any
	}{
		// Deferred constraint triggers would otherwise block re-enabling.
		{sql: `SET CONSTRAINTS ALL IMMEDIATE`},
		{sql: `ALTER TABLE workflow_runs DISABLE TRIGGER workflow_runs_capture_lifecycle_event`},
		{sql: populateRunHistorySQL, args: []any{snapshot, runHistoryOwners, runHistoryKeys, runHistorySize}},
		{sql: `ALTER TABLE workflow_runs ENABLE TRIGGER workflow_runs_capture_lifecycle_event`},
	} {
		if _, err := tx.Exec(ctx, statement.sql, statement.args...); err != nil {
			t.Fatalf("populate Run history: %v", err)
		}
	}
	if err := tx.Commit(ctx); err != nil {
		t.Fatalf("commit Run history: %v", err)
	}
	if _, err := pool.Exec(ctx, `VACUUM ANALYZE workflow_runs`); err != nil {
		t.Fatal(err)
	}
}

// explainPlan returns the planner's text plan without executing the
// statement. generic asks for the plan a cached prepared statement may reuse,
// which does not see the argument values.
func explainPlan(t *testing.T, ctx context.Context, pool *pgxpool.Pool, statement statementCapture, generic bool) string {
	t.Helper()
	var lines []string
	if generic {
		// GENERIC_PLAN keeps the $n placeholders, which only the simple query
		// protocol passes through without binding arguments.
		conn, err := pool.Acquire(ctx)
		if err != nil {
			t.Fatal(err)
		}
		defer conn.Release()
		results, err := conn.Conn().PgConn().Exec(ctx, "EXPLAIN (GENERIC_PLAN) "+statement.sql).ReadAll()
		if err != nil {
			t.Fatalf("explain generic plan: %v", err)
		}
		for _, row := range results[0].Rows {
			lines = append(lines, string(row[0]))
		}
		return strings.Join(lines, "\n")
	}
	rows, err := pool.Query(ctx, "EXPLAIN "+statement.sql, statement.args...)
	if err != nil {
		t.Fatalf("explain: %v", err)
	}
	lines, err = pgx.CollectRows(rows, pgx.RowTo[string])
	if err != nil {
		t.Fatalf("read plan: %v", err)
	}
	return strings.Join(lines, "\n")
}

// assertPlanUsesIndexes fails unless both the custom and the generic plan of
// statement use every named index and neither scans workflow_runs whole.
func assertPlanUsesIndexes(
	t *testing.T, ctx context.Context, pool *pgxpool.Pool, name string, statement statementCapture, indexes ...string,
) {
	t.Helper()
	for _, generic := range []bool{false, true} {
		plan := explainPlan(t, ctx, pool, statement, generic)
		for _, index := range indexes {
			if !regexp.MustCompile(`\b` + index + `\b`).MatchString(plan) {
				t.Errorf("%s (generic plan %v) does not use %s:\n%s", name, generic, index, plan)
			}
		}
		if strings.Contains(plan, "Seq Scan on workflow_runs") {
			t.Errorf("%s (generic plan %v) reads the whole Run history:\n%s", name, generic, plan)
		}
	}
}

// TestPostgresActiveRunReadsUseActiveStateIndexes checks the reads over active
// Runs on a history that is mostly terminal: the polled Queue page and the
// LLM credential deletion check use their partial index over the active
// states and the Scheduler claim its claim-order indexes over the claimable
// Runs, and the Queue and credential reads return exactly the active Runs,
// pending and waiting ones included.
func TestPostgresActiveRunReadsUseActiveStateIndexes(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	populateRunHistory(t, ctx, pool)
	store := NewPostgresStore(pool)

	const queueOwner, llmCredential, workflowCredential = "owner-1", "llm-20", "workflow-20"
	var wantQueue, wantLLM, wantWorkflow []string
	wantStates := map[WorkflowRunState]bool{}
	for i := 1; i <= runHistorySize; i++ {
		run := runHistoryEntry(i)
		if !RunLifecycleActive.Includes(run.state) {
			continue
		}
		if run.owner == queueOwner {
			wantQueue = append(wantQueue, run.id)
			wantStates[run.state] = true
		}
		if run.llm == llmCredential {
			wantLLM = append(wantLLM, run.id)
		}
		if run.workflowModel == workflowCredential {
			wantWorkflow = append(wantWorkflow, run.id)
		}
	}
	if len(wantQueue) < 4 || len(wantStates) != 4 || len(wantLLM) < 4 || len(wantWorkflow) < 4 {
		t.Fatalf("history does not cover the active states: queue %d in %d states, LLM %d, Workflow %d",
			len(wantQueue), len(wantStates), len(wantLLM), len(wantWorkflow))
	}

	queue, err := store.ListRunQueue(ctx, ListRunQueueParams{OwnerID: queueOwner, Limit: 201})
	if err != nil {
		t.Fatal(err)
	}
	gotQueue := make([]string, 0, len(queue))
	for _, item := range queue {
		gotQueue = append(gotQueue, item.RunID)
	}
	if !slices.Equal(gotQueue, wantQueue) {
		t.Errorf("Queue = %v, want the owner's active Runs %v", gotQueue, wantQueue)
	}
	for credential, want := range map[string][]string{llmCredential: wantLLM, workflowCredential: wantWorkflow} {
		got, err := store.ListNonTerminalRunIDsByCredential(ctx, credential, 128)
		if err != nil || !slices.Equal(got, want) {
			t.Errorf("Runs pinning %s = (%v, %v), want %v", credential, got, err, want)
		}
	}

	assertPlanUsesIndexes(t, ctx, pool, "Queue page", captureStatement(t, func(store *PostgresStore) error {
		_, err := store.ListRunQueue(ctx, ListRunQueueParams{OwnerID: queueOwner, Limit: 51})
		return err
	}), "workflow_runs_owner_nonterminal_queue_idx")
	assertPlanUsesIndexes(t, ctx, pool, "LLM credential Runs", captureStatement(t, func(store *PostgresStore) error {
		_, err := store.ListNonTerminalRunIDsByCredential(ctx, llmCredential, 128)
		return err
	}), "workflow_runs_runtime_llm_credentials_idx", "workflow_runs_workflow_credentials_idx")
	assertPlanUsesIndexes(t, ctx, pool, "Scheduler claim", captureStatement(t, func(store *PostgresStore) error {
		_, err := store.ClaimRunnableRun(ctx, "claim-plan", time.Minute)
		return err
	}), "workflow_runs_claim_idx", "workflow_runs_deferred_claim_idx")
}

var (
	predicateStateList  = regexp.MustCompile(`\bstate = ANY \(ARRAY\[([^\]]*)\]\)`)
	predicateStateValue = regexp.MustCompile(`\bstate = '([a-z_]+)'::text`)
)

// predicateStates returns the WorkflowRun states a partial-index predicate
// admits, or nil when it does not restrict state.
func predicateStates(predicate string) []WorkflowRunState {
	var states []WorkflowRunState
	for _, match := range predicateStateList.FindAllStringSubmatch(predicate, -1) {
		for _, value := range sqlValue.FindAllStringSubmatch(match[1], -1) {
			states = append(states, WorkflowRunState(value[1]))
		}
	}
	for _, match := range predicateStateValue.FindAllStringSubmatch(predicate, -1) {
		states = append(states, WorkflowRunState(match[1]))
	}
	return states
}

// TestPostgresRunStateSetsMatchSchema holds the migrated workflow_runs schema
// to the one definition: the state CHECK lists exactly the WorkflowRun
// states, every other CHECK spells a declared set, and every partial index
// has the state predicate its readers need. A new partial index must be added
// to wantPredicates, so its state set is decided rather than copied.
func TestPostgresRunStateSetsMatchSchema(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	declared := map[string][]WorkflowRunState{}
	for _, set := range declaredRunStateSets() {
		declared[set.name] = set.states
	}

	rows, err := pool.Query(ctx, `
SELECT conname, pg_get_constraintdef(oid)
FROM pg_constraint
WHERE conrelid = 'workflow_runs'::regclass AND contype = 'c'
ORDER BY conname`)
	if err != nil {
		t.Fatal(err)
	}
	type definition struct{ Name, Text string }
	checks, err := pgx.CollectRows(rows, pgx.RowToStructByPos[definition])
	if err != nil {
		t.Fatal(err)
	}
	stateCheck := false
	for _, check := range checks {
		lists, _ := sqlValueLists(check.Text)
		for _, list := range lists {
			name, err := classifyRunStateList(list)
			if err != nil {
				t.Errorf("CHECK %s: %v", check.Name, err)
			}
			if check.Name == "workflow_runs_state_check" {
				stateCheck = true
				if name != "all" {
					t.Errorf("CHECK %s lists %v, want every WorkflowRun state", check.Name, list)
				}
			}
		}
	}
	if !stateCheck {
		t.Error("workflow_runs_state_check is missing or lists no states")
	}

	wantPredicates := map[string][]WorkflowRunState{
		"workflow_runs_claim_idx":                          nil,
		"workflow_runs_deferred_claim_idx":                 nil,
		"workflow_runs_owner_nonterminal_queue_idx":        declared["active"],
		"workflow_runs_runtime_llm_credentials_idx":        declared["active"],
		"workflow_runs_workflow_credentials_idx":           declared["active"],
		"workflow_runs_runtime_credentials_idx":            declared["active"],
		"workflow_runs_project_http_target_credential_idx": declared["active"],
		"workflow_runs_owner_idempotency_key":              nil,
		"workflow_runs_owner_project_created_idx":          nil,
		"workflow_runs_publication_mode_idx":               nil,
	}
	rows, err = pool.Query(ctx, `
SELECT index.relname, pg_get_expr(i.indpred, i.indrelid)
FROM pg_index AS i
JOIN pg_class AS index ON index.oid = i.indexrelid
WHERE i.indrelid = 'workflow_runs'::regclass AND i.indpred IS NOT NULL
ORDER BY index.relname`)
	if err != nil {
		t.Fatal(err)
	}
	predicates, err := pgx.CollectRows(rows, pgx.RowToStructByPos[definition])
	if err != nil {
		t.Fatal(err)
	}
	seen := map[string]bool{}
	for _, predicate := range predicates {
		seen[predicate.Name] = true
		want, declaredIndex := wantPredicates[predicate.Name]
		if !declaredIndex {
			t.Errorf("partial index %s (%s) has no declared state set", predicate.Name, predicate.Text)
			continue
		}
		if got := predicateStates(predicate.Text); !sameRunStates(got, want) {
			t.Errorf("partial index %s admits states %v, want %v (%s)", predicate.Name, got, want, predicate.Text)
		}
	}
	for name := range wantPredicates {
		if !seen[name] {
			t.Errorf("partial index %s is missing", name)
		}
	}

	// The claim indexes' predicates restrict scheduler_claim_tier, whose
	// expression ranks every active state and no terminal one: an active
	// state it left out could never be claimed, and a terminal one it ranked
	// would be claimed after it ended.
	var tierExpression string
	if err := pool.QueryRow(ctx, `
SELECT pg_get_expr(def.adbin, def.adrelid)
FROM pg_attrdef AS def
JOIN pg_attribute AS attribute ON attribute.attrelid = def.adrelid AND attribute.attnum = def.adnum
WHERE def.adrelid = 'workflow_runs'::regclass AND attribute.attname = 'scheduler_claim_tier'`).Scan(&tierExpression); err != nil {
		t.Fatalf("read scheduler_claim_tier expression: %v", err)
	}
	if got := predicateStates(tierExpression); !sameRunStates(slices.Compact(sortedRunStates(got)), declared["active"]) {
		t.Errorf("scheduler_claim_tier ranks states %v, want the active states %v (%s)", got, declared["active"], tierExpression)
	}
}

func sortedRunStates(states []WorkflowRunState) []WorkflowRunState {
	sorted := slices.Clone(states)
	slices.Sort(sorted)
	return sorted
}

// TestPostgresRunEventSequenceBumpIsHeapOnlyUpdate pins why the Queue index
// carries no INCLUDE columns: every Run event append bumps only
// next_run_event_sequence, and while no index covers that column PostgreSQL
// updates the row in place (HOT) without touching any workflow_runs index.
func TestPostgresRunEventSequenceBumpIsHeapOnlyUpdate(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	run := createTestRun(t, ctx, NewPostgresStore(pool), "run-event-sequence-hot")
	tx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tx.Rollback(context.Background()) }()
	// The counters also hold this backend's not yet reported earlier
	// transactions, so only their change across the update counts.
	counters := func() (updated, heapOnly int64) {
		t.Helper()
		if err := tx.QueryRow(ctx, `
SELECT pg_stat_get_xact_tuples_updated('workflow_runs'::regclass),
       pg_stat_get_xact_tuples_hot_updated('workflow_runs'::regclass)`).Scan(&updated, &heapOnly); err != nil {
			t.Fatal(err)
		}
		return updated, heapOnly
	}
	updatedBefore, heapOnlyBefore := counters()
	if _, err := tx.Exec(ctx, `
UPDATE workflow_runs SET next_run_event_sequence = next_run_event_sequence + 1 WHERE run_id = $1`, run.RunID); err != nil {
		t.Fatal(err)
	}
	updatedAfter, heapOnlyAfter := counters()
	if updated, heapOnly := updatedAfter-updatedBefore, heapOnlyAfter-heapOnlyBefore; updated != 1 || heapOnly != 1 {
		t.Fatalf("Run event sequence bump: %d updates, %d heap-only; an index covers next_run_event_sequence", updated, heapOnly)
	}
}

// TestPostgresClaimRenewalIsHeapOnlyUpdate renews a Scheduler claim: the
// renewal writes only the lease expiry, which no workflow_runs index covers,
// so PostgreSQL updates the row in place (HOT) without touching an index,
// the Run's updated_at (its public updatedAt and claim order) stays as it
// was, and the renewal returns the Run's state (SCH-R166).
func TestPostgresClaimRenewalIsHeapOnlyUpdate(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	store := NewPostgresStore(pool)
	run := createTestRun(t, ctx, store, "run-claim-renewal-hot")
	if _, err := admitTestRun(ctx, store, run.RunID); err != nil {
		t.Fatal(err)
	}
	claimed, err := store.ClaimRunnableRun(ctx, "claim-hot", time.Minute)
	if err != nil || claimed.RunID != run.RunID {
		t.Fatalf("claim = (%s, %v)", claimed.RunID, err)
	}
	tx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tx.Rollback(context.Background()) }()
	counters := func() (updated, heapOnly int64) {
		t.Helper()
		if err := tx.QueryRow(ctx, `
SELECT pg_stat_get_xact_tuples_updated('workflow_runs'::regclass),
       pg_stat_get_xact_tuples_hot_updated('workflow_runs'::regclass)`).Scan(&updated, &heapOnly); err != nil {
			t.Fatal(err)
		}
		return updated, heapOnly
	}
	updatedBefore, heapOnlyBefore := counters()
	state, err := NewPostgresStore(tx).RenewRunClaim(ctx, run.RunID, "claim-hot", 2*time.Minute)
	if err != nil || state != RunRunning {
		t.Fatalf("renewal = (%s, %v), want running", state, err)
	}
	updatedAfter, heapOnlyAfter := counters()
	if updated, heapOnly := updatedAfter-updatedBefore, heapOnlyAfter-heapOnlyBefore; updated != 1 || heapOnly != 1 {
		t.Fatalf("claim renewal: %d updates, %d heap-only; an index covers what the renewal writes", updated, heapOnly)
	}
	renewed, err := NewPostgresStore(tx).GetRun(ctx, run.RunID)
	if err != nil || !renewed.UpdatedAt.Equal(claimed.UpdatedAt) || renewed.SchedulerClaim == nil ||
		!renewed.SchedulerClaim.ExpiresAt.After(claimed.SchedulerClaim.ExpiresAt) {
		t.Fatalf("renewed Run updatedAt %v (claimed %v), claim %+v (claimed %+v), err %v",
			renewed.UpdatedAt, claimed.UpdatedAt, renewed.SchedulerClaim, claimed.SchedulerClaim, err)
	}
}

// TestPostgresWorkflowCredentialMatchKeepsItsExactSemantics checks that the
// indexed superset in the LLM credential read only narrows candidates: the
// read still reports exactly the Runs whose Workflow snapshot has a
// credentialId equal to the credential, including values nested in arrays,
// and no Run that merely holds the string elsewhere.
func TestPostgresWorkflowCredentialMatchKeepsItsExactSemantics(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	store := NewPostgresStore(pool)
	var want []string
	for index, shape := range []struct {
		snapshot string
		match    bool
	}{
		{snapshot: `{"stage":{"credentialId":"shape-key"}}`, match: true},
		{snapshot: `{"stages":[{"model":{"credentialId":"shape-key"}}]}`, match: true},
		{snapshot: `{"stage":{"credentialId":["shape-key"]}}`, match: true},
		{snapshot: `{"stage":{"credentialId":["other-key",["shape-key"]]}}`, match: true},
		{snapshot: `{"stage":{"credentialId":{"id":"shape-key"}}}`, match: false},
		{snapshot: `{"stage":{"credentialId":"other-key","name":"shape-key"}}`, match: false},
		{snapshot: `{"stage":{"credential":"shape-key"}}`, match: false},
	} {
		params := testRunParams(fmt.Sprintf("run-shape-%d", index))
		params.WorkflowSnapshot = json.RawMessage(shape.snapshot)
		if _, err := store.CreateRun(ctx, params); err != nil {
			t.Fatal(err)
		}
		if shape.match {
			want = append(want, params.RunID)
		}
	}
	got, err := store.ListNonTerminalRunIDsByCredential(ctx, "shape-key", 128)
	if err != nil || !slices.Equal(got, want) {
		t.Fatalf("Runs whose Workflow snapshot pins shape-key = (%v, %v), want %v", got, err, want)
	}
}
