package pgstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/persistence/explaintest"
	"github.com/jackc/pgx/v5"
)

// originalNextLiveRunForDeletionSQL is nextLiveRunForDeletionSQL as V414-029
// found it: it joined every execution of the deleting Audit with
// workflow_runs before it ordered them, so each deletion step read the
// Audit's whole execution history and, through a hash join, every Run. It is
// the reference of the traversal test and the statement the scale test must
// reject.
var originalNextLiveRunForDeletionSQL = `
SELECT run.run_id
  FROM audits AS audit
  JOIN audit_controller_claims AS claim USING (audit_id)
  JOIN audit_executions AS execution USING (audit_id)
  JOIN workflow_runs AS run ON run.run_id = execution.run_id
 WHERE audit.audit_id = $1 AND audit.state = 'deleting'
   AND ` + auditClaims.LiveCondition("claim", "$2", "$3") + `
 ORDER BY execution.created_at, execution.execution_id
 LIMIT 1`

// originalPurgeClaimedAuditSQL is purgeClaimedAuditSQL as V414-029 found it:
// its last gate joined every execution of the Audit with workflow_runs, and
// purgeClaimedAudit evaluates it on every purge pass.
var originalPurgeClaimedAuditSQL = `
SELECT audit.project_id
  FROM audits AS audit
  JOIN audit_controller_claims AS claim USING (audit_id)
 WHERE audit.audit_id = $1 AND audit.state = 'deleting'
   AND audit.dispatch_state = 'closed' AND audit.hold_state <> 'held'
   AND audit.deletion_requested_at IS NOT NULL
   AND ` + auditClaims.LiveCondition("claim", "$2", "$3") + `
   AND NOT EXISTS (
       SELECT 1 FROM audit_items AS item
        WHERE item.audit_id = audit.audit_id AND item.state <> 'settled'
   )
   AND NOT ` + uncollectedExecutionSQL + `
   AND NOT EXISTS (
       SELECT 1
         FROM audit_executions AS execution
         JOIN workflow_runs AS run ON run.run_id = execution.run_id
        WHERE execution.audit_id = audit.audit_id
   )
 FOR UPDATE OF audit, claim`

// The deletion-probe fixture's history: each phase gives the deleting Audit
// deletionProbeGone executions whose Run is stamped deleted, created before
// its live ones, and another Audit of the Project deletionProbeUnrelated
// executions with live child Runs.
const (
	deletionProbeGone      = 20_000
	deletionProbeUnrelated = 15_000
	// deletionProbeLive live child Runs follow the fixture's own execution
	// in the deleting Audit of the live-tail fixture.
	deletionProbeLive = 50
	// deletionProbePlannerRuns are the workflow_runs_pkey entries the planner
	// reads, the lowest and the highest, to estimate the Run probe; the
	// statements read them while they are planned, before they run.
	deletionProbePlannerRuns = 2
	// deletionProbeDescents are the index descents of one probe that may
	// each pass one more B-tree level once the tables have grown: the
	// execution and Run probes and the planner's two.
	deletionProbeDescents = 4
)

// deletionProbeFixture is a deleting Audit with a live Controller claim and
// its dispatch hold released, as the Controller holds it while it deletes
// the child Runs, and another Audit of the same Project.
type deletionProbeFixture struct {
	collectingAuditFixture
	background string
}

// newDeletionProbeFixture collects the fixture's execution and deletes the
// Audit, keeping the execution's Run, in a database of its own.
func newDeletionProbeFixture(t *testing.T, name string) deletionProbeFixture {
	t.Helper()
	f := newCollectingAuditFixture(t, name, 1<<20, collectingAuditFixtureOptions{ownDatabase: true})
	// The copies and the reads outlast the collecting fixture's deadline.
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	t.Cleanup(cancel)
	f.ctx = ctx
	collectPurgeFinding(t, f, "receipt-"+name)
	background, _, err := f.store.CreateDraft(f.ctx, auditstore.CreateDraftParams{
		AuditID: "audit-" + name + "-background", OwnerID: f.audit.OwnerID, ProjectID: f.project.ProjectID,
		Profile: f.audit.Profile, ProfileSnapshot: f.audit.ProfileSnapshot,
		InputSelection: json.RawMessage(`{}`), Limits: f.audit.Limits,
		IdempotencyKey: name + "-background", RequestDigest: testDigest("e"),
	})
	if err != nil {
		t.Fatal(err)
	}
	drainDeletedAuditForPurge(t, f, purgeDrainOptions{keepRun: true})
	return deletionProbeFixture{collectingAuditFixture: f, background: background.AuditID}
}

// deleteChildRunAsRunDeletion stamps the execution's Run deletion and deletes
// the Run in one transaction, as internal/rundeletion does.
func deleteChildRunAsRunDeletion(t *testing.T, f collectingAuditFixture, executionID, runID string) {
	t.Helper()
	tx, err := f.pool.Begin(f.ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tx.Rollback(context.WithoutCancel(f.ctx)) }()
	if tag, err := tx.Exec(f.ctx, markExecutionRunDeletedSQL, executionID, runID); err != nil || tag.RowsAffected() != 1 {
		t.Fatalf("stamp Run %s deleted = (%v, %v)", runID, tag, err)
	}
	if tag, err := tx.Exec(f.ctx, `DELETE FROM workflow_runs WHERE run_id = $1`, runID); err != nil || tag.RowsAffected() != 1 {
		t.Fatalf("delete Run %s = (%v, %v)", runID, tag, err)
	}
	if err := tx.Commit(f.ctx); err != nil {
		t.Fatal(err)
	}
}

// deletionProbeExecution describes copies of the fixture's execution: count
// executions of audit named <prefix>-<n>, created offset plus n milliseconds
// after it, with a live Run unless gone, when their Run is stamped deleted.
// Copies for another Audit are its discovery attempts attempts+n.
type deletionProbeExecution struct {
	audit, prefix   string
	count, attempts int
	offset          string
	gone            bool
}

// addDeletionProbeExecutions copies the fixture's execution, and its Run for
// each copy that keeps one, in one transaction, then vacuums and analyzes
// the tables the probe reads. The copies satisfy every constraint and
// foreign key but not the triggers that guard the store's own execution and
// Run writes; the test owns its schema, so it switches them off for the
// copies. They are read-model fixtures, not store-made executions.
func addDeletionProbeExecutions(t *testing.T, f deletionProbeFixture, groups ...deletionProbeExecution) {
	t.Helper()
	setup, err := f.pool.Begin(f.ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = setup.Rollback(context.WithoutCancel(f.ctx)) }()
	toggle := func(action string) {
		for _, table := range []string{"audit_executions", "workflow_runs"} {
			if _, err := setup.Exec(f.ctx, `ALTER TABLE `+table+` `+action+` TRIGGER USER`); err != nil {
				t.Fatal(err)
			}
		}
	}
	toggle("DISABLE")
	// Runs are copies of the fixture's child Run while it exists, then of
	// the first unrelated Run.
	sourceRun := *f.execution.RunID
	var present bool
	if err := setup.QueryRow(f.ctx, `SELECT EXISTS (SELECT 1 FROM workflow_runs WHERE run_id = $1)`, sourceRun).Scan(&present); err != nil {
		t.Fatal(err)
	}
	if !present {
		sourceRun = "run-unrelated-1-1"
	}
	for _, group := range groups {
		overrides := map[string]string{
			"execution_id":          `'` + group.prefix + `-' || n`,
			"audit_id":              `'` + group.audit + `'`,
			"submission_key":        `'submission-` + group.prefix + `-' || n`,
			"run_id":                `'run-` + group.prefix + `-' || n`,
			"run_provenance":        `jsonb_set(src.run_provenance, '{runId}', to_jsonb('run-` + group.prefix + `-' || n))`,
			"collection_receipt_id": `'collection-` + group.prefix + `-' || n`,
			"created_at":            `src.created_at + interval '` + group.offset + `' + n * interval '1 millisecond'`,
		}
		if group.audit != f.audit.AuditID {
			// Another Audit's executions belong to no Round of it.
			overrides["round_id"], overrides["role"] = `NULL`, `'task'`
			overrides["role_attempt"] = `NULL`
		}
		overrides["run_deleted_at"] = `NULL`
		if group.gone {
			overrides["run_deleted_at"] = `clock_timestamp()`
		}
		cloneRows(t, f.ctx, setup, "audit_executions", "execution_id", f.execution.ExecutionID, group.count, overrides)
		if group.gone {
			continue
		}
		cloneRows(t, f.ctx, setup, "workflow_runs", "run_id", sourceRun, group.count, map[string]string{
			"run_id":               `'run-` + group.prefix + `-' || n`,
			"audit_execution_id":   `'` + group.prefix + `-' || n`,
			"audit_submission_key": `'submission-` + group.prefix + `-' || n`,
		})
	}
	toggle("ENABLE")
	// The copies can take longer than the fixture's claim lease.
	if _, err := setup.Exec(f.ctx, `
UPDATE audit_controller_claims SET expires_at = clock_timestamp() + interval '10 minutes'
 WHERE audit_id = $1 AND holder_id = $2 AND epoch = $3`, f.claim.AuditID, f.claim.HolderID, f.claim.Epoch); err != nil {
		t.Fatal(err)
	}
	if err := setup.Commit(f.ctx); err != nil {
		t.Fatal(err)
	}
	vacuumDeletionProbe(t, f)
}

// vacuumDeletionProbe vacuums, index entries included, and analyzes the
// tables the probes read.
func vacuumDeletionProbe(t *testing.T, f deletionProbeFixture) {
	t.Helper()
	if _, err := f.pool.Exec(f.ctx,
		`VACUUM (INDEX_CLEANUP ON, ANALYZE) audit_executions, workflow_runs, audits, audit_controller_claims`); err != nil {
		t.Fatal(err)
	}
}

// deletionProbeHistory is one phase of history: executions of the deleting
// Audit whose Runs are gone and live child Runs of another Audit.
func deletionProbeHistory(f deletionProbeFixture, phase int) []deletionProbeExecution {
	return []deletionProbeExecution{
		{
			audit: f.background, prefix: fmt.Sprintf("unrelated-%d", phase), count: deletionProbeUnrelated,
			attempts: phase * deletionProbeUnrelated, offset: "-1 hour",
		},
		{audit: f.audit.AuditID, prefix: fmt.Sprintf("gone-%d", phase), count: deletionProbeGone, offset: "-1 hour", gone: true},
	}
}

// deletionProbeMeasurement is what one statement read: index entries and
// rows of audit_executions and workflow_runs, and its shared blocks.
type deletionProbeMeasurement struct {
	executions, runs, blocks int64
	plan                     string
}

// explainDeletionProbe analyzes sql for the fixture's claim in mode, rolled
// back, and requires it to return want rows.
func explainDeletionProbe(
	t *testing.T, f deletionProbeFixture, sql string, mode explaintest.Mode, want int,
) deletionProbeMeasurement {
	t.Helper()
	analysis, err := explaintest.ExplainPrepared(f.ctx, f.pool, mode, sql, f.claim.AuditID, f.claim.HolderID, int64(f.claim.Epoch))
	if err != nil {
		t.Fatal(err)
	}
	if rows := int(analysis.Plan.Rows); rows != want {
		t.Fatalf("%s plan returned %d rows, want %d:\n%s", mode, rows, want, analysis.Plan)
	}
	return deletionProbeMeasurement{
		executions: analysis.Reads.Of("audit_executions"), runs: analysis.Reads.Of("workflow_runs"),
		blocks: analysis.Plan.Blocks(), plan: analysis.Plan.String(),
	}
}

// deletionProbeBudget is the budget of one statement on the fixture: the
// first execution whose Run is not stamped deleted when there is one, its
// Run, and the planner's Run entries. The store's own writes of the fixture's
// execution (collection and, in the drained fixture, the Run deletion stamp)
// superseded versions of it, of its Run and of the Audit; the fixture's
// database is its own, so VACUUM removes their entries and the statements
// read live entries only.
func deletionProbeBudget(live bool) deletionProbeMeasurement {
	budget := deletionProbeMeasurement{runs: deletionProbePlannerRuns}
	if live {
		budget.executions++
		budget.runs++
	}
	return budget
}

func (m deletionProbeMeasurement) within(budget deletionProbeMeasurement) bool {
	return m.executions <= budget.executions && m.runs <= budget.runs
}

// TestPostgresAuditDeletionProbesReadOnlyTheLiveRuns measures one deletion
// step's next-live-Run probe and the purge gate (V414-029) on a deleting
// Audit with 20,000, then 40,000 executions whose Runs are gone, with a live
// tail of 51 child Runs and after it is drained, while another Audit has
// 15,000, then 30,000 live child Runs. In Custom and Generic plans both read
// at most the first execution whose Run is not stamped deleted and its Run
// (deletionProbeBudget), and their blocks grow by at most one B-tree level
// per index descent, however long the histories are, while the original
// statements read every execution of the Audit or every Run.
func TestPostgresAuditDeletionProbesReadOnlyTheLiveRuns(t *testing.T) {
	type probe struct {
		name, sql, original string
		want                int
	}
	cases := []struct {
		name    string
		keepRun bool
		probes  []probe
	}{
		{name: "deletion-probe-live", keepRun: true, probes: []probe{
			{"next live Run", nextLiveRunForDeletionSQL, originalNextLiveRunForDeletionSQL, 1},
			{"purge gate", purgeClaimedAuditSQL, originalPurgeClaimedAuditSQL, 0},
		}},
		{name: "deletion-probe-drained", keepRun: false, probes: []probe{
			{"next live Run", nextLiveRunForDeletionSQL, originalNextLiveRunForDeletionSQL, 0},
			{"purge gate", purgeClaimedAuditSQL, originalPurgeClaimedAuditSQL, 1},
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			f := newDeletionProbeFixture(t, tc.name)
			groups := deletionProbeHistory(f, 1)
			if tc.keepRun {
				groups = append(groups, deletionProbeExecution{
					audit: f.audit.AuditID, prefix: "live", count: deletionProbeLive, offset: "0 seconds",
				})
			}
			addDeletionProbeExecutions(t, f, groups...)
			if !tc.keepRun {
				deleteChildRunAsRunDeletion(t, f.collectingAuditFixture, f.execution.ExecutionID, *f.execution.RunID)
				vacuumDeletionProbe(t, f)
			}
			before := map[string]deletionProbeMeasurement{}
			measure := func(phase string) {
				budget := deletionProbeBudget(tc.keepRun)
				for _, p := range tc.probes {
					for _, mode := range explaintest.Modes {
						key := p.name + "/" + string(mode)
						measured := explainDeletionProbe(t, f, p.sql, mode, p.want)
						t.Logf("%s, %s: %d executions, %d Runs, %d blocks", phase, key, measured.executions, measured.runs, measured.blocks)
						if !measured.within(budget) {
							t.Errorf("%s, %s read %d executions and %d Runs, budget %d and %d:\n%s", phase, key,
								measured.executions, measured.runs, budget.executions, budget.runs, measured.plan)
						}
						if previous, ok := before[key]; ok {
							if measured.blocks > previous.blocks+deletionProbeDescents {
								t.Errorf("%s, %s read %d blocks, %d before the history grew", phase, key, measured.blocks, previous.blocks)
							}
						} else {
							before[key] = measured
						}
						original := explainDeletionProbe(t, f, p.original, mode, p.want)
						t.Logf("%s, original %s: %d executions, %d Runs, %d blocks", phase, key, original.executions, original.runs, original.blocks)
						if original.within(budget) {
							t.Errorf("%s, the original %s read %d executions and %d Runs, within the budget", phase, key, original.executions, original.runs)
						}
					}
				}
			}
			measure(fmt.Sprintf("%d gone executions", deletionProbeGone))
			addDeletionProbeExecutions(t, f, deletionProbeHistory(f, 2)...)
			measure(fmt.Sprintf("%d gone executions", 2*deletionProbeGone))
		})
	}
}

// TestPostgresAuditDeletionProbeTraversesRunsInOriginalOrder deletes a
// deleting Audit's child Runs one probe at a time (V414-029): the rewritten
// probe returns the Run the original statement returns at every step, in
// (created_at, execution_id) order with ties, past executions whose Run is
// stamped deleted, a Run deleted out of order and a Run removed without its
// stamp, and never another Audit's Run; a lost claim stays ErrClaimLost, and
// the purge gate, rewritten and original, passes only once no child Run is
// left.
func TestPostgresAuditDeletionProbeTraversesRunsInOriginalOrder(t *testing.T) {
	f := newDeletionProbeFixture(t, "deletion-traversal")
	addDeletionProbeExecutions(t, f,
		deletionProbeExecution{audit: f.background, prefix: "unrelated-1", count: 5, offset: "-2 seconds"},
		deletionProbeExecution{audit: f.audit.AuditID, prefix: "early", count: 2, offset: "-1 second"},
		deletionProbeExecution{audit: f.audit.AuditID, prefix: "tie-b", count: 3, offset: "1 second"},
		deletionProbeExecution{audit: f.audit.AuditID, prefix: "tie-a", count: 3, offset: "1 second"},
		deletionProbeExecution{audit: f.audit.AuditID, prefix: "gone", count: 3, offset: "1 second", gone: true},
	)
	// Executions created at the same instant are ordered by their IDs.
	remaining := []string{
		"run-early-1", "run-early-2", *f.execution.RunID,
		"run-tie-a-1", "run-tie-b-1", "run-tie-a-2", "run-tie-b-2", "run-tie-a-3", "run-tie-b-3",
	}
	executionOf := func(runID string) string {
		if runID == *f.execution.RunID {
			return f.execution.ExecutionID
		}
		return runID[len("run-"):]
	}
	original := func() (string, bool) {
		var runID string
		err := f.pool.QueryRow(f.ctx, originalNextLiveRunForDeletionSQL, f.claim.AuditID, f.claim.HolderID, f.claim.Epoch).Scan(&runID)
		if errors.Is(err, pgx.ErrNoRows) {
			return "", false
		}
		if err != nil {
			t.Fatal(err)
		}
		return runID, true
	}
	purgeGates := func() (rewritten, originalGate bool) {
		tx, err := f.pool.Begin(f.ctx)
		if err != nil {
			t.Fatal(err)
		}
		defer func() { _ = tx.Rollback(context.WithoutCancel(f.ctx)) }()
		passes := func(sql string) bool {
			var projectID string
			err := tx.QueryRow(f.ctx, sql, f.claim.AuditID, f.claim.HolderID, f.claim.Epoch).Scan(&projectID)
			if errors.Is(err, pgx.ErrNoRows) {
				return false
			}
			if err != nil || projectID != f.project.ProjectID {
				t.Fatalf("purge gate = (%q, %v)", projectID, err)
			}
			return true
		}
		return passes(purgeClaimedAuditSQL), passes(originalPurgeClaimedAuditSQL)
	}
	lost := f.claim
	lost.Epoch++
	namespace := auditdomain.ArtifactNamespace(f.audit.AuditID)
	for step := 0; len(remaining) > 0; step++ {
		runID, ok, err := f.store.NextLiveRunForDeletion(f.ctx, f.claim)
		want, wantOK := original()
		if err != nil || !ok || runID != remaining[0] || !wantOK || want != runID {
			t.Fatalf("step %d: probe = (%q, %t, %v), original = (%q, %t), want %q", step, runID, ok, err, want, wantOK, remaining[0])
		}
		if _, _, err := f.store.NextLiveRunForDeletion(f.ctx, lost); !errors.Is(err, auditstore.ErrClaimLost) {
			t.Fatalf("step %d: probe with a lost claim = %v, want ErrClaimLost", step, err)
		}
		if rewritten, originalGate := purgeGates(); rewritten || originalGate {
			t.Fatalf("step %d: purge gate passed with Run %s left (rewritten %t, original %t)", step, runID, rewritten, originalGate)
		}
		if err := f.store.PurgeClaimed(f.ctx, f.claim, namespace); !errors.Is(err, auditstore.ErrPrecondition) {
			t.Fatalf("step %d: purge with Run %s left = %v, want ErrPrecondition", step, runID, err)
		}
		switch step {
		case 1:
			// A Run deleted out of order leaves the order of the others.
			deleteChildRunAsRunDeletion(t, f.collectingAuditFixture, "tie-b-2", "run-tie-b-2")
			remaining = slices.DeleteFunc(remaining, func(id string) bool { return id == "run-tie-b-2" })
			deleteChildRunAsRunDeletion(t, f.collectingAuditFixture, executionOf(runID), runID)
		case 3:
			// A Run removed without its stamp, as the purge fixtures
			// remove one, is no longer returned either.
			if _, err := f.pool.Exec(f.ctx, `DELETE FROM workflow_runs WHERE run_id = $1`, runID); err != nil {
				t.Fatal(err)
			}
		default:
			deleteChildRunAsRunDeletion(t, f.collectingAuditFixture, executionOf(runID), runID)
		}
		remaining = remaining[1:]
	}
	if runID, ok, err := f.store.NextLiveRunForDeletion(f.ctx, f.claim); err != nil || ok {
		t.Fatalf("drained probe = (%q, %t, %v)", runID, ok, err)
	}
	if runID, ok := original(); ok {
		t.Fatalf("drained original probe returned %q", runID)
	}
	if _, _, err := f.store.NextLiveRunForDeletion(f.ctx, lost); !errors.Is(err, auditstore.ErrClaimLost) {
		t.Fatalf("drained probe with a lost claim = %v, want ErrClaimLost", err)
	}
	if rewritten, originalGate := purgeGates(); !rewritten || !originalGate {
		t.Fatalf("drained purge gate: rewritten %t, original %t", rewritten, originalGate)
	}
	if err := purgeFixtureUntilAbsent(t, f.collectingAuditFixture, f.claim); err != nil {
		t.Fatalf("purge of the drained Audit = %v", err)
	}
	var unrelated int
	if err := f.pool.QueryRow(f.ctx, `SELECT count(*) FROM workflow_runs WHERE run_id LIKE 'run-unrelated-1-%'`).Scan(&unrelated); err != nil || unrelated != 5 {
		t.Fatalf("unrelated Runs = (%d, %v), want 5", unrelated, err)
	}
}
