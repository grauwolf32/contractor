package explaintest

import (
	"context"
	"maps"
	"slices"
	"strings"
	"testing"

	"github.com/jackc/pgx/v5"
)

// explainedUpdate is EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) output of an
// UPDATE whose InitPlan probes an index of another table, as PostgreSQL 17
// prints it, trimmed to the members this package reads.
const explainedUpdate = `[{
  "Plan": {
    "Node Type": "ModifyTable", "Relation Name": "runs", "Actual Rows": 0, "Actual Loops": 1,
    "Shared Hit Blocks": 40, "Shared Read Blocks": 2,
    "Plans": [
      {"Node Type": "Index Only Scan", "Parent Relationship": "InitPlan", "Relation Name": "audits",
       "Index Name": "audits_project_idx", "Actual Rows": 0, "Actual Loops": 1,
       "Shared Hit Blocks": 2, "Shared Read Blocks": 0},
      {"Node Type": "Nested Loop", "Parent Relationship": "Outer", "Actual Rows": 3, "Actual Loops": 1,
       "Shared Hit Blocks": 30, "Shared Read Blocks": 2,
       "Plans": [
         {"Node Type": "Bitmap Heap Scan", "Relation Name": "runs", "Actual Rows": 3, "Actual Loops": 1,
          "Rows Removed by Index Recheck": 1, "Shared Hit Blocks": 6, "Shared Read Blocks": 0,
          "Plans": [{"Node Type": "Bitmap Index Scan", "Index Name": "runs_state_idx",
                     "Actual Rows": 400, "Actual Loops": 1, "Shared Hit Blocks": 3, "Shared Read Blocks": 0}]},
         {"Node Type": "Index Scan", "Relation Name": "stages", "Index Name": "stages_pkey",
          "Index Cond": "(stage_id = runs.stage_id)", "Actual Rows": 0.5, "Actual Loops": 3, "Rows Removed by Filter": 2,
          "Shared Hit Blocks": 24, "Shared Read Blocks": 2}
       ]}
    ]
  },
  "Triggers": [
    {"Trigger Name": "runs_guard", "Relation": "runs", "Calls": 3},
    {"Trigger Name": "stages_capture", "Relation": "stages", "Calls": 1}
  ]
}]`

func TestParseReadsThePlanAndItsTriggers(t *testing.T) {
	analysis, err := parse([]byte(explainedUpdate))
	if err != nil {
		t.Fatal(err)
	}
	plan := analysis.Plan
	if got := plan.Blocks(); got != 42 {
		t.Errorf("Blocks = %d, want the root's hit and read blocks, 42", got)
	}
	var nodes []string
	for _, node := range plan.Nodes() {
		nodes = append(nodes, node.Type)
	}
	if want := []string{
		"ModifyTable", "Index Only Scan", "Nested Loop", "Bitmap Heap Scan", "Bitmap Index Scan", "Index Scan",
	}; !slices.Equal(nodes, want) {
		t.Errorf("Nodes = %v, want the root first and each node before its children, %v", nodes, want)
	}
	if !plan.Uses("runs_state_idx") || !plan.Uses("audits_project_idx") || plan.Uses("runs_pkey") {
		t.Errorf("Uses does not follow the plan's indexes:\n%s", plan)
	}
	if scans := plan.SeqScans(); len(scans) != 0 {
		t.Errorf("SeqScans = %v, want none", scans)
	}
	want := map[Trigger]int64{{Relation: "runs", Name: "runs_guard"}: 3, {Relation: "stages", Name: "stages_capture"}: 1}
	if !maps.Equal(analysis.Triggers, want) {
		t.Errorf("Triggers = %v, want %v", analysis.Triggers, want)
	}
	rendered := strings.Split(plan.String(), "\n")
	if len(rendered) != 6 || rendered[1] != "  InitPlan: Index Only Scan on audits using audits_project_idx rows=0 removed=0 blocks=2" ||
		rendered[5] != "    Index Scan on stages using stages_pkey where (stage_id = runs.stage_id) rows=2 removed=6 blocks=26" {
		t.Errorf("String =\n%s", plan)
	}
}

func TestReadsOfRejectsATableTheSchemaLacks(t *testing.T) {
	reads := Reads{"runs": 3, "stages": 0}
	if reads.Of("runs") != 3 || reads.Of("stages") != 0 {
		t.Errorf("Of does not return the counts: runs %d, stages %d", reads.Of("runs"), reads.Of("stages"))
	}
	defer func() {
		if recover() == nil {
			t.Error("Of(run) returned a count for a table the schema lacks")
		}
	}()
	reads.Of("run")
}

func TestParseRejectsOutputWithoutOnePlan(t *testing.T) {
	for _, output := range []string{`[]`, `[{"Plan":{}}]`, `{"Plan":{"Node Type":"Result"}}`, `not json`} {
		if _, err := parse([]byte(output)); err == nil {
			t.Errorf("parse(%s) succeeded", output)
		}
	}
}

func TestProbeCountsFiredTriggersPerRelationUntilReset(t *testing.T) {
	probe := &Probe{t: t, fired: map[Trigger]int64{}}
	for trigger, calls := range map[Trigger]int64{
		{Relation: "runs", Name: "guard"}: 2, {Relation: "stages", Name: "guard"}: 1,
		{Relation: "runs", Name: "RI_ConstraintTrigger_c_16500"}: 1,
	} {
		probe.fired[trigger] += calls
	}
	if got := probe.Fired("runs"); !maps.Equal(got, map[string]int64{"guard": 2}) {
		t.Errorf("Fired(runs) = %v", got)
	}
	// A savepoint begun through the probe adds to its counts, before and
	// after a Reset.
	savepoint, err := (&Probe{Tx: savepointTx{}, t: t, fired: probe.fired}).Begin(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	nested, ok := savepoint.(*Probe)
	if !ok {
		t.Fatalf("Begin returned %T, want a *Probe", savepoint)
	}
	nested.fired[Trigger{Relation: "stages", Name: "guard"}]++
	if got := probe.Fired("stages"); !maps.Equal(got, map[string]int64{"guard": 2}) {
		t.Errorf("Fired(stages) with a savepoint = %v", got)
	}
	probe.Reset()
	if got := slices.Collect(maps.Keys(probe.Fired("stages"))); len(got) != 0 {
		t.Errorf("Fired(stages) after Reset = %v", got)
	}
	nested.fired[Trigger{Relation: "stages", Name: "guard"}]++
	if got := probe.Fired("stages"); !maps.Equal(got, map[string]int64{"guard": 1}) {
		t.Errorf("Fired(stages) of a savepoint after Reset = %v", got)
	}
}

func TestProbeSkipsOnlyNamedUtilityStatements(t *testing.T) {
	for sql, want := range map[string]bool{
		"SELECT 1": true, "\n  -- reads\n  with x AS (SELECT 1) SELECT * FROM x": true, "UPDATE\tt SET x = 1": true,
		"INSERT INTO t VALUES (1)": true, "DELETE FROM t": true, "VALUES(1)": true, "(SELECT 1)": true,
		"/* why */ UPDATE t SET x = 1": true, "DO $$ BEGIN END $$": true, "CALL p()": true,
		"SHOW transaction_isolation": false, "set constraints all immediate": false, "RESET ALL": false,
		"SET;": false, "-- only a comment": false, "": false,
	} {
		if got := analyzable(sql); got != want {
			t.Errorf("analyzable(%q) = %t, want %t", sql, got, want)
		}
	}
}

// savepointTx is a transaction whose Begin starts another one.
type savepointTx struct{ pgx.Tx }

func (savepointTx) Begin(context.Context) (pgx.Tx, error) { return savepointTx{}, nil }
