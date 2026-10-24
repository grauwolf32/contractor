package explaintest

import (
	"maps"
	"slices"
	"strings"
	"testing"
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
          "Actual Rows": 0.5, "Actual Loops": 3, "Rows Removed by Filter": 2,
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
	// The ModifyTable node names runs but scans nothing; the bitmap index
	// entries are not rows.
	for relation, want := range map[string]int64{"runs": 4, "stages": 8, "audits": 0, "other": 0} {
		if got := plan.Examined(relation); got != want {
			t.Errorf("Examined(%s) = %d, want %d", relation, got, want)
		}
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
		rendered[5] != "    Index Scan on stages using stages_pkey rows=2 removed=6 blocks=26" {
		t.Errorf("String =\n%s", plan)
	}
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
	probe.Reset()
	if got := slices.Collect(maps.Keys(probe.Fired("stages"))); len(got) != 0 {
		t.Errorf("Fired(stages) after Reset = %v", got)
	}
}
