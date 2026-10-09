package evalstore

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func TestPostgresClaimSkipsIdlePausedAndFindsActionableWork(t *testing.T) {
	pool := testPool(t)
	scope := setupProject(t, pool, "claim-owner", "claim-project")
	ctx := t.Context()
	_, err := pool.Exec(ctx, `
INSERT INTO eval_experiments
  (experiment_id,owner_id,project_id,portable_id,control_mode,name,state,max_in_flight,wall_ms,started_at,deadline_at)
SELECT 'paused-'||n, $1, $2, 'paused-'||n, 'external', 'paused fixture', 'paused', 1, 1000,
       clock_timestamp()-interval '7 days', clock_timestamp()+interval '6 days'
FROM generate_series(1,40) AS n`, scope.OwnerID, scope.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	_, err = pool.Exec(ctx, `
INSERT INTO eval_experiments
  (experiment_id,owner_id,project_id,portable_id,control_mode,name,state,max_in_flight,wall_ms)
VALUES ('running-one',$1,$2,'running-one','external','running fixture','running',1,1000)`, scope.OwnerID, scope.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	_, err = pool.Exec(ctx, `INSERT INTO eval_controller_claims(experiment_id)
SELECT experiment_id FROM eval_experiments`)
	if err != nil {
		t.Fatal(err)
	}
	store := NewPostgresStore(pool)
	for poll := 0; poll < 10; poll++ {
		claims, err := store.Claim(ctx, "claim-controller", time.Minute, 4)
		if err != nil || len(claims) != 1 || claims[0].ExperimentID != "running-one" {
			t.Fatalf("poll %d claims = %+v, %v", poll, claims, err)
		}
		if err := store.ReleaseClaim(ctx, claims[0]); err != nil {
			t.Fatal(err)
		}
	}
	_, err = pool.Exec(ctx, `
INSERT INTO eval_experiments
  (experiment_id,owner_id,project_id,portable_id,control_mode,name,state,max_in_flight,wall_ms,started_at,deadline_at)
VALUES ('paused-expired',$1,$2,'paused-expired','external','expired pause','paused',1,1000,
        clock_timestamp()-interval '7 days',clock_timestamp()-interval '1 day')`, scope.OwnerID, scope.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `INSERT INTO eval_controller_claims(experiment_id) VALUES ('paused-expired')`); err != nil {
		t.Fatal(err)
	}
	_, err = pool.Exec(ctx, `INSERT INTO eval_commands
  (command_id,experiment_id,actor_id,kind,state,accepted_revision)
VALUES ('resume-command','paused-2','claim-owner','resume','accepted',1),
       ('cancel-command','paused-3','claim-owner','cancel','accepted',1)`)
	if err != nil {
		t.Fatal(err)
	}
	_, err = pool.Exec(ctx, `INSERT INTO eval_projection_queue(experiment_id,revision,published_revision)
VALUES ('paused-4',2,1)`)
	if err != nil {
		t.Fatal(err)
	}
	claims, err := store.Claim(ctx, "claim-controller", time.Minute, 10)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]bool{
		"running-one": true, "paused-expired": true, "paused-2": true,
		"paused-3": true, "paused-4": true,
	}
	if len(claims) != len(want) {
		t.Fatalf("actionable claims = %+v, want %v", claims, want)
	}
	for _, claim := range claims {
		if !want[claim.ExperimentID] {
			t.Fatalf("unexpected claim %+v", claim)
		}
		delete(want, claim.ExperimentID)
	}
	if len(want) != 0 {
		t.Fatalf("missing actionable claims: %v", want)
	}
}

func TestPostgresClaimPlanSkipsRetainedTerminalExperiments(t *testing.T) {
	pool := testPool(t)
	scope := setupProject(t, pool, "claim-plan-owner", "claim-plan-project")
	ctx := t.Context()
	_, err := pool.Exec(ctx, `
INSERT INTO eval_experiments
  (experiment_id,owner_id,project_id,portable_id,control_mode,name,state,max_in_flight,wall_ms)
SELECT 'finished-'||n, $1, $2, 'finished-'||n, 'external', 'finished fixture', 'finished', 1, 1000
FROM generate_series(1,20000) AS n`, scope.OwnerID, scope.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	_, err = pool.Exec(ctx, `INSERT INTO eval_experiments
  (experiment_id,owner_id,project_id,portable_id,control_mode,name,state,max_in_flight,wall_ms)
VALUES ('running-one',$1,$2,'running-one','external','running fixture','running',1,1000)`, scope.OwnerID, scope.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	_, err = pool.Exec(ctx, `INSERT INTO eval_controller_claims(experiment_id)
SELECT experiment_id FROM eval_experiments`)
	if err != nil {
		t.Fatal(err)
	}
	for _, table := range []string{"eval_experiments", "eval_controller_claims", "eval_commands", "eval_projection_queue"} {
		if _, err := pool.Exec(ctx, "ANALYZE "+table); err != nil {
			t.Fatal(err)
		}
	}
	var raw string
	if err := pool.QueryRow(ctx, "EXPLAIN (FORMAT JSON) "+claimStatement,
		"plan-controller", int64(time.Minute/time.Millisecond), 4).Scan(&raw); err != nil {
		t.Fatal(err)
	}
	var plan []any
	if err := json.Unmarshal([]byte(raw), &plan); err != nil {
		t.Fatal(err)
	}
	if planScansExperimentTable(plan) || planScansTable(plan, "eval_controller_claims") {
		t.Fatalf("claim plan scans retained terminal experiments or claims: %s", raw)
	}
	claims, err := NewPostgresStore(pool).Claim(ctx, "plan-controller", time.Minute, 4)
	if err != nil || len(claims) != 1 || claims[0].ExperimentID != "running-one" {
		t.Fatalf("claim from terminal-heavy table = %+v, %v", claims, err)
	}
}

func TestPostgresClaimPlanReadsPausedDeadlinesAsRange(t *testing.T) {
	pool := testPool(t)
	scope := setupProject(t, pool, "claim-paused-owner", "claim-paused-project")
	ctx := t.Context()
	// Only the expired pauses are actionable; the rest wait for their deadline.
	_, err := pool.Exec(ctx, `
INSERT INTO eval_experiments
  (experiment_id,owner_id,project_id,portable_id,control_mode,name,state,max_in_flight,wall_ms,started_at,deadline_at)
SELECT 'paused-'||n, $1, $2, 'paused-'||n, 'external', 'paused fixture', 'paused', 1, 1000,
       clock_timestamp()-interval '7 days',
       CASE WHEN n <= 3 THEN clock_timestamp()-interval '1 day'
            ELSE clock_timestamp()+n*interval '1 minute' END
FROM generate_series(1,2000) AS n`, scope.OwnerID, scope.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	_, err = pool.Exec(ctx, `INSERT INTO eval_controller_claims(experiment_id)
SELECT experiment_id FROM eval_experiments`)
	if err != nil {
		t.Fatal(err)
	}
	for _, table := range []string{"eval_experiments", "eval_controller_claims", "eval_commands", "eval_projection_queue"} {
		if _, err := pool.Exec(ctx, "ANALYZE "+table); err != nil {
			t.Fatal(err)
		}
	}
	var raw string
	if err := pool.QueryRow(ctx, "EXPLAIN (FORMAT JSON) "+claimStatement,
		"plan-controller", int64(time.Minute/time.Millisecond), 10).Scan(&raw); err != nil {
		t.Fatal(err)
	}
	var plan []any
	if err := json.Unmarshal([]byte(raw), &plan); err != nil {
		t.Fatal(err)
	}
	conditions := planIndexConditions(plan, "eval_experiments_claim_paused_deadline_idx")
	if len(conditions) == 0 || planScansExperimentTable(plan) {
		t.Fatalf("claim plan does not read paused experiments through their deadline index: %s", raw)
	}
	for _, condition := range conditions {
		if !strings.Contains(condition, "deadline_at") {
			t.Fatalf("claim plan reads every paused experiment instead of a deadline range: %s", raw)
		}
	}
	claims, err := NewPostgresStore(pool).Claim(ctx, "plan-controller", time.Minute, 10)
	if err != nil {
		t.Fatal(err)
	}
	got := map[string]bool{}
	for _, claim := range claims {
		got[claim.ExperimentID] = true
	}
	if len(claims) != 3 || !got["paused-1"] || !got["paused-2"] || !got["paused-3"] {
		t.Fatalf("expired paused claims = %+v", claims)
	}
}

// planIndexConditions returns the index condition of every plan node reading index.
func planIndexConditions(value any, index string) []string {
	var conditions []string
	switch node := value.(type) {
	case map[string]any:
		if node["Index Name"] == index {
			condition, _ := node["Index Cond"].(string)
			conditions = append(conditions, condition)
		}
		for _, child := range node {
			conditions = append(conditions, planIndexConditions(child, index)...)
		}
	case []any:
		for _, child := range node {
			conditions = append(conditions, planIndexConditions(child, index)...)
		}
	}
	return conditions
}

func planScansExperimentTable(value any) bool {
	return planScansTable(value, "eval_experiments")
}

func planScansTable(value any, table string) bool {
	switch node := value.(type) {
	case map[string]any:
		if node["Node Type"] == "Seq Scan" && node["Relation Name"] == table {
			return true
		}
		for _, child := range node {
			if planScansTable(child, table) {
				return true
			}
		}
	case []any:
		for _, child := range node {
			if planScansTable(child, table) {
				return true
			}
		}
	}
	return false
}

func TestPostgresClaimLocksOnlyItsBoundedBatch(t *testing.T) {
	pool := testPool(t)
	scope := setupProject(t, pool, "bounded-owner", "bounded-project")
	ctx := t.Context()
	if _, err := pool.Exec(ctx, `INSERT INTO eval_experiments
(experiment_id,owner_id,project_id,portable_id,control_mode,name,state,max_in_flight,wall_ms)
SELECT 'running-'||n, $1, $2, 'running-'||n, 'external', 'running', 'running', 1, 1000
FROM generate_series(1,3) n`, scope.OwnerID, scope.ProjectID); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `INSERT INTO eval_controller_claims(experiment_id) SELECT experiment_id FROM eval_experiments`); err != nil {
		t.Fatal(err)
	}
	tx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer tx.Rollback(ctx)
	first, err := NewTxStore(tx).Claim(ctx, "first", time.Minute, 1)
	if err != nil || len(first) != 1 {
		t.Fatalf("first bounded claim: %+v %v", first, err)
	}
	// A second holder skips the first locked tuple and claims both remaining
	// rows immediately. Ordering must not lock rows beyond the first limit.
	bounded, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	second, err := NewPostgresStore(pool).Claim(bounded, "second", time.Minute, 2)
	if err != nil || len(second) != 2 {
		t.Fatalf("second holder blocked or lost candidates: %+v %v", second, err)
	}
	for _, claim := range second {
		if claim.ExperimentID == first[0].ExperimentID {
			t.Fatal("holders shared a locked claim")
		}
	}
}
