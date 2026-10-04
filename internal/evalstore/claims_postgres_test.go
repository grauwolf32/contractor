package evalstore

import (
	"encoding/json"
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
	if planScansExperimentTable(plan) {
		t.Fatalf("claim plan scans retained terminal experiments: %s", raw)
	}
	claims, err := NewPostgresStore(pool).Claim(ctx, "plan-controller", time.Minute, 4)
	if err != nil || len(claims) != 1 || claims[0].ExperimentID != "running-one" {
		t.Fatalf("claim from terminal-heavy table = %+v, %v", claims, err)
	}
}

func planScansExperimentTable(value any) bool {
	switch node := value.(type) {
	case map[string]any:
		if node["Node Type"] == "Seq Scan" && node["Relation Name"] == "eval_experiments" {
			return true
		}
		for _, child := range node {
			if planScansExperimentTable(child) {
				return true
			}
		}
	case []any:
		for _, child := range node {
			if planScansExperimentTable(child) {
				return true
			}
		}
	}
	return false
}
