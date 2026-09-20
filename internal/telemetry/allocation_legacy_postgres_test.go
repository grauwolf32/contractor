package telemetry_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

func TestAllocationReadersRejectMissingPersistedAuthority(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool := isolatedTelemetryPool(t, ctx)
	runs := runstore.NewPostgresStore(pool)
	createTelemetryRun(t, ctx, runs)
	stage := createTelemetryStage(t, ctx, runs, "stage-authority", "allocation-current", "session-authority")
	finishTelemetryStage(t, ctx, runs, stage)
	if current, err := runs.ListStageAllocations(ctx, stage); err != nil || len(current) != 1 || current[0].RuntimeConfiguration == nil {
		t.Fatalf("current allocation = %+v, %v", current, err)
	}
	for _, test := range []struct {
		name                          string
		missingRuntime, missingPolicy bool
	}{
		{"runtime", true, false},
		{"policy", false, true},
		{"both", true, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			tx, err := pool.Begin(ctx)
			if err != nil {
				t.Fatal(err)
			}
			defer tx.Rollback(ctx)
			// Historical rows are inserted directly; current writers reject them.
			if _, err := tx.Exec(ctx, `INSERT INTO stage_allocations (
allocation_id,stage_execution_id,logical_agent_name,namespace,agent_template_ref,worker_runtime_ref,
runtime_agent_instance_id,runtime_agent_id,runtime_agent_label_revision,
runtime_configuration_schema_version,runtime_configuration,performance_collection_policy)
SELECT 'allocation-historical',stage_execution_id,'historical',namespace,agent_template_ref,worker_runtime_ref,
runtime_agent_instance_id,CASE WHEN $1 THEN NULL ELSE runtime_agent_id END,
CASE WHEN $1 THEN NULL ELSE runtime_agent_label_revision END,
CASE WHEN $1 THEN NULL ELSE runtime_configuration_schema_version END,
CASE WHEN $1 THEN NULL ELSE runtime_configuration END,
CASE WHEN $2 THEN NULL ELSE performance_collection_policy END
FROM stage_allocations WHERE allocation_id='allocation-current'`, test.missingRuntime, test.missingPolicy); err != nil {
				t.Fatal(err)
			}
			const snapshot = `SELECT jsonb_agg(to_jsonb(a) ORDER BY allocation_id)::text FROM stage_allocations a`
			var before, after string
			if err := tx.QueryRow(ctx, snapshot).Scan(&before); err != nil {
				t.Fatal(err)
			}
			for range 2 {
				store := runstore.NewPostgresStore(tx)
				if rows, err := store.ListStageAllocations(ctx, stage); err == nil || rows != nil {
					t.Fatalf("incomplete allocation read = %+v, %v", rows, err)
				}
				if rows, err := store.ListStageAllocationsBatch(ctx, []string{stage}); err == nil || rows != nil {
					t.Fatalf("incomplete allocation batch = %+v, %v", rows, err)
				}
				if test.missingPolicy {
					reports := telemetry.NewRepository(tx)
					if rows, err := reports.ListAllocationResourceHistory(ctx, telemetry.AllocationResourceHistoryParams{OwnerID: "user", Limit: 101}); err == nil || rows != nil {
						t.Fatalf("missing-policy history = %+v, %v", rows, err)
					}
					if rows, err := reports.ListStageAllocationResources(ctx, "user", []string{stage}); err == nil || rows != nil {
						t.Fatalf("missing-policy stage resources = %+v, %v", rows, err)
					}
					if rows, err := reports.ListStageAllocationResources(ctx, "foreign", []string{stage}); err != nil || len(rows) != 0 {
						t.Fatalf("foreign stage resources = %+v, %v", rows, err)
					}
				}
			}
			if err := tx.QueryRow(ctx, snapshot).Scan(&after); err != nil || before != after {
				t.Fatalf("rejected reads changed stored authority: %v", err)
			}
		})
	}
}

func TestAllocationReportRejectsLegacyPolicyWithoutChangingRetainedReport(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool := isolatedTelemetryPool(t, ctx)
	runs := runstore.NewPostgresStore(pool)
	reports := telemetry.NewRepository(pool)
	createTelemetryRun(t, ctx, runs)
	stage := createTelemetryStage(t, ctx, runs, "stage-policy", "allocation-policy", "session-policy")
	current := allocationEnvelope(stage, "allocation-policy", time.Now().UTC(), true)
	if err := reports.RecordAllocationReport(ctx, current); err != nil {
		t.Fatal(err)
	}
	const snapshot = `SELECT jsonb_agg(to_jsonb(r) ORDER BY report_id)::text FROM allocation_execution_reports r`
	var before, after string
	if err := pool.QueryRow(ctx, snapshot).Scan(&before); err != nil {
		t.Fatal(err)
	}
	for _, policy := range []contracts.PerformanceCollectionPolicy{"", "legacy", "unknown"} {
		invalid := current
		invalid.PerformanceCollectionPolicy = policy
		for range 2 {
			if err := reports.RecordAllocationReport(ctx, invalid); !errors.Is(err, telemetry.ErrInvalid) {
				t.Fatalf("policy %q error = %v", policy, err)
			}
		}
	}
	if err := reports.RecordAllocationReport(ctx, current); err != nil {
		t.Fatalf("current report replay: %v", err)
	}
	if err := pool.QueryRow(ctx, snapshot).Scan(&after); err != nil || before != after {
		t.Fatalf("policy rejection or replay changed the retained report: %v", err)
	}
}
