package telemetry_test

import (
	"context"
	"encoding/json"
	"reflect"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/telemetry"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestAllocationResourceProjectionPreservesOptionalResourceStates(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedTelemetryPool(t, ctx)
	runs := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
	reports := telemetry.NewRepository(pool)
	createTelemetryRun(t, ctx, runs)

	for _, test := range []struct {
		name         string
		policy       reporting.PerformanceCollectionPolicy
		report       bool
		released     bool
		expired      bool
		cleaned      bool
		resources    string
		wantStatus   telemetry.AllocationResourceStatus
		wantReason   telemetry.AllocationResourceReason
		wantResource bool
	}{
		{name: "complete", report: true, resources: `{"version":1,"scope":"runtime_process","status":"complete","durationSeconds":10,"cpuUserSeconds":0,"cpuSystemSeconds":0,"rssStartBytes":100,"rssEndBytes":125,"rssPeakObservedBytes":150,"rssSampleCount":2,"maxSampleGapSeconds":10}`, wantStatus: telemetry.AllocationResourceAvailable, wantResource: true},
		{name: "partial", report: true, resources: `{"version":1,"scope":"runtime_process","status":"partial","reason":"read_failed","durationSeconds":0}`, wantStatus: telemetry.AllocationResourcePartial, wantReason: "read_failed", wantResource: true},
		{name: "unavailable", report: true, resources: `{"version":1,"scope":"runtime_process","status":"unavailable","reason":"unsupported_platform"}`, wantStatus: telemetry.AllocationResourceUnavailable, wantReason: "unsupported_platform", wantResource: true},
		{name: "invalid-sentinel", report: true, resources: `{"version":1,"scope":"runtime_process","status":"unavailable","reason":"invalid_report"}`, wantStatus: telemetry.AllocationResourceUnavailable, wantReason: "invalid_report"},
		{name: "absent", report: true, wantStatus: telemetry.AllocationResourceUnavailable, wantReason: telemetry.AllocationResourceReportMissing},
		{name: "null", report: true, resources: `null`, wantStatus: telemetry.AllocationResourceUnavailable, wantReason: telemetry.AllocationResourceReportMissing},
		{name: "negative", report: true, resources: `{"version":1,"scope":"runtime_process","status":"partial","durationSeconds":-1}`, wantStatus: telemetry.AllocationResourceUnavailable, wantReason: telemetry.AllocationResourceReportMissing},
		{name: "unknown-field", report: true, resources: `{"version":1,"scope":"runtime_process","status":"partial","extra":1}`, wantStatus: telemetry.AllocationResourceUnavailable, wantReason: telemetry.AllocationResourceReportMissing},
		{name: "future-version", report: true, resources: `{"version":2,"scope":"runtime_process","status":"partial"}`, wantStatus: telemetry.AllocationResourceUnavailable, wantReason: telemetry.AllocationResourceReportMissing},
		{name: "wrong-type", report: true, resources: `[]`, wantStatus: telemetry.AllocationResourceUnavailable, wantReason: telemetry.AllocationResourceReportMissing},
		{name: "disabled", policy: reporting.PerformanceCollectionDisabled, wantStatus: telemetry.AllocationResourceDisabled},
		{name: "unsupported", policy: reporting.PerformanceCollectionUnsupported, wantStatus: telemetry.AllocationResourceUnsupported},
		{name: "pending", wantStatus: telemetry.AllocationResourcePending},
		{name: "missing", released: true, wantStatus: telemetry.AllocationResourceUnavailable, wantReason: telemetry.AllocationResourceReportMissing},
		// A report that was received and has expired is not a missing one,
		// whether cleanup has deleted it yet or not, and before release too.
		{name: "expired", report: true, expired: true, released: true, resources: `{"version":1,"scope":"runtime_process","status":"partial"}`, wantStatus: telemetry.AllocationResourceUnavailable, wantReason: telemetry.AllocationResourceReportExpired},
		{name: "expired-cleaned", report: true, expired: true, cleaned: true, released: true, resources: `{"version":1,"scope":"runtime_process","status":"partial"}`, wantStatus: telemetry.AllocationResourceUnavailable, wantReason: telemetry.AllocationResourceReportExpired},
		{name: "expired-unreleased", report: true, expired: true, resources: `{"version":1,"scope":"runtime_process","status":"partial"}`, wantStatus: telemetry.AllocationResourceUnavailable, wantReason: telemetry.AllocationResourceReportExpired},
	} {
		t.Run(test.name, func(t *testing.T) {
			policy := test.policy
			if policy == "" {
				policy = reporting.PerformanceCollectionRequested
			}
			allocation := "allocation-" + test.name
			stage := createTelemetryStageWithPolicy(t, ctx, runs, "stage-"+test.name, allocation, "session-"+test.name, policy)
			finishTelemetryStage(t, ctx, runs, stage)
			if test.released {
				if err := runs.MarkStageAllocationReleased(ctx, allocation); err != nil {
					t.Fatal(err)
				}
			}
			if test.report {
				envelope := allocationEnvelope(stage, allocation, time.Now().UTC(), true)
				envelope.PerformanceCollectionPolicy = policy
				payload, err := json.Marshal(envelope.Report)
				if err != nil {
					t.Fatal(err)
				}
				// Seed old optional blocks as immutable reports. The production
				// writer would normalize malformed resources before persisting them.
				var document map[string]json.RawMessage
				if err := json.Unmarshal(payload, &document); err != nil {
					t.Fatal(err)
				}
				if test.resources != "" {
					var runtimeReport map[string]json.RawMessage
					if err := json.Unmarshal(document["runtime"], &runtimeReport); err != nil {
						t.Fatal(err)
					}
					runtimeReport["resources"] = json.RawMessage(test.resources)
					document["runtime"], err = json.Marshal(runtimeReport)
					if err != nil {
						t.Fatal(err)
					}
				}
				payload, err = json.Marshal(document)
				if err != nil {
					t.Fatal(err)
				}
				insertAllocationResourceFixture(t, ctx, pool, envelope, payload, test.expired)
			}
			if test.cleaned {
				if _, err := reports.CleanupExpired(ctx, time.Now().UTC(), 10_000); err != nil {
					t.Fatal(err)
				}
				var stored int
				if err := pool.QueryRow(ctx, `SELECT count(*) FROM allocation_execution_reports WHERE allocation_id = $1`, allocation).Scan(&stored); err != nil || stored != 0 {
					t.Fatalf("cleanup kept %d reports (%v)", stored, err)
				}
			}

			history, err := reports.ListAllocationResourceHistory(ctx, telemetry.AllocationResourceHistoryParams{OwnerID: "user", Limit: 101})
			if err != nil {
				t.Fatal(err)
			}
			var item *telemetry.AllocationResourceSummary
			for i := range history {
				if history[i].AllocationID == allocation {
					item = &history[i]
				}
			}
			if item == nil || item.Status != test.wantStatus || item.CollectionPolicy != policy ||
				(item.Resources != nil) != test.wantResource {
				t.Fatalf("resource summary = %+v", item)
			}
			if (item.Reason == nil) != (test.wantReason == "") || item.Reason != nil && *item.Reason != test.wantReason {
				t.Fatalf("resource reason = %v, want %q", item.Reason, test.wantReason)
			}
			if test.wantResource {
				var want reporting.RuntimeResources
				if err := json.Unmarshal([]byte(test.resources), &want); err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(item.Resources, &want) {
					t.Fatalf("resource observations changed: got %+v, want %+v", item.Resources, want)
				}
			}
			batch, err := reports.ListStageAllocationResources(ctx, "user", []string{stage})
			var stageItem *telemetry.AllocationResourceSummary
			for i := range batch[stage] {
				if batch[stage][i].AllocationID == allocation {
					stageItem = &batch[stage][i]
				}
			}
			if err != nil || !reflect.DeepEqual(stageItem, item) {
				t.Fatalf("stage summary differs from history: %+v, %v", batch, err)
			}
			foreign, err := reports.ListStageAllocationResources(ctx, "another-owner", []string{stage})
			if err != nil || len(foreign) != 0 {
				t.Fatalf("foreign stage resources = (%+v, %v)", foreign, err)
			}
		})
	}
}

// RecordAllocationReport applies the one resource policy definition
// (reporting.RuntimeReport.ApplyResourcePolicy) before it persists a report,
// with the outcomes the Control Plane client applies to a Runtime response.
func TestRecordAllocationReportAppliesThePinnedResourcePolicy(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedTelemetryPool(t, ctx)
	runs := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
	reports := telemetry.NewRepository(pool)
	createTelemetryRun(t, ctx, runs)
	duration, negative := 1.0, -1.0
	decodeError := reporting.ResourceInvalidReport
	reason := reporting.ResourceInvalidReport
	valid := &reporting.RuntimeResources{Version: 1, Scope: "runtime_process", Status: reporting.ResourcePartial, DurationSeconds: &duration}
	invalid := &reporting.RuntimeResources{Version: 1, Scope: "runtime_process", Status: reporting.ResourcePartial, DurationSeconds: &negative}
	sentinel := &reporting.RuntimeResources{Version: 1, Scope: "runtime_process", Status: reporting.ResourceUnavailable, Reason: &reason}
	for _, test := range []struct {
		name      string
		policy    reporting.PerformanceCollectionPolicy
		resources *reporting.RuntimeResources
		decodeErr *reporting.ResourceReason
		want      *reporting.RuntimeResources
	}{
		{name: "requested-valid", policy: reporting.PerformanceCollectionRequested, resources: valid, want: valid},
		{name: "requested-malformed", policy: reporting.PerformanceCollectionRequested, decodeErr: &decodeError, want: sentinel},
		{name: "requested-invalid", policy: reporting.PerformanceCollectionRequested, resources: invalid, want: sentinel},
		{name: "requested-absent", policy: reporting.PerformanceCollectionRequested},
		{name: "disabled-valid", policy: reporting.PerformanceCollectionDisabled, resources: valid},
		{name: "unsupported-malformed", policy: reporting.PerformanceCollectionUnsupported, decodeErr: &decodeError},
	} {
		t.Run(test.name, func(t *testing.T) {
			allocation := "allocation-policy-" + test.name
			stage := createTelemetryStageWithPolicy(t, ctx, runs, "stage-policy-"+test.name, allocation, "session-policy-"+test.name, test.policy)
			envelope := allocationEnvelope(stage, allocation, time.Now().UTC(), true)
			envelope.PerformanceCollectionPolicy = test.policy
			envelope.Report.Runtime.Resources = test.resources
			envelope.Report.Runtime.ResourcesError = test.decodeErr
			if err := reports.RecordAllocationReport(ctx, envelope); err != nil {
				t.Fatal(err)
			}
			stored, err := reports.ListAllocationReports(ctx, stage)
			if err != nil || len(stored) != 1 {
				t.Fatalf("stored reports = (%d, %v)", len(stored), err)
			}
			runtime := stored[0].Report.Runtime
			if !reflect.DeepEqual(runtime.Resources, test.want) || runtime.ResourcesError != nil || !runtime.Complete {
				t.Fatalf("stored resources = %+v, diagnostic %v; want %+v", runtime.Resources, runtime.ResourcesError, test.want)
			}
		})
	}
}

func TestAllocationResourceProjectionSelectsEffectiveReportAndPreservesPagination(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedTelemetryPool(t, ctx)
	runs := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
	reports := telemetry.NewRepository(pool)
	createTelemetryRun(t, ctx, runs)
	now := time.Now().UTC()
	for _, name := range []string{"selected", "expired"} {
		allocation := "allocation-" + name
		stage := createTelemetryStageWithPolicy(t, ctx, runs, "stage-"+name, allocation, "session-"+name, reporting.PerformanceCollectionRequested)
		finishTelemetryStage(t, ctx, runs, stage)
		// An older complete report wins over a newer partial report, but
		// only while the complete report is still within retention.
		for _, value := range []struct {
			name     string
			complete bool
			received time.Time
			duration float64
		}{
			{"partial", false, now.Add(-time.Minute), 20},
			{"complete", true, now.Add(-2 * time.Minute), 10},
		} {
			envelope := allocationEnvelope(stage, allocation, value.received, value.complete)
			envelope.Report.ReportID += "-" + value.name
			envelope.PerformanceCollectionPolicy = reporting.PerformanceCollectionRequested
			envelope.Report.Runtime.Resources = &reporting.RuntimeResources{Version: 1, Scope: "runtime_process", Status: reporting.ResourcePartial, DurationSeconds: &value.duration}
			insertAllocationResourceFixture(t, ctx, pool, envelope, nil, name == "expired" && value.complete)
		}
	}
	createTelemetryStage(t, ctx, runs, "stage-active", "allocation-active", "session-active")
	first, err := reports.ListAllocationResourceHistory(ctx, telemetry.AllocationResourceHistoryParams{OwnerID: "user", RunID: "run-telemetry", Limit: 1})
	if err != nil || len(first) != 1 || first[0].AllocationID != "allocation-expired" ||
		first[0].Resources == nil || first[0].Resources.DurationSeconds == nil || *first[0].Resources.DurationSeconds != 20 {
		t.Fatalf("first history page = (%+v, %v)", first, err)
	}
	second, err := reports.ListAllocationResourceHistory(ctx, telemetry.AllocationResourceHistoryParams{
		OwnerID: "user", RunID: "run-telemetry", Limit: 1,
		UpperFinishedAt: &first[0].FinishedAt, UpperAllocationID: first[0].AllocationID,
		AfterFinishedAt: &first[0].FinishedAt, AfterAllocationID: first[0].AllocationID,
	})
	if err != nil || len(second) != 1 || second[0].AllocationID != "allocation-selected" ||
		second[0].Resources == nil || second[0].Resources.DurationSeconds == nil || *second[0].Resources.DurationSeconds != 10 {
		t.Fatalf("second history page = (%+v, %v)", second, err)
	}
	batch, err := reports.ListStageAllocationResources(ctx, "user", []string{"stage-selected", "stage-expired", "stage-active"})
	if err != nil || len(batch) != 2 || len(batch["stage-selected"]) != 1 || len(batch["stage-expired"]) != 1 ||
		!reflect.DeepEqual(batch["stage-selected"][0], second[0]) || !reflect.DeepEqual(batch["stage-expired"][0], first[0]) {
		t.Fatalf("effective stage resources = (%+v, %v)", batch, err)
	}
}

func TestAllocationResourceProjectionRejectsInvalidReportIdentity(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedTelemetryPool(t, ctx)
	runs := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
	reports := telemetry.NewRepository(pool)
	createTelemetryRun(t, ctx, runs)
	// These JSON identities pass the database's text-based CHECK constraint
	// (or its SQL NULL semantics), but must still fail typed read validation.
	stage := createTelemetryStage(t, ctx, runs, "stage-identity", "123", "session-identity")
	finishTelemetryStage(t, ctx, runs, stage)
	for _, test := range []struct{ name, identity string }{
		{"numeric", `123`}, {"null", `null`}, {"missing", ""},
	} {
		t.Run(test.name, func(t *testing.T) {
			envelope := allocationEnvelope(stage, "123", time.Now().UTC(), true)
			payload, err := json.Marshal(envelope.Report)
			if err != nil {
				t.Fatal(err)
			}
			var document map[string]json.RawMessage
			if err := json.Unmarshal(payload, &document); err != nil {
				t.Fatal(err)
			}
			delete(document, "allocationId")
			if test.identity != "" {
				document["allocationId"] = json.RawMessage(test.identity)
			}
			payload, err = json.Marshal(document)
			if err != nil {
				t.Fatal(err)
			}
			insertAllocationResourceFixture(t, ctx, pool, envelope, payload, false)
			t.Cleanup(func() {
				if _, err := pool.Exec(ctx, `DELETE FROM allocation_execution_reports WHERE allocation_id='123'`); err != nil {
					t.Error(err)
				}
			})
			if _, err := reports.ListAllocationResourceHistory(ctx, telemetry.AllocationResourceHistoryParams{OwnerID: "user", Limit: 10}); err == nil {
				t.Fatal("history accepted invalid report identity")
			}
			if _, err := reports.ListStageAllocationResources(ctx, "user", []string{stage}); err == nil {
				t.Fatal("stage resources accepted invalid report identity")
			}
		})
	}
}

// Direct insertion models already persisted legacy rows and explicit retention
// without weakening the database's report immutability or writer validation.
func insertAllocationResourceFixture(
	t *testing.T, ctx context.Context, pool *pgxpool.Pool,
	envelope telemetry.AllocationReportEnvelope, payload []byte, expired bool,
) {
	t.Helper()
	if payload == nil {
		var err error
		payload, err = json.Marshal(envelope.Report)
		if err != nil {
			t.Fatal(err)
		}
	}
	expiresAt := time.Now().UTC().Add(24 * time.Hour)
	if expired {
		expiresAt = time.Now().UTC().Add(-24 * time.Hour)
	}
	if _, err := pool.Exec(ctx, `INSERT INTO allocation_execution_reports
(report_id,stage_execution_id,allocation_id,logical_agent_name,report_schema_version,report,received_at,expires_at)
VALUES ($1,$2,$3,$4,$5,$6::jsonb,$7,$8)`, envelope.Report.ReportID, envelope.StageExecutionID,
		envelope.AllocationID, envelope.LogicalAgentName, envelope.ReportSchemaVersion, payload, envelope.ReceivedAt, expiresAt); err != nil {
		t.Fatal(err)
	}
}
