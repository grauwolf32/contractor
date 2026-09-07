package telemetry

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

var (
	ErrInvalid  = errors.New("invalid telemetry")
	ErrConflict = errors.New("telemetry conflict")
	ErrNotFound = errors.New("telemetry not found")
)

type AllocationReportEnvelope struct {
	StageExecutionID            string
	AllocationID                string
	LogicalAgentName            string
	ReportSchemaVersion         string
	Report                      contracts.AllocationFinalReport
	PerformanceCollectionPolicy contracts.PerformanceCollectionPolicy
	Secrets                     []string
	ReceivedAt                  time.Time
}

type PlannerReportEnvelope struct {
	StageExecutionID    string
	SessionID           string
	InvocationID        string
	StartedAt           time.Time
	FinishedAt          time.Time
	ReportSchemaVersion string
	Report              contracts.ExecutionReport
	Secrets             []string
	ReceivedAt          time.Time
}

type StoredAllocationReport struct {
	AllocationReportEnvelope
	ExpiresAt time.Time
}

type StoredPlannerReport struct {
	PlannerReportEnvelope
	ExpiresAt time.Time
}

type StageMetricsRecord struct {
	StageExecutionID     string
	MetricsSchemaVersion string
	Metrics              contracts.StageMetrics
	Summary              Summary
	UpdatedAt            time.Time
	ExpiresAt            time.Time
}

type Repository struct {
	db persistencepostgres.DBTX
}

func NewRepository(db persistencepostgres.DBTX) *Repository {
	return &Repository{db: db}
}

func (r *Repository) RecordAllocationReport(
	ctx context.Context,
	envelope AllocationReportEnvelope,
) error {
	if err := validateAllocationEnvelope(envelope); err != nil {
		return err
	}
	normalizeAllocationResources(&envelope.Report.Runtime, envelope.PerformanceCollectionPolicy)
	report, err := NewPolicy(envelope.Secrets...).NormalizeAllocationReport(envelope.Report)
	if err != nil {
		return fmt.Errorf("%w: %v", ErrInvalid, err)
	}
	if report.AllocationID != envelope.AllocationID {
		return fmt.Errorf("%w: allocation report identifies another allocation", ErrInvalid)
	}
	encoded, err := json.Marshal(report)
	if err != nil {
		return fmt.Errorf("encode allocation report: %w", err)
	}
	receivedAt := envelope.ReceivedAt
	if receivedAt.IsZero() {
		receivedAt = time.Now().UTC()
	}
	if !report.Worker.Complete || !report.Runtime.Complete {
		existing, existingErr := r.allocationReportForParticipant(
			ctx, envelope.StageExecutionID, envelope.LogicalAgentName,
		)
		if existingErr == nil && existing.AllocationID == envelope.AllocationID &&
			existing.Report.ReportID != report.ReportID &&
			existing.Report.Worker.Complete && existing.Report.Runtime.Complete {
			return nil
		}
		if existingErr != nil && !errors.Is(existingErr, ErrNotFound) {
			return existingErr
		}
	}
	command, err := r.db.Exec(ctx, `
INSERT INTO allocation_execution_reports (
    report_id, stage_execution_id, allocation_id, logical_agent_name,
    report_schema_version, report, received_at, expires_at
) VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::timestamptz, $7::timestamptz + interval '30 days')
ON CONFLICT DO NOTHING`,
		report.ReportID, envelope.StageExecutionID, envelope.AllocationID,
		envelope.LogicalAgentName, envelope.ReportSchemaVersion, encoded, receivedAt,
	)
	if err != nil {
		return classifyWriteError("record allocation report", err)
	}
	if command.RowsAffected() == 1 {
		return nil
	}
	existing, err := r.allocationReportForIdentity(
		ctx, report.ReportID, envelope.StageExecutionID, envelope.AllocationID,
		envelope.LogicalAgentName,
	)
	if err != nil {
		return err
	}
	if existing.StageExecutionID != envelope.StageExecutionID ||
		existing.AllocationID != envelope.AllocationID ||
		existing.LogicalAgentName != envelope.LogicalAgentName ||
		existing.ReportSchemaVersion != envelope.ReportSchemaVersion ||
		!sameJSON(existing.Report, report) {
		return fmt.Errorf("record allocation report: %w", ErrConflict)
	}
	return nil
}

func normalizeAllocationResources(
	report *contracts.RuntimeReport,
	policy contracts.PerformanceCollectionPolicy,
) {
	if policy != contracts.PerformanceCollectionRequested {
		report.Resources = nil
		report.ResourcesError = nil
		return
	}
	if report.ResourcesError != nil || (report.Resources != nil && report.Resources.Validate() != nil) {
		reason := contracts.ResourceInvalidReport
		report.Resources = &contracts.RuntimeResources{
			Version: contracts.PerformanceMetricsVersion, Scope: "runtime_process",
			Status: contracts.ResourceUnavailable, Reason: &reason,
		}
		report.ResourcesError = nil
	}
}

func (r *Repository) ListAllocationReports(
	ctx context.Context,
	stageExecutionID string,
) ([]StoredAllocationReport, error) {
	if err := requireText("stageExecutionID", stageExecutionID); err != nil {
		return nil, err
	}
	rows, err := r.db.Query(ctx, `
WITH effective AS (
    SELECT DISTINCT ON (logical_agent_name)
           stage_execution_id, allocation_id, logical_agent_name,
           report_schema_version, report, received_at, expires_at
    FROM allocation_execution_reports
    WHERE stage_execution_id = $1
    ORDER BY logical_agent_name,
             ((report->'worker'->>'complete')::boolean
              AND (report->'runtime'->>'complete')::boolean) DESC,
             received_at DESC, report_id
)
SELECT stage_execution_id, allocation_id, logical_agent_name,
       report_schema_version, report, received_at, expires_at
FROM effective ORDER BY logical_agent_name`, stageExecutionID)
	if err != nil {
		return nil, fmt.Errorf("list allocation reports: %w", err)
	}
	defer rows.Close()
	result := make([]StoredAllocationReport, 0)
	for rows.Next() {
		item, err := scanAllocationReport(rows)
		if err != nil {
			return nil, fmt.Errorf("list allocation reports: %w", err)
		}
		result = append(result, item)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("list allocation reports: %w", err)
	}
	return result, nil
}

func (r *Repository) RecordPlannerReport(
	ctx context.Context,
	envelope PlannerReportEnvelope,
) error {
	if err := validatePlannerEnvelope(envelope); err != nil {
		return err
	}
	envelope.StartedAt = envelope.StartedAt.UTC().Truncate(time.Microsecond)
	envelope.FinishedAt = envelope.FinishedAt.UTC().Truncate(time.Microsecond)
	report, err := NewPolicy(envelope.Secrets...).NormalizeExecutionReport(
		envelope.Report, MaxReportJSONBytes,
	)
	if err != nil {
		return fmt.Errorf("%w: %v", ErrInvalid, err)
	}
	encoded, err := json.Marshal(report)
	if err != nil {
		return fmt.Errorf("encode Planner report: %w", err)
	}
	receivedAt := envelope.ReceivedAt
	if receivedAt.IsZero() {
		receivedAt = time.Now().UTC()
	}
	if !report.Complete {
		existing, existingErr := r.plannerReportForStage(ctx, envelope.StageExecutionID)
		if existingErr == nil && existing.SessionID == envelope.SessionID &&
			existing.InvocationID == envelope.InvocationID &&
			existing.Report.ReportID != report.ReportID && existing.Report.Complete {
			return nil
		}
		if existingErr != nil && !errors.Is(existingErr, ErrNotFound) {
			return existingErr
		}
	}
	command, err := r.db.Exec(ctx, `
INSERT INTO planner_execution_reports (
    report_id, stage_execution_id, session_id, invocation_id, started_at, finished_at,
    report_schema_version, report, received_at, expires_at
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9::timestamptz, $9::timestamptz + interval '30 days')
ON CONFLICT DO NOTHING`,
		report.ReportID, envelope.StageExecutionID, envelope.SessionID, envelope.InvocationID,
		envelope.StartedAt, envelope.FinishedAt, envelope.ReportSchemaVersion, encoded, receivedAt,
	)
	if err != nil {
		return classifyWriteError("record Planner report", err)
	}
	if command.RowsAffected() == 1 {
		return nil
	}
	existing, err := r.plannerReportForIdentity(ctx, report.ReportID, envelope.StageExecutionID)
	if err != nil {
		return err
	}
	if existing.StageExecutionID != envelope.StageExecutionID ||
		existing.SessionID != envelope.SessionID ||
		existing.InvocationID != envelope.InvocationID ||
		existing.ReportSchemaVersion != envelope.ReportSchemaVersion {
		return fmt.Errorf("record Planner report: %w", ErrConflict)
	}
	// An incomplete report is an ensure-placeholder operation. A complete
	// report that already won the identity race is strictly better and remains
	// immutable; recovery may safely treat that as success.
	if !report.Complete && existing.Report.Complete {
		return nil
	}
	if !existing.StartedAt.Equal(envelope.StartedAt) ||
		!existing.FinishedAt.Equal(envelope.FinishedAt) ||
		!sameJSON(existing.Report, report) {
		return fmt.Errorf("record Planner report: %w", ErrConflict)
	}
	return nil
}

func (r *Repository) RebuildStageMetrics(
	ctx context.Context,
	stageExecutionID string,
	schemaVersion string,
) (StageMetricsRecord, error) {
	if err := requireText("stageExecutionID", stageExecutionID); err != nil {
		return StageMetricsRecord{}, err
	}
	if err := requireText("metricsSchemaVersion", schemaVersion); err != nil {
		return StageMetricsRecord{}, err
	}
	var planner *contracts.ExecutionReport
	plannerRow, err := r.plannerReportForStage(ctx, stageExecutionID)
	if err == nil {
		value := plannerRow.Report
		planner = &value
	} else if !errors.Is(err, ErrNotFound) {
		return StageMetricsRecord{}, err
	}
	allocationRows, err := r.ListAllocationReports(ctx, stageExecutionID)
	if err != nil {
		return StageMetricsRecord{}, err
	}
	allocations := make(map[string]contracts.AllocationFinalReport, len(allocationRows))
	for _, item := range allocationRows {
		allocations[item.LogicalAgentName] = item.Report
	}
	metrics := BuildStageMetrics(planner, allocations)
	if err := metrics.Validate(); err != nil {
		return StageMetricsRecord{}, fmt.Errorf("%w: %v", ErrInvalid, err)
	}
	summary := Summarize(metrics)
	encodedMetrics, err := json.Marshal(metrics)
	if err != nil {
		return StageMetricsRecord{}, fmt.Errorf("encode StageMetrics: %w", err)
	}
	encodedSummary, err := json.Marshal(summary)
	if err != nil {
		return StageMetricsRecord{}, fmt.Errorf("encode telemetry summary: %w", err)
	}
	row := r.db.QueryRow(ctx, `
INSERT INTO stage_metrics (
    stage_execution_id, metrics_schema_version, metrics, summary
) VALUES ($1, $2, $3::jsonb, $4::jsonb)
ON CONFLICT (stage_execution_id) DO UPDATE SET
    metrics_schema_version = EXCLUDED.metrics_schema_version,
    metrics = EXCLUDED.metrics,
    summary = EXCLUDED.summary,
    updated_at = clock_timestamp(),
    expires_at = GREATEST(stage_metrics.expires_at, EXCLUDED.expires_at)
RETURNING metrics_schema_version, metrics, summary, updated_at, expires_at`,
		stageExecutionID, schemaVersion, encodedMetrics, encodedSummary,
	)
	result := StageMetricsRecord{StageExecutionID: stageExecutionID}
	var storedMetrics, storedSummary []byte
	if err := row.Scan(
		&result.MetricsSchemaVersion, &storedMetrics, &storedSummary,
		&result.UpdatedAt, &result.ExpiresAt,
	); err != nil {
		return StageMetricsRecord{}, fmt.Errorf("persist StageMetrics: %w", err)
	}
	if err := json.Unmarshal(storedMetrics, &result.Metrics); err != nil {
		return StageMetricsRecord{}, fmt.Errorf("decode StageMetrics: %w", err)
	}
	if err := json.Unmarshal(storedSummary, &result.Summary); err != nil {
		return StageMetricsRecord{}, fmt.Errorf("decode telemetry summary: %w", err)
	}
	return result, nil
}

func (r *Repository) GetStageMetrics(
	ctx context.Context,
	stageExecutionID string,
) (StageMetricsRecord, error) {
	if err := requireText("stageExecutionID", stageExecutionID); err != nil {
		return StageMetricsRecord{}, err
	}
	result := StageMetricsRecord{StageExecutionID: stageExecutionID}
	var encodedMetrics, encodedSummary []byte
	err := r.db.QueryRow(ctx, `
SELECT metrics_schema_version, metrics, summary, updated_at, expires_at
FROM stage_metrics WHERE stage_execution_id = $1`, stageExecutionID).Scan(
		&result.MetricsSchemaVersion, &encodedMetrics, &encodedSummary,
		&result.UpdatedAt, &result.ExpiresAt,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return StageMetricsRecord{}, fmt.Errorf("get StageMetrics: %w", ErrNotFound)
	}
	if err != nil {
		return StageMetricsRecord{}, fmt.Errorf("get StageMetrics: %w", err)
	}
	if err := json.Unmarshal(encodedMetrics, &result.Metrics); err != nil {
		return StageMetricsRecord{}, fmt.Errorf("decode StageMetrics: %w", err)
	}
	if err := json.Unmarshal(encodedSummary, &result.Summary); err != nil {
		return StageMetricsRecord{}, fmt.Errorf("decode telemetry summary: %w", err)
	}
	return result, nil
}

// CleanupExpired removes only telemetry for semantically terminal executions.
// The explicit batch cap applies across all telemetry tables.
func (r *Repository) CleanupExpired(
	ctx context.Context,
	now time.Time,
	batchSize int,
) (int64, error) {
	if now.IsZero() {
		return 0, fmt.Errorf("%w: cleanup time is required", ErrInvalid)
	}
	if batchSize <= 0 || batchSize > 10_000 {
		return 0, fmt.Errorf("%w: cleanup batch must be between 1 and 10000", ErrInvalid)
	}
	queries := []string{
		`WITH doomed AS (
            SELECT m.stage_execution_id FROM stage_metrics m
            JOIN stage_executions e USING (stage_execution_id)
            WHERE m.expires_at <= $1
              AND e.state IN ('succeeded', 'failed', 'interrupted', 'cancelled')
            ORDER BY m.expires_at, m.stage_execution_id LIMIT $2
        ) DELETE FROM stage_metrics m USING doomed d
          WHERE m.stage_execution_id = d.stage_execution_id`,
		`WITH doomed AS (
            SELECT r.report_id FROM allocation_execution_reports r
            JOIN stage_executions e USING (stage_execution_id)
            WHERE r.expires_at <= $1
              AND e.state IN ('succeeded', 'failed', 'interrupted', 'cancelled')
            ORDER BY r.expires_at, r.report_id LIMIT $2
        ) DELETE FROM allocation_execution_reports r USING doomed d
          WHERE r.report_id = d.report_id`,
		`WITH doomed AS (
            SELECT r.report_id FROM planner_execution_reports r
            JOIN stage_executions e USING (stage_execution_id)
            WHERE r.expires_at <= $1
              AND e.state IN ('succeeded', 'failed', 'interrupted', 'cancelled')
            ORDER BY r.expires_at, r.report_id LIMIT $2
        ) DELETE FROM planner_execution_reports r USING doomed d
          WHERE r.report_id = d.report_id`,
	}
	var deleted int64
	for _, query := range queries {
		remaining := int64(batchSize) - deleted
		if remaining <= 0 {
			break
		}
		command, err := r.db.Exec(ctx, query, now.UTC(), remaining)
		if err != nil {
			return deleted, fmt.Errorf("cleanup expired telemetry: %w", err)
		}
		deleted += command.RowsAffected()
	}
	return deleted, nil
}

func (r *Repository) allocationReportForIdentity(
	ctx context.Context,
	reportID string,
	stageExecutionID string,
	allocationID string,
	logicalAgentName string,
) (StoredAllocationReport, error) {
	result, err := scanAllocationReport(r.db.QueryRow(ctx, `
SELECT stage_execution_id, allocation_id, logical_agent_name,
       report_schema_version, report, received_at, expires_at
FROM allocation_execution_reports
WHERE report_id = $1 OR allocation_id = $3
   OR (stage_execution_id = $2 AND logical_agent_name = $4)
ORDER BY (report_id = $1) DESC, (allocation_id = $3) DESC
LIMIT 1`, reportID, stageExecutionID, allocationID, logicalAgentName))
	if errors.Is(err, pgx.ErrNoRows) {
		return StoredAllocationReport{}, fmt.Errorf("get allocation report: %w", ErrNotFound)
	}
	if err != nil {
		return StoredAllocationReport{}, fmt.Errorf("get allocation report: %w", err)
	}
	return result, nil
}

func (r *Repository) allocationReportForParticipant(
	ctx context.Context,
	stageExecutionID string,
	logicalAgentName string,
) (StoredAllocationReport, error) {
	result, err := scanAllocationReport(r.db.QueryRow(ctx, `
SELECT stage_execution_id, allocation_id, logical_agent_name,
       report_schema_version, report, received_at, expires_at
FROM allocation_execution_reports
WHERE stage_execution_id = $1 AND logical_agent_name = $2
ORDER BY ((report->'worker'->>'complete')::boolean
          AND (report->'runtime'->>'complete')::boolean) DESC,
         received_at DESC, report_id
LIMIT 1`, stageExecutionID, logicalAgentName))
	if errors.Is(err, pgx.ErrNoRows) {
		return StoredAllocationReport{}, fmt.Errorf("get allocation participant report: %w", ErrNotFound)
	}
	if err != nil {
		return StoredAllocationReport{}, fmt.Errorf("get allocation participant report: %w", err)
	}
	return result, nil
}

func (r *Repository) plannerReportForIdentity(
	ctx context.Context,
	reportID string,
	stageExecutionID string,
) (StoredPlannerReport, error) {
	return r.scanPlannerReport(r.db.QueryRow(ctx, `
SELECT stage_execution_id, session_id, invocation_id,
	   started_at, finished_at, report_schema_version, report, received_at, expires_at
FROM planner_execution_reports
WHERE report_id = $1 OR stage_execution_id = $2
ORDER BY (report_id = $1) DESC LIMIT 1`, reportID, stageExecutionID))
}

func (r *Repository) plannerReportForStage(
	ctx context.Context,
	stageExecutionID string,
) (StoredPlannerReport, error) {
	return r.scanPlannerReport(r.db.QueryRow(ctx, `
SELECT stage_execution_id, session_id, invocation_id,
	   started_at, finished_at, report_schema_version, report, received_at, expires_at
FROM planner_execution_reports
WHERE stage_execution_id = $1
ORDER BY (report->>'complete')::boolean DESC, received_at DESC, report_id
LIMIT 1`, stageExecutionID))
}

func (r *Repository) scanPlannerReport(row rowScanner) (StoredPlannerReport, error) {
	var result StoredPlannerReport
	var encoded []byte
	if err := row.Scan(
		&result.StageExecutionID, &result.SessionID, &result.InvocationID,
		&result.StartedAt, &result.FinishedAt, &result.ReportSchemaVersion,
		&encoded, &result.ReceivedAt, &result.ExpiresAt,
	); errors.Is(err, pgx.ErrNoRows) {
		return StoredPlannerReport{}, fmt.Errorf("get Planner report: %w", ErrNotFound)
	} else if err != nil {
		return StoredPlannerReport{}, fmt.Errorf("get Planner report: %w", err)
	}
	if err := json.Unmarshal(encoded, &result.Report); err != nil {
		return StoredPlannerReport{}, fmt.Errorf("decode Planner report: %w", err)
	}
	return result, nil
}

type rowScanner interface {
	Scan(...any) error
}

func scanAllocationReport(row rowScanner) (StoredAllocationReport, error) {
	var result StoredAllocationReport
	var encoded []byte
	if err := row.Scan(
		&result.StageExecutionID, &result.AllocationID, &result.LogicalAgentName,
		&result.ReportSchemaVersion, &encoded, &result.ReceivedAt, &result.ExpiresAt,
	); err != nil {
		return StoredAllocationReport{}, err
	}
	if err := json.Unmarshal(encoded, &result.Report); err != nil {
		return StoredAllocationReport{}, fmt.Errorf("decode allocation report: %w", err)
	}
	return result, nil
}

func validateAllocationEnvelope(value AllocationReportEnvelope) error {
	for field, current := range map[string]string{
		"stageExecutionID":    value.StageExecutionID,
		"allocationID":        value.AllocationID,
		"logicalAgentName":    value.LogicalAgentName,
		"reportSchemaVersion": value.ReportSchemaVersion,
	} {
		if err := requireText(field, current); err != nil {
			return err
		}
	}
	switch value.PerformanceCollectionPolicy {
	case "", contracts.PerformanceCollectionLegacy:
		// Empty/legacy is accepted only for reports belonging to allocations
		// created before collection policy became durable provenance.
	default:
		if err := value.PerformanceCollectionPolicy.ValidatePinned(); err != nil {
			return fmt.Errorf("%w: performance collection policy is invalid", ErrInvalid)
		}
	}
	return nil
}

func validatePlannerEnvelope(value PlannerReportEnvelope) error {
	for field, current := range map[string]string{
		"stageExecutionID":    value.StageExecutionID,
		"sessionID":           value.SessionID,
		"invocationID":        value.InvocationID,
		"reportSchemaVersion": value.ReportSchemaVersion,
	} {
		if err := requireText(field, current); err != nil {
			return err
		}
	}
	if value.StartedAt.IsZero() || value.FinishedAt.IsZero() ||
		value.FinishedAt.Before(value.StartedAt) {
		return fmt.Errorf("%w: Planner report timestamps are invalid", ErrInvalid)
	}
	return nil
}

func requireText(field string, value string) error {
	if strings.TrimSpace(value) == "" {
		return fmt.Errorf("%w: %s is required", ErrInvalid, field)
	}
	return nil
}

func classifyWriteError(operation string, err error) error {
	switch persistencepostgres.SQLState(err) {
	case "23503":
		return fmt.Errorf("%s: %w", operation, ErrNotFound)
	case "23505", "23514":
		return fmt.Errorf("%s: %w", operation, ErrConflict)
	default:
		return fmt.Errorf("%s: %w", operation, err)
	}
}

func sameJSON(left, right any) bool {
	leftJSON, leftErr := json.Marshal(left)
	rightJSON, rightErr := json.Marshal(right)
	return leftErr == nil && rightErr == nil && bytes.Equal(leftJSON, rightJSON)
}
