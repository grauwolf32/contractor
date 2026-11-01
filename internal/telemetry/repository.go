package telemetry

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
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
	Report                      reporting.AllocationFinalReport
	PerformanceCollectionPolicy reporting.PerformanceCollectionPolicy
	Secrets                     []string `json:"-"`
	ReceivedAt                  time.Time
}

// Format redacts the envelope as a whole for every verb (SEC-R88): Secrets holds secret values in plain text; StoredAllocationReport embeds it.
func (AllocationReportEnvelope) Format(state fmt.State, _ rune) {
	_, _ = io.WriteString(state, "telemetry.AllocationReportEnvelope([REDACTED])")
}

type PlannerReportEnvelope struct {
	StageExecutionID    string
	SessionID           string
	InvocationID        string
	StartedAt           time.Time
	FinishedAt          time.Time
	ReportSchemaVersion string
	Report              reporting.ExecutionReport
	Secrets             []string `json:"-"`
	ReceivedAt          time.Time
}

// Format redacts the envelope as a whole for every verb (SEC-R88): Secrets holds secret values in plain text; StoredPlannerReport embeds it.
func (PlannerReportEnvelope) Format(state fmt.State, _ rune) {
	_, _ = io.WriteString(state, "telemetry.PlannerReportEnvelope([REDACTED])")
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
	Metrics              reporting.StageMetrics
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
	envelope.Report.Runtime.ApplyResourcePolicy(envelope.PerformanceCollectionPolicy)
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
	existing, sameReport, err := r.allocationReportForIdentity(
		ctx, report.ReportID, envelope.StageExecutionID, envelope.AllocationID,
		envelope.LogicalAgentName, encoded,
	)
	if err != nil {
		return err
	}
	if existing.StageExecutionID != envelope.StageExecutionID ||
		existing.AllocationID != envelope.AllocationID ||
		existing.LogicalAgentName != envelope.LogicalAgentName ||
		existing.ReportSchemaVersion != envelope.ReportSchemaVersion ||
		!sameReport {
		return fmt.Errorf("record allocation report: %w", ErrConflict)
	}
	return nil
}

func (r *Repository) ListAllocationReports(
	ctx context.Context,
	stageExecutionID string,
) ([]StoredAllocationReport, error) {
	if err := requireText("stageExecutionID", stageExecutionID); err != nil {
		return nil, err
	}
	rows, err := r.db.Query(ctx, listAllocationReportsSQL, stageExecutionID)
	if err != nil {
		return nil, fmt.Errorf("list allocation reports: %w", err)
	}
	result, err := pgx.CollectRows(rows, func(row pgx.CollectableRow) (StoredAllocationReport, error) {
		return scanAllocationReport(row)
	})
	if err != nil {
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
	envelope.StartedAt = persistencepostgres.Timestamp(envelope.StartedAt)
	envelope.FinishedAt = persistencepostgres.Timestamp(envelope.FinishedAt)
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
	existing, sameReport, err := r.plannerReportForIdentity(
		ctx, report.ReportID, envelope.StageExecutionID, encoded,
	)
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
		!sameReport {
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
	var planner *reporting.ExecutionReport
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
	allocations := make(map[string]reporting.AllocationFinalReport, len(allocationRows))
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
	row := r.db.QueryRow(ctx, rebuildStageMetricsSQL,
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
	if result.Metrics, err = decodeStoredStageMetrics(storedMetrics); err != nil {
		return StageMetricsRecord{}, fmt.Errorf("decode StageMetrics: %w", err)
	}
	if result.Summary, err = decodeStoredSummary(storedSummary); err != nil {
		return StageMetricsRecord{}, fmt.Errorf("decode telemetry summary: %w", err)
	}
	return result, nil
}

// CleanupExpired removes only telemetry for semantically terminal executions,
// at most batchSize rows of each telemetry table, and returns the number of
// expired rows it settled: deleted, or renewed StageMetrics that an Eval
// member reads (cleanupStageMetricsSQL).
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
	// Each table gets its own bounded budget. A large metrics backlog must
	// not starve expired allocation and Planner reports.
	var settled int64
	if err := r.db.QueryRow(ctx, cleanupStageMetricsSQL, now.UTC(), batchSize).Scan(&settled); err != nil {
		return 0, fmt.Errorf("cleanup expired telemetry: %w", err)
	}
	queries := []string{
		`WITH doomed AS (
            SELECT r.report_id FROM allocation_execution_reports r
            JOIN stage_executions e USING (stage_execution_id)
            WHERE r.expires_at <= $1
              AND e.state IN ('succeeded', 'failed', 'interrupted', 'cancelled')
            ORDER BY r.expires_at, r.report_id LIMIT $2
            FOR UPDATE OF r SKIP LOCKED
        ) DELETE FROM allocation_execution_reports r USING doomed d
          WHERE r.report_id = d.report_id`,
		`WITH doomed AS (
            SELECT r.report_id FROM planner_execution_reports r
            JOIN stage_executions e USING (stage_execution_id)
            WHERE r.expires_at <= $1
              AND e.state IN ('succeeded', 'failed', 'interrupted', 'cancelled')
            ORDER BY r.expires_at, r.report_id LIMIT $2
            FOR UPDATE OF r SKIP LOCKED
        ) DELETE FROM planner_execution_reports r USING doomed d
          WHERE r.report_id = d.report_id`,
	}
	for _, query := range queries {
		command, err := r.db.Exec(ctx, query, now.UTC(), batchSize)
		if err != nil {
			return settled, fmt.Errorf("cleanup expired telemetry: %w", err)
		}
		settled += command.RowsAffected()
	}
	return settled, nil
}

// allocationReportForIdentity returns the stored report that holds one of the
// identities of a replayed report, and whether the stored document equals
// encoded. PostgreSQL compares the two as jsonb, so a number the column
// normalized (5e-05 stored as 0.00005) still equals the literal the Runtime
// wrote, and a replay stays idempotent (OBS-R42).
func (r *Repository) allocationReportForIdentity(
	ctx context.Context,
	reportID string,
	stageExecutionID string,
	allocationID string,
	logicalAgentName string,
	encoded []byte,
) (StoredAllocationReport, bool, error) {
	var sameReport bool
	result, err := scanAllocationReport(sameReportRow{r.db.QueryRow(ctx, `
SELECT stage_execution_id, allocation_id, logical_agent_name,
       report_schema_version, report, received_at, expires_at, report = $5::jsonb
FROM allocation_execution_reports
WHERE report_id = $1 OR allocation_id = $3
   OR (stage_execution_id = $2 AND logical_agent_name = $4)
ORDER BY (report_id = $1) DESC, (allocation_id = $3) DESC
LIMIT 1`, reportID, stageExecutionID, allocationID, logicalAgentName, encoded), &sameReport})
	if errors.Is(err, pgx.ErrNoRows) {
		return StoredAllocationReport{}, false, fmt.Errorf("get allocation report: %w", ErrNotFound)
	}
	if err != nil {
		return StoredAllocationReport{}, false, fmt.Errorf("get allocation report: %w", err)
	}
	return result, sameReport, nil
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

// plannerReportForIdentity returns the stored Planner report that holds one of
// the identities of a replayed report, and whether the stored document equals
// encoded as jsonb, like allocationReportForIdentity.
func (r *Repository) plannerReportForIdentity(
	ctx context.Context,
	reportID string,
	stageExecutionID string,
	encoded []byte,
) (StoredPlannerReport, bool, error) {
	var sameReport bool
	result, err := r.scanPlannerReport(sameReportRow{r.db.QueryRow(ctx, `
SELECT stage_execution_id, session_id, invocation_id,
	   started_at, finished_at, report_schema_version, report, received_at, expires_at,
	   report = $3::jsonb
FROM planner_execution_reports
WHERE report_id = $1 OR stage_execution_id = $2
ORDER BY (report_id = $1) DESC LIMIT 1`, reportID, stageExecutionID, encoded), &sameReport})
	return result, sameReport, err
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
	// A stored report is decoded as strictly as the allocation report beside
	// it (AllocationFinalReport.UnmarshalJSON); RebuildStageMetrics applies
	// the report rules to both (OBS-R110).
	if err := strictjson.DecodeNonNull(encoded, &result.Report); err != nil {
		return StoredPlannerReport{}, fmt.Errorf("decode Planner report: %w", err)
	}
	return result, nil
}

type rowScanner interface {
	Scan(...any) error
}

// sameReportRow scans a replay identity row whose last column is the jsonb
// comparison of the stored and the replayed report into same.
type sameReportRow struct {
	row  rowScanner
	same *bool
}

func (r sameReportRow) Scan(dest ...any) error {
	return r.row.Scan(append(dest, r.same)...)
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
	if err := value.PerformanceCollectionPolicy.ValidatePinned(); err != nil {
		return fmt.Errorf("%w: performance collection policy is invalid", ErrInvalid)
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

// classifyWriteError keeps a deterministic database rejection permanent: a
// constraint or trigger violation is invalid, never a retryable conflict.
func classifyWriteError(operation string, err error) error {
	switch persistencepostgres.Classify(err) {
	case persistencepostgres.ClassReferenceViolation:
		return fmt.Errorf("%s: %w", operation, ErrNotFound)
	case persistencepostgres.ClassDuplicate:
		return fmt.Errorf("%s: %w", operation, ErrConflict)
	case persistencepostgres.ClassInvalid, persistencepostgres.ClassRejected:
		return fmt.Errorf("%s: %w", operation, ErrInvalid)
	default:
		return fmt.Errorf("%s: %w", operation, err)
	}
}
