package runstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"sort"

	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

func (s *PostgresStore) RecordStageExecutionReport(
	ctx context.Context,
	params RecordStageExecutionReportParams,
) error {
	if err := validateOpaque("stageExecutionID", params.StageExecutionID); err != nil {
		return err
	}
	if err := validateOpaque("allocationID", params.AllocationID); err != nil {
		return err
	}
	if err := validateOpaque("logicalAgentName", params.LogicalAgentName); err != nil {
		return err
	}
	if err := validateOpaque("reportSchemaVersion", params.ReportSchemaVersion); err != nil {
		return err
	}
	report := normalizeExecutionReport(params.Report)
	if report.AllocationID != params.AllocationID {
		return fmt.Errorf("%w: execution report identifies another allocation", ErrInvalid)
	}
	if err := (contracts.AllocationFinalResponse{
		APIVersion: params.ReportSchemaVersion,
		Report:     report,
	}).Validate(); err != nil {
		return fmt.Errorf("%w: invalid execution report: %v", ErrInvalid, err)
	}
	encoded, err := json.Marshal(report)
	if err != nil {
		return fmt.Errorf("encode execution report: %w", err)
	}
	command, err := s.db.Exec(ctx, `
INSERT INTO stage_execution_reports (
    stage_execution_id, allocation_id, logical_agent_name,
    report_schema_version, report
) VALUES ($1, $2, $3, $4, $5::jsonb)
ON CONFLICT DO NOTHING`,
		params.StageExecutionID, params.AllocationID, params.LogicalAgentName,
		params.ReportSchemaVersion, encoded,
	)
	if err != nil {
		switch persistencepostgres.SQLState(err) {
		case "23503":
			return fmt.Errorf("record StageExecution report: %w", ErrNotFound)
		case "23505":
			return fmt.Errorf("record StageExecution report: %w", ErrConflict)
		default:
			return fmt.Errorf("record StageExecution report: %w", err)
		}
	}
	if command.RowsAffected() == 1 {
		return nil
	}
	existing, err := s.executionReportForIdentity(
		ctx, params.StageExecutionID, params.AllocationID, params.LogicalAgentName,
	)
	if err != nil {
		return err
	}
	if existing.StageExecutionID != params.StageExecutionID ||
		existing.LogicalAgentName != params.LogicalAgentName ||
		existing.ReportSchemaVersion != params.ReportSchemaVersion ||
		!reflect.DeepEqual(existing.Report, report) {
		return fmt.Errorf("record StageExecution report: %w", ErrConflict)
	}
	return nil
}

func (s *PostgresStore) ListStageExecutionReports(
	ctx context.Context,
	stageExecutionID string,
) ([]StageExecutionReport, error) {
	if err := validateOpaque("stageExecutionID", stageExecutionID); err != nil {
		return nil, err
	}
	rows, err := s.db.Query(ctx, `
SELECT stage_execution_id, allocation_id, logical_agent_name,
       report_schema_version, report, received_at
FROM stage_execution_reports
WHERE stage_execution_id = $1
ORDER BY logical_agent_name`, stageExecutionID)
	if err != nil {
		return nil, fmt.Errorf("list StageExecution reports: %w", err)
	}
	defer rows.Close()
	result := make([]StageExecutionReport, 0)
	for rows.Next() {
		current, err := scanStageExecutionReport(rows)
		if err != nil {
			return nil, fmt.Errorf("list StageExecution reports: %w", err)
		}
		result = append(result, current)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("list StageExecution reports: %w", err)
	}
	return result, nil
}

func (s *PostgresStore) executionReportForIdentity(
	ctx context.Context,
	stageExecutionID, allocationID, logicalAgentName string,
) (StageExecutionReport, error) {
	result, err := scanStageExecutionReport(s.db.QueryRow(ctx, `
SELECT stage_execution_id, allocation_id, logical_agent_name,
       report_schema_version, report, received_at
FROM stage_execution_reports
WHERE allocation_id = $1
   OR (stage_execution_id = $2 AND logical_agent_name = $3)
ORDER BY (allocation_id = $1) DESC
LIMIT 1`, allocationID, stageExecutionID, logicalAgentName))
	if errors.Is(err, pgx.ErrNoRows) {
		return StageExecutionReport{}, fmt.Errorf("get StageExecution report: %w", ErrNotFound)
	}
	if err != nil {
		return StageExecutionReport{}, fmt.Errorf("get StageExecution report: %w", err)
	}
	return result, nil
}

type reportScanner interface {
	Scan(...any) error
}

func scanStageExecutionReport(row reportScanner) (StageExecutionReport, error) {
	var result StageExecutionReport
	var encoded []byte
	if err := row.Scan(
		&result.StageExecutionID, &result.AllocationID, &result.LogicalAgentName,
		&result.ReportSchemaVersion, &encoded, &result.ReceivedAt,
	); err != nil {
		return StageExecutionReport{}, err
	}
	if err := json.Unmarshal(encoded, &result.Report); err != nil {
		return StageExecutionReport{}, fmt.Errorf("decode execution report: %w", err)
	}
	result.Report = normalizeExecutionReport(result.Report)
	return result, nil
}

func normalizeExecutionReport(source contracts.ExecutionReport) contracts.ExecutionReport {
	result := source
	result.StartedAt = result.StartedAt.UTC().Round(0)
	result.FinishedAt = result.FinishedAt.UTC().Round(0)
	result.Counters = make(map[string]int64, len(source.Counters))
	keys := make([]string, 0, len(source.Counters))
	for key := range source.Counters {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		result.Counters[key] = source.Counters[key]
	}
	result.Errors = append([]contracts.TerminationError{}, source.Errors...)
	return result
}
