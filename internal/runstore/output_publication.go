package runstore

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

const outputPublicationColumns = `
run_id, project_id, output_name, status,
source_namespace, source_name, source_revision,
target_namespace, target_name, target_revision,
error_code, error_message, created_at`

func (s *PostgresStore) RecordRunOutputPublication(
	ctx context.Context,
	params RecordRunOutputPublicationParams,
) (RunOutputPublication, bool, error) {
	if err := validateOutputPublication(params); err != nil {
		return RunOutputPublication{}, false, err
	}
	var targetRevision *string
	if params.Target != nil {
		targetRevision = params.Target.Revision
	}
	row := s.db.QueryRow(ctx, `
INSERT INTO workflow_run_output_publications (
    run_id, project_id, output_name, status,
    source_namespace, source_name, source_revision,
    target_namespace, target_name, target_revision,
    error_code, error_message
) VALUES (
    $1, $2, $3, $4,
    $5, $6, $7,
    'outputs', $3, $8,
    NULLIF($9, ''), NULLIF($10, '')
)
ON CONFLICT (run_id, output_name) DO NOTHING
RETURNING `+outputPublicationColumns+`, true`,
		params.RunID, params.ProjectID, params.OutputName, params.Status,
		params.Source.Namespace, params.Source.Name, *params.Source.Revision,
		targetRevision, params.ErrorCode, params.ErrorMessage,
	)
	record, created, err := scanOutputPublication(row)
	if errors.Is(err, pgx.ErrNoRows) {
		record, created, err = scanOutputPublication(s.db.QueryRow(ctx, `
SELECT `+outputPublicationColumns+`, false
FROM workflow_run_output_publications
WHERE run_id = $1 AND output_name = $2`, params.RunID, params.OutputName))
	}
	if err != nil {
		if persistencepostgres.SQLState(err) == "23503" {
			return RunOutputPublication{}, false, fmt.Errorf("record Project output publication: %w", ErrNotFound)
		}
		return RunOutputPublication{}, false, fmt.Errorf("record Project output publication: %w", err)
	}
	if !created && !sameOutputPublication(record, params) {
		return RunOutputPublication{}, false, fmt.Errorf("record Project output publication: %w", ErrConflict)
	}
	return record, created, nil
}

func (s *PostgresStore) ListRunOutputPublications(
	ctx context.Context,
	runID string,
) ([]RunOutputPublication, error) {
	if err := validateOpaque("runID", runID); err != nil {
		return nil, err
	}
	rows, err := s.db.Query(ctx, `
SELECT `+outputPublicationColumns+`, false
FROM workflow_run_output_publications
WHERE run_id = $1
ORDER BY output_name`, runID)
	if err != nil {
		return nil, fmt.Errorf("list Project output publications: %w", err)
	}
	defer rows.Close()
	result := make([]RunOutputPublication, 0)
	for rows.Next() {
		record, _, scanErr := scanOutputPublication(rows)
		if scanErr != nil {
			return nil, fmt.Errorf("scan Project output publication: %w", scanErr)
		}
		result = append(result, record)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate Project output publications: %w", err)
	}
	return result, nil
}

type publicationRow interface {
	Scan(...any) error
}

func scanOutputPublication(row publicationRow) (RunOutputPublication, bool, error) {
	var result RunOutputPublication
	var status string
	var sourceRevision string
	var targetNamespace string
	var targetName string
	var targetRevision *string
	var errorCode *string
	var errorMessage *string
	var created bool
	err := row.Scan(
		&result.RunID, &result.ProjectID, &result.OutputName, &status,
		&result.Source.Namespace, &result.Source.Name, &sourceRevision,
		&targetNamespace, &targetName, &targetRevision,
		&errorCode, &errorMessage, &result.CreatedAt, &created,
	)
	if err != nil {
		return RunOutputPublication{}, false, err
	}
	result.Status = OutputPublicationStatus(status)
	result.Source.Revision = &sourceRevision
	if targetRevision != nil {
		result.Target = &contracts.ArtifactRef{
			Namespace: targetNamespace,
			Name:      targetName,
			Revision:  targetRevision,
		}
	}
	if errorCode != nil {
		result.ErrorCode = *errorCode
	}
	if errorMessage != nil {
		result.ErrorMessage = *errorMessage
	}
	return result, created, nil
}

func validateOutputPublication(params RecordRunOutputPublicationParams) error {
	if err := validateOpaque("runID", params.RunID); err != nil {
		return err
	}
	if err := validateOpaque("projectID", params.ProjectID); err != nil {
		return err
	}
	if strings.TrimSpace(params.OutputName) == "" || strings.Contains(params.OutputName, "/") ||
		len(params.OutputName) > 128 {
		return invalidf("output publication name is invalid")
	}
	if err := params.Source.ValidateExact(); err != nil ||
		params.Source.Namespace != "outputs" || params.Source.Name != params.OutputName {
		return invalidf("output publication source is invalid")
	}
	switch params.Status {
	case OutputPublicationPublished:
		if params.Target == nil || params.Target.ValidateExact() != nil ||
			params.Target.Namespace != "outputs" || params.Target.Name != params.OutputName ||
			params.ErrorCode != "" || params.ErrorMessage != "" {
			return invalidf("published output publication is invalid")
		}
	case OutputPublicationAlreadyPresent:
		if params.Target != nil || params.ErrorCode != "" || params.ErrorMessage != "" {
			return invalidf("already-present output publication is invalid")
		}
	case OutputPublicationFailed:
		if params.Target != nil || strings.TrimSpace(params.ErrorCode) == "" ||
			len(params.ErrorCode) > 128 || strings.TrimSpace(params.ErrorMessage) == "" ||
			len(params.ErrorMessage) > 2048 {
			return invalidf("failed output publication is invalid")
		}
	default:
		return invalidf("unknown output publication status %q", params.Status)
	}
	return nil
}

func sameOutputPublication(
	record RunOutputPublication,
	params RecordRunOutputPublicationParams,
) bool {
	if record.RunID != params.RunID || record.ProjectID != params.ProjectID ||
		record.OutputName != params.OutputName || record.Status != params.Status ||
		!sameExactArtifactRef(record.Source, params.Source) ||
		record.ErrorCode != params.ErrorCode || record.ErrorMessage != params.ErrorMessage {
		return false
	}
	if record.Target == nil || params.Target == nil {
		return record.Target == nil && params.Target == nil
	}
	return sameExactArtifactRef(*record.Target, *params.Target)
}

func sameExactArtifactRef(left, right contracts.ArtifactRef) bool {
	return left.Namespace == right.Namespace && left.Name == right.Name &&
		left.Revision != nil && right.Revision != nil && *left.Revision == *right.Revision
}
