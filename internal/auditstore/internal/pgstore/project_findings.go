package pgstore

import (
	"context"
	"encoding/json"
	"errors"
	"strconv"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	artifactpolicy "github.com/grauwolf32/contractor/internal/artifacts/policy"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"github.com/jackc/pgx/v5"
)

func scanProjectFinding(row pgx.Row) (auditstore.ProjectFinding, error) {
	var value auditstore.ProjectFinding
	var proposal []byte
	err := row.Scan(&value.FindingID, &value.ProjectID, &value.FirstReceiptID, &proposal, &value.Title,
		&value.State, &value.AnalystVerdict, &value.AnalystSeverity, &value.Rationale, &value.ActorID,
		&value.AssessedAt, &value.DuplicateTargetID, &value.Revision, &value.CreatedAt, &value.UpdatedAt, &value.CorrelationAlgorithm, &value.CorrelationValue, &value.CorrelationGroupKey, &value.CorrelationReason)
	if errors.Is(err, pgx.ErrNoRows) {
		return value, auditstore.ErrNotFound
	}
	if err != nil {
		return value, err
	}
	if err := strictjson.Decode(proposal, &value.FirstProposal); err != nil || value.FirstProposal.Validate() != nil || !value.FirstProposal.HasMetadata() {
		return value, blob.ErrIntegrity
	}
	if value.Validate() != nil {
		return value, errors.New("stored Project Finding is invalid")
	}

	return value, nil
}

func (s *QueryStore) GetProjectFinding(ctx context.Context, ownerID, projectID, findingID string) (auditstore.ProjectFinding, error) {
	return scanProjectFinding(s.db.QueryRow(ctx, getProjectFindingSQL, ownerID, projectID, findingID))
}

// GetAuditProjectFinding is a genuine filtered view of the Project record.
func (s *QueryStore) GetAuditProjectFinding(ctx context.Context, ownerID, auditID, findingID string) (auditstore.ProjectFinding, error) {
	return scanProjectFinding(s.db.QueryRow(ctx, getAuditProjectFindingSQL, ownerID, auditID, findingID))
}

// AssessProjectFinding changes only the canonical Project record. The caller
// holds the common keyed-mutation receipt lock and records its returned snapshot.
func (s *OwnerTxStore) AssessProjectFinding(ctx context.Context, edit auditstore.ProjectFindingEdit) (auditstore.ProjectFinding, error) {
	var active bool
	if err := s.tx.QueryRow(ctx, `SELECT true FROM projects WHERE owner_id=$1 AND project_id=$2 AND kind='project' AND lifecycle_state<>'deleting'`, edit.OwnerID, edit.ProjectID).Scan(&active); err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return auditstore.ProjectFinding{}, auditstore.ErrNotFound
		}
		return auditstore.ProjectFinding{}, err
	}
	if err := NewFindingEventsTx(s.tx, s.storage).LockFindingCorrelationFence(ctx, edit.ProjectID); err != nil {
		return auditstore.ProjectFinding{}, err
	}
	if edit.DuplicateTargetID != nil {
		var exists, cycle, tooDeep bool
		if err := s.tx.QueryRow(ctx, projectDuplicateTargetSQL, edit.ProjectID, *edit.DuplicateTargetID, edit.FindingID).Scan(&exists, &cycle, &tooDeep); err != nil {
			return auditstore.ProjectFinding{}, err
		}
		if !exists {
			return auditstore.ProjectFinding{}, auditstore.ErrNotFound
		}
		if cycle || tooDeep || *edit.DuplicateTargetID == edit.FindingID {
			return auditstore.ProjectFinding{}, auditstore.ErrInvalid
		}
	}
	result, err := scanProjectFinding(s.tx.QueryRow(ctx, assessProjectFindingSQL, edit.OwnerID, edit.ProjectID,
		edit.FindingID, edit.ExpectedRevision, edit.Verdict, edit.Severity, edit.Rationale, edit.DuplicateTargetID))
	if errors.Is(err, auditstore.ErrNotFound) {
		return result, auditstore.ErrPrecondition
	}
	if err != nil {
		return result, err
	}
	scope, err := artifacts.ProjectScope(result.ProjectID)
	if err != nil {
		return result, err
	}
	body, err := json.Marshal(map[string]any{"schema": auditdomain.FindingSourceHistorySchema,
		"receiptId": result.FirstReceiptID, "kind": "analyst-edit", "original": result, "materials": map[string]any{}})
	if err != nil {
		return result, err
	}
	if _, err := artifacts.NewService(s.storage.Repository(s.tx)).CreateFrozen(ctx, scope,
		contracts.ArtifactRef{Namespace: artifactpolicy.FindingHistoryNamespace,
			Name: auditdomain.FindingSourceHistoryName(result.FirstReceiptID, "analyst-edit", result.FindingID+":"+strconv.FormatUint(result.Revision, 10))},
		artifacts.Payload{MediaType: "application/json", Data: body}); err != nil {
		return result, err
	}
	if _, err := s.tx.Exec(ctx, `SELECT contractor_advance_finding_correlation(ARRAY[$1::text])`, edit.ProjectID); err != nil {
		return result, err
	}
	return result, nil
}
