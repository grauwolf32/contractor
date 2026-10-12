package pgstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/jackc/pgx/v5"
)

// ReadProvenanceRevisions reads the revisions of an owner's Audit and of one
// of its findings, which bound a consistent provenance page.
func (s *QueryStore) ReadProvenanceRevisions(
	ctx context.Context, params auditstore.ProvenanceListParams,
) (auditRevision, findingRevision uint64, err error) {
	if err := validateProvenanceList(params); err != nil {
		return 0, 0, err
	}
	err = s.db.QueryRow(ctx, `
SELECT audit.revision, finding.revision
  FROM audits AS audit
  JOIN audit_findings AS finding USING (audit_id)
 WHERE audit.owner_id = $1 AND audit.audit_id = $2 AND finding.finding_id = $3`,
		params.OwnerID, params.AuditID, params.FindingID).Scan(&auditRevision, &findingRevision)
	if errors.Is(err, pgx.ErrNoRows) {
		return 0, 0, auditstore.ErrNotFound
	}
	if err != nil {
		return 0, 0, fmt.Errorf("read finding provenance revisions: %w", err)
	}
	return auditRevision, findingRevision, nil
}

// ListFindingProvenance reads one keyset page of the records that produced a
// finding: its source proposals, the check attempts that verified them and
// its direct verifications.
func (s *QueryStore) ListFindingProvenance(
	ctx context.Context, params auditstore.ProvenanceListParams,
) ([]auditstore.FindingProvenanceRecord, error) {
	if err := validateProvenanceList(params); err != nil {
		return nil, err
	}
	rows, err := s.db.Query(ctx, listFindingProvenanceSQL, params.AuditID, params.FindingID,
		params.AfterCreatedAt, params.AfterRecordID, params.Limit)
	if err != nil {
		return nil, fmt.Errorf("list finding provenance: %w", err)
	}
	defer rows.Close()
	result := make([]auditstore.FindingProvenanceRecord, 0, params.Limit)
	for rows.Next() {
		value, err := scanProvenanceRecord(rows)
		if err != nil {
			return nil, err
		}
		result = append(result, value)
	}
	return result, rows.Err()
}

func scanProvenanceRecord(rows pgx.Rows) (auditstore.FindingProvenanceRecord, error) {
	var value auditstore.FindingProvenanceRecord
	var kind string
	var proposalJSON, resultJSON, contractJSON []byte
	var assessmentID, semantic, resultDigest, itemID, executionItemID, collectionID, contractDigest *string
	var executionID, executionRole, itemState, disposition, terminalOutcome, runID *string
	var itemAttempt *int
	var runProvenanceJSON, taskJSON, itemOriginJSON []byte
	var taskDigest, workflowRole *string
	var runDeleted bool
	var collectedAt, assessmentAcceptedAt *time.Time
	var direct bool
	if err := rows.Scan(
		&value.CreatedAt, &value.RecordID, &kind, &value.ReceiptID, &value.Relation,
		&proposalJSON, &assessmentID, &semantic, &resultJSON, &resultDigest,
		&itemID, &executionItemID, &collectionID, &direct, &contractJSON, &contractDigest,
		&executionID, &executionRole, &itemAttempt, &itemState, &disposition,
		&terminalOutcome, &runID, &runProvenanceJSON, &runDeleted,
		&taskJSON, &taskDigest, &itemOriginJSON, &workflowRole, &collectedAt,
		&assessmentAcceptedAt,
	); err != nil {
		return value, fmt.Errorf("scan finding provenance: %w", err)
	}
	value.Kind = auditstore.FindingProvenanceKind(kind)
	if !value.Kind.Valid() {
		return value, errors.New("stored finding provenance kind is invalid")
	}
	if json.Unmarshal(proposalJSON, &value.Proposal) != nil || value.Proposal.Ref.ValidateExact() != nil {
		return value, errors.New("stored finding proposal provenance is invalid")
	}
	if assessmentID != nil {
		if semantic == nil || assessmentAcceptedAt == nil {
			return value, errors.New("stored finding assessment provenance is incomplete")
		}
		assessment := &auditstore.FindingAssessment{
			AssessmentID: *assessmentID, SemanticAssessment: *semantic,
			ReceiptID: value.ReceiptID, ItemID: itemID, ExecutionItemID: executionItemID,
			CollectionReceiptID: collectionID, DirectVerification: direct,
			AcceptedAt: *assessmentAcceptedAt,
		}
		if resultDigest == nil || json.Unmarshal(resultJSON, &assessment.Result.Ref) != nil {
			return value, errors.New("stored finding assessment provenance is invalid")
		}
		assessment.Result.Digest = *resultDigest
		if contractJSON != nil {
			if contractDigest == nil {
				return value, errors.New("stored direct verification contract is invalid")
			}
			assessment.Contract = &auditstore.ExactArtifact{Digest: *contractDigest}
			if json.Unmarshal(contractJSON, &assessment.Contract.Ref) != nil {
				return value, errors.New("stored direct verification contract is invalid")
			}
		}
		value.Assessment = assessment
	}
	if executionID != nil {
		if executionRole == nil || itemAttempt == nil || itemState == nil || itemID == nil ||
			executionItemID == nil || taskDigest == nil || workflowRole == nil {
			return value, errors.New("stored finding attempt provenance is incomplete")
		}
		attempt := &auditstore.FindingAttempt{
			ExecutionID: *executionID, ExecutionItemID: *executionItemID,
			ItemID: *itemID, ItemAttempt: *itemAttempt,
			Role: auditstore.ExecutionRole(*executionRole), WorkflowRole: *workflowRole,
			State: auditstore.ItemState(*itemState), RunID: runID,
			RunDeleted: runDeleted, CreatedAt: value.CreatedAt, CollectedAt: collectedAt,
			Task: auditstore.ExactArtifact{Digest: *taskDigest},
		}
		if !attempt.Role.Valid() || !attempt.State.Valid() || attempt.ItemAttempt < 1 {
			return value, errors.New("stored finding attempt provenance is invalid")
		}
		if disposition != nil {
			parsed := auditstore.CollectionDisposition(*disposition)
			if !parsed.Valid() {
				return value, errors.New("stored finding attempt disposition is invalid")
			}
			attempt.CollectionDisposition = &parsed
		}
		if terminalOutcome != nil {
			parsed := auditstore.TerminalOutcome(*terminalOutcome)
			if !parsed.Valid() {
				return value, errors.New("stored finding attempt terminal outcome is invalid")
			}
			attempt.TerminalOutcome = &parsed
		}
		if len(runProvenanceJSON) != 0 {
			attempt.RunProvenance = &auditstore.RunProvenance{}
			if json.Unmarshal(runProvenanceJSON, attempt.RunProvenance) != nil {
				return value, errors.New("stored finding Run provenance is invalid")
			}
		}
		if json.Unmarshal(taskJSON, &attempt.Task.Ref) != nil ||
			attempt.Task.Ref.ValidateExact() != nil ||
			json.Unmarshal(itemOriginJSON, &attempt.ItemOrigin) != nil ||
			attempt.ItemOrigin.Schema != auditstore.ItemOriginSchema ||
			attempt.ItemOrigin.EntryKey == "" {
			return value, errors.New("stored finding item provenance is invalid")
		}
		if resultDigest != nil && len(resultJSON) != 0 {
			attempt.Result = &auditstore.ExactArtifact{Digest: *resultDigest}
			if json.Unmarshal(resultJSON, &attempt.Result.Ref) != nil ||
				attempt.Result.Ref.ValidateExact() != nil {
				return value, errors.New("stored finding attempt result is invalid")
			}
		}
		value.Attempt = attempt
	}
	return value, nil
}
