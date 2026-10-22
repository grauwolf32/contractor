package pgstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/pagelimit"
	"github.com/jackc/pgx/v5"
)

// ListFindings reads one keyset page of the findings of an owner's Audit in
// creation order. An empty page of a missing or foreign Audit is
// auditstore.ErrNotFound.
func (s *QueryStore) ListFindings(
	ctx context.Context, params auditstore.FindingListParams,
) ([]auditstore.FindingRecord, error) {
	if err := validateFindingList(params); err != nil {
		return nil, err
	}
	var state, verdict, severity *string
	if params.State != nil {
		value := string(*params.State)
		state = &value
	}
	if params.Verdict != nil {
		value := string(*params.Verdict)
		verdict = &value
	}
	if params.Severity != nil {
		value := string(*params.Severity)
		severity = &value
	}
	rows, err := s.db.Query(ctx, listFindingsSQL, params.OwnerID, params.AuditID, state, verdict, params.Unreviewed,
		severity, params.AfterCreatedAt, params.AfterFindingID, params.Limit)
	if err != nil {
		return nil, fmt.Errorf("list Audit findings: %w", err)
	}
	defer rows.Close()
	result := make([]auditstore.FindingRecord, 0, params.Limit)
	for rows.Next() {
		row, scanErr := scanCorrelatedFindingRecord(rows)
		if scanErr != nil {
			return nil, scanErr
		}
		result = append(result, row)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate Audit findings: %w", err)
	}
	rows.Close()
	if len(result) == 0 {
		if _, err := s.get(ctx, params.OwnerID, params.AuditID); err != nil {
			return nil, err
		}
	}
	return result, nil
}

// ListOwnerFindings reads one keyset page of every finding of an owner's
// Audits in listed Projects, newest first.
func (s *QueryStore) ListOwnerFindings(
	ctx context.Context, params auditstore.OwnerFindingListParams,
) ([]auditstore.FindingRecord, error) {
	if err := validateOwnerPage(params.OwnerID, params.Limit, params.BeforeCreatedAt, params.BeforeFindingID, params.AuditStates); err != nil {
		return nil, err
	}
	if (params.State != nil && !params.State.Valid()) || (params.Severity != nil && !params.Severity.Valid()) ||
		(params.Verdict != nil && *params.Verdict != auditstore.VerdictTruePositive && *params.Verdict != auditstore.VerdictFalsePositive) ||
		(params.Unreviewed && (params.Verdict != nil || params.Severity != nil)) {
		return nil, auditstore.ErrInvalid
	}
	rows, err := s.db.Query(ctx, listOwnerFindingsSQL,
		params.OwnerID, nil, params.State, params.Verdict, params.Unreviewed, params.Severity,
		params.BeforeCreatedAt, params.BeforeFindingID, params.Limit, params.AuditStates)
	if err != nil {
		return nil, fmt.Errorf("list owner findings: %w", err)
	}
	defer rows.Close()
	result := make([]auditstore.FindingRecord, 0, params.Limit)
	for rows.Next() {
		row, err := scanCorrelatedFindingRecord(rows)
		if err != nil {
			return nil, err
		}
		result = append(result, row)
	}
	return result, rows.Err()
}

// GetFinding reads one finding of an owner's Audit.
func (s *QueryStore) GetFinding(
	ctx context.Context, ownerID, auditID, findingID string,
) (auditstore.FindingRecord, error) {
	if !auditstore.ValidReviewIdentity(ownerID, 256) || !auditstore.ValidReviewIdentity(auditID, 256) ||
		!auditstore.ValidReviewIdentity(findingID, 256) {
		return auditstore.FindingRecord{}, auditstore.ErrInvalid
	}
	row, err := scanCorrelatedFindingRecord(s.db.QueryRow(ctx, getCorrelatedFindingSQL, ownerID, auditID, findingID))
	if errors.Is(err, pgx.ErrNoRows) {
		return auditstore.FindingRecord{}, auditstore.ErrNotFound
	}
	if err != nil {
		return auditstore.FindingRecord{}, fmt.Errorf("read Audit finding: %w", err)
	}
	return row, nil
}

// FindingPageBasis reads the revision of an owner's Audit, the statement
// time and the number of the Audit's findings that match the page's
// filters, in one statement.
func (s *QueryStore) FindingPageBasis(
	ctx context.Context, params auditstore.FindingListParams,
) (auditstore.PageBasis, error) {
	var result auditstore.PageBasis
	if err := validateFindingList(params); err != nil {
		return result, err
	}
	err := s.db.QueryRow(ctx, findingPageBasisSQL,
		params.OwnerID, params.AuditID, params.State, params.Verdict, params.Unreviewed, params.Severity,
	).Scan(&result.AuditRevision, &result.AsOf, &result.Total)
	if errors.Is(err, pgx.ErrNoRows) {
		return result, auditstore.ErrNotFound
	}
	return result, err
}

// ReadFindingDecisions reads analyst decisions by their (Audit, decision)
// pairs, each through its own Audit of the owner, keyed by decision.
func (s *QueryStore) ReadFindingDecisions(
	ctx context.Context, ownerID string, auditIDs, decisionIDs []string,
) (map[string]auditstore.ReviewDecision, error) {
	result := make(map[string]auditstore.ReviewDecision, len(decisionIDs))
	if len(auditIDs) != len(decisionIDs) {
		return nil, auditstore.ErrInvalid
	}
	if len(decisionIDs) == 0 {
		return result, nil
	}
	rows, err := s.db.Query(ctx, decisionSelect+`
  JOIN audits AS audit ON audit.audit_id = decision.audit_id
 WHERE audit.owner_id = $1
   AND (decision.audit_id, decision.decision_id) IN (
       SELECT * FROM unnest($2::text[], $3::text[]))`, ownerID, auditIDs, decisionIDs)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	for rows.Next() {
		decision, err := scanDecision(rows)
		if err != nil {
			return nil, err
		}
		result[decision.DecisionID] = decision
	}
	return result, rows.Err()
}

// ReadFindingAssessments reads finding assessments by their (Audit,
// assessment) pairs, each through its own Audit of the owner, keyed by
// assessment.
func (s *QueryStore) ReadFindingAssessments(
	ctx context.Context, ownerID string, auditIDs, assessmentIDs []string,
) (map[string]auditstore.FindingAssessment, error) {
	result := make(map[string]auditstore.FindingAssessment, len(assessmentIDs))
	if len(auditIDs) != len(assessmentIDs) {
		return nil, auditstore.ErrInvalid
	}
	if len(assessmentIDs) == 0 {
		return result, nil
	}
	rows, err := s.db.Query(ctx, assessmentSelect+`
  JOIN audits AS audit ON audit.audit_id = assessment.audit_id
 WHERE audit.owner_id = $1
   AND (assessment.audit_id, assessment.assessment_id) IN (
       SELECT * FROM unnest($2::text[], $3::text[]))`, ownerID, auditIDs, assessmentIDs)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	for rows.Next() {
		assessment, err := scanAssessment(rows)
		if err != nil {
			return nil, err
		}
		result[assessment.AssessmentID] = assessment
	}
	return result, rows.Err()
}

func scanFindingRecord(row scanner) (auditstore.FindingRecord, error) {
	var result auditstore.FindingRecord
	var revision int64
	err := row.Scan(
		&result.FindingID, &result.AuditID, &result.FirstReceiptID, &result.State,
		&result.RejectionReason, &result.DuplicateTargetID, &result.CurrentAssessmentID,
		&result.CurrentDecisionID, &revision, &result.CreatedAt, &result.UpdatedAt,
		&result.CorrelationProjectID, &result.CorrelationAlgorithm, &result.CorrelationValue, &result.CorrelationGroupKey, &result.CorrelationReason,
	)
	if err != nil {
		return auditstore.FindingRecord{}, err
	}
	if revision < 1 || !result.State.Valid() {
		return auditstore.FindingRecord{}, errors.New("stored Audit finding is invalid")
	}
	result.Revision = uint64(revision)
	if result.CorrelationGroupKey != "" {
		expected, keyErr := auditdomain.FindingGroupKey(result.CorrelationProjectID, result.CorrelationAlgorithm, result.CorrelationValue)
		if keyErr != nil || expected != result.CorrelationGroupKey || result.CorrelationReason != "" {
			return auditstore.FindingRecord{}, errors.New("stored finding primary key is invalid")
		}
	} else if result.CorrelationAlgorithm != "" || result.CorrelationValue != "" ||
		result.CorrelationReason != "not_recorded" && result.CorrelationReason != "no_qualified_key" && result.CorrelationReason != "primary_key_ambiguous" {
		return auditstore.FindingRecord{}, errors.New("stored finding primary unavailability is invalid")
	}
	return result, nil
}

func scanDecision(row scanner) (auditstore.ReviewDecision, error) {
	var result auditstore.ReviewDecision
	var findingID, action, verdict, severity *string
	err := row.Scan(&result.DecisionID, &result.RequestID, &result.AuditID,
		&findingID, &result.ActorID, &action, &verdict, &severity,
		&result.Rationale, &result.DuplicateTargetID, &result.SubjectRevision,
		&result.SubjectDigest, &result.CreatedAt)
	if err != nil {
		return auditstore.ReviewDecision{}, err
	}
	if findingID != nil {
		result.FindingID = *findingID
	}
	if action != nil && findingID == nil {
		result.Action = auditstore.ReviewAction(*action)
	}
	if verdict != nil {
		result.Verdict = auditstore.AnalystVerdict(*verdict)
	}
	if severity != nil {
		value := auditstore.FindingSeverity(*severity)
		result.Severity = &value
	}
	return result, nil
}

func scanAssessment(row scanner) (auditstore.FindingAssessment, error) {
	var result auditstore.FindingAssessment
	var resultRef, contractRef []byte
	var contractDigest *string
	err := row.Scan(&result.AssessmentID, &result.SemanticAssessment,
		&resultRef, &result.Result.Digest, &result.ReceiptID,
		&result.ItemID, &result.ExecutionItemID, &result.CollectionReceiptID,
		&result.DirectVerification, &contractRef, &contractDigest, &result.AcceptedAt)
	if err != nil {
		return auditstore.FindingAssessment{}, err
	}
	if json.Unmarshal(resultRef, &result.Result.Ref) != nil || result.Result.Ref.ValidateExact() != nil {
		return auditstore.FindingAssessment{}, errors.New("stored finding assessment is invalid")
	}
	if contractRef != nil {
		if contractDigest == nil {
			return auditstore.FindingAssessment{}, errors.New("stored finding assessment contract is invalid")
		}
		result.Contract = &auditstore.ExactArtifact{Digest: *contractDigest}
		if json.Unmarshal(contractRef, &result.Contract.Ref) != nil {
			return auditstore.FindingAssessment{}, errors.New("stored finding assessment contract is invalid")
		}
	}
	return result, nil
}

func validateFindingList(params auditstore.FindingListParams) error {
	if !auditstore.ValidReviewIdentity(params.OwnerID, 256) || !auditstore.ValidReviewIdentity(params.AuditID, 256) ||
		!pagelimit.ValidFetch(params.Limit) ||
		(params.AfterCreatedAt == nil) != (params.AfterFindingID == "") ||
		(params.State != nil && !params.State.Valid()) ||
		(params.Severity != nil && !params.Severity.Valid()) {
		return auditstore.ErrInvalid
	}
	if params.Verdict != nil && *params.Verdict != auditstore.VerdictTruePositive && *params.Verdict != auditstore.VerdictFalsePositive {
		return auditstore.ErrInvalid
	}
	if params.Unreviewed && (params.Verdict != nil || params.Severity != nil) {
		return auditstore.ErrInvalid
	}
	return nil
}

func validateReviewList(params auditstore.ReviewListParams) error {
	if !auditstore.ValidReviewIdentity(params.OwnerID, 256) || !auditstore.ValidReviewIdentity(params.AuditID, 256) ||
		!pagelimit.ValidFetch(params.Limit) ||
		(params.AfterCreatedAt == nil) != (params.AfterRequestID == "") ||
		(params.State != nil && !params.State.Valid()) ||
		(params.FindingID != nil && !auditstore.ValidReviewIdentity(*params.FindingID, 256)) {
		return auditstore.ErrInvalid
	}
	return nil
}

func validateProvenanceList(params auditstore.ProvenanceListParams) error {
	if !auditstore.ValidReviewIdentity(params.OwnerID, 256) || !auditstore.ValidReviewIdentity(params.AuditID, 256) ||
		!auditstore.ValidReviewIdentity(params.FindingID, 256) || !pagelimit.ValidFetch(params.Limit) ||
		(params.AfterCreatedAt == nil) != (params.AfterRecordID == "") {
		return auditstore.ErrInvalid
	}
	return nil
}

func validateOwnerPage(ownerID string, limit int, before *time.Time, id string, states []auditstore.AuditState) error {
	if !auditstore.ValidReviewIdentity(ownerID, 256) || !pagelimit.ValidFetch(limit) ||
		(before == nil) != (id == "") ||
		(before != nil && (before.IsZero() || !auditstore.ValidReviewIdentity(id, 256))) || len(states) > 10 {
		return auditstore.ErrInvalid
	}
	for _, state := range states {
		if !state.Valid() {
			return auditstore.ErrInvalid
		}
	}
	return nil
}

func scanCorrelatedFindingRecord(row scanner) (auditstore.FindingRecord, error) {
	var revision int64
	var representativeID, representativeAudit *string
	var hasOthers bool
	result, err := scanFindingRecord(withColumns{row: row, after: []any{&revision, &representativeID, &representativeAudit, &hasOthers}})
	if err != nil {
		return auditstore.FindingRecord{}, err
	}
	projection := auditdomain.FindingCorrelation{
		Schema: auditdomain.FindingCorrelationSchema, PolicyVersion: auditdomain.FindingCorrelationPolicy,
		Scope:           auditdomain.FindingCorrelationScope{Kind: "project", ProjectID: result.CorrelationProjectID},
		ProjectRevision: uint64(revision), Availability: "unavailable", Reason: result.CorrelationReason,
	}
	if result.CorrelationGroupKey != "" {
		if representativeID == nil || representativeAudit == nil {
			return auditstore.FindingRecord{}, errors.New("stored finding group has no representative")
		}
		projection.Availability = "available"
		projection.GroupKey = result.CorrelationGroupKey
		projection.Basis = &auditdomain.FindingCorrelationBasis{Algorithm: result.CorrelationAlgorithm, Value: result.CorrelationValue}
		projection.MatchedDomain = "snapshot"
		projection.Representative = &auditdomain.FindingCorrelationReference{AuditID: *representativeAudit, FindingID: *representativeID}
		projection.AutomaticDuplicate = *representativeID != result.FindingID
		projection.HasOtherObservations = hasOthers
	} else if representativeID != nil || representativeAudit != nil || hasOthers {
		return auditstore.FindingRecord{}, errors.New("unqualified finding has group presentation")
	}
	if revision < 1 || auditdomain.ValidateFindingCorrelation(projection, result.AuditID, result.FindingID) != nil {
		return auditstore.FindingRecord{}, errors.New("stored finding correlation presentation is invalid")
	}
	result.Correlation = &projection
	return result, nil
}
