package auditservice

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/auditstore"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

type findingRow struct {
	findingID           string
	auditID             string
	firstReceiptID      string
	state               FindingState
	rejectionReason     *string
	duplicateTargetID   *string
	currentAssessmentID *string
	currentDecisionID   *string
	revision            uint64
	createdAt           time.Time
	updatedAt           time.Time
}

type reviewQuerier interface {
	QueryRow(context.Context, string, ...any) pgx.Row
	Query(context.Context, string, ...any) (pgx.Rows, error)
}

const findingRowColumns = `
finding.finding_id, finding.audit_id, finding.first_receipt_id, finding.state,
finding.rejection_reason, finding.duplicate_target_id,
finding.current_assessment_id, finding.current_decision_id,
finding.revision, finding.created_at, finding.updated_at`

func (s *Service) ListFindings(
	ctx context.Context, params FindingListParams,
) ([]Finding, error) {
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
	rows, err := s.pool.Query(ctx, `
SELECT `+findingRowColumns+`
  FROM audit_findings AS finding
  JOIN audits AS audit USING (audit_id)
  LEFT JOIN audit_review_decisions AS decision
    ON decision.decision_id = finding.current_decision_id
 WHERE audit.owner_id = $1 AND finding.audit_id = $2
   AND ($3::text IS NULL OR finding.state = $3)
   AND ($4::text IS NULL OR decision.verdict = $4)
   AND (NOT $5::boolean OR finding.current_decision_id IS NULL)
   AND ($6::text IS NULL OR decision.severity = $6)
   AND ($7::timestamptz IS NULL OR
        (finding.created_at, finding.finding_id) > ($7, $8))
 ORDER BY finding.created_at, finding.finding_id
 LIMIT $9`, params.OwnerID, params.AuditID, state, verdict, params.Unreviewed,
		severity, params.AfterCreatedAt, params.AfterFindingID, params.Limit)
	if err != nil {
		return nil, fmt.Errorf("list Audit findings: %w", err)
	}
	defer rows.Close()
	result := make([]Finding, 0, params.Limit)
	for rows.Next() {
		row, scanErr := scanFindingRow(rows)
		if scanErr != nil {
			return nil, scanErr
		}
		finding, loadErr := s.hydrateFinding(ctx, params.OwnerID, row)
		if loadErr != nil {
			return nil, loadErr
		}
		result = append(result, finding)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate Audit findings: %w", err)
	}
	if len(result) == 0 {
		if _, err := auditstore.NewPostgresStore(s.pool).Get(ctx, params.OwnerID, params.AuditID); err != nil {
			return nil, err
		}
	}
	return result, nil
}

func (s *Service) GetFinding(
	ctx context.Context, ownerID, auditID, findingID string,
) (Finding, error) {
	if !validReviewIdentity(ownerID, 256) || !validReviewIdentity(auditID, 256) ||
		!validReviewIdentity(findingID, 256) {
		return Finding{}, auditstore.ErrInvalid
	}
	row, err := scanFindingRow(s.pool.QueryRow(ctx, `
SELECT `+findingRowColumns+`
  FROM audit_findings AS finding
  JOIN audits AS audit USING (audit_id)
 WHERE audit.owner_id = $1 AND finding.audit_id = $2 AND finding.finding_id = $3`,
		ownerID, auditID, findingID))
	if errors.Is(err, pgx.ErrNoRows) {
		return Finding{}, auditstore.ErrNotFound
	}
	if err != nil {
		return Finding{}, fmt.Errorf("read Audit finding: %w", err)
	}
	return s.hydrateFinding(ctx, ownerID, row)
}

func (s *Service) CreateFindingReview(
	ctx context.Context, params CreateFindingReviewParams,
) (FindingReviewResult, error) {
	if err := validateCreateFindingReview(params, s.now()); err != nil {
		return FindingReviewResult{}, err
	}
	var requestID string
	var replayed bool
	err := persistencepostgres.InTx(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if _, err := tx.Exec(ctx, `SELECT pg_advisory_xact_lock(hashtextextended($1, 0))`,
			"audit-finding-review:"+params.AuditID); err != nil {
			return err
		}
		var storedDigest string
		err := tx.QueryRow(ctx, `
SELECT request_id, request_digest
  FROM audit_review_requests
 WHERE audit_id = $1 AND idempotency_key = $2`, params.AuditID, params.IdempotencyKey).Scan(
			&requestID, &storedDigest,
		)
		if err == nil {
			if storedDigest != params.RequestDigest {
				return auditstore.ErrConflict
			}
			replayed = true
			return nil
		}
		if !errors.Is(err, pgx.ErrNoRows) {
			return err
		}
		row, state, err := lockFinding(ctx, tx, params.OwnerID, params.AuditID, params.FindingID)
		if err != nil {
			return err
		}
		if state == auditstore.AuditDeleting || row.revision != params.ExpectedRevision {
			return auditstore.ErrPrecondition
		}
		now := s.now().UTC()
		if _, err := tx.Exec(ctx, `
UPDATE audit_review_requests
   SET state = 'expired', revision = revision + 1,
       updated_at = GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
 WHERE audit_id = $1 AND finding_id = $2 AND state = 'pending'
   AND expires_at IS NOT NULL AND expires_at <= $3`, params.AuditID, params.FindingID, now); err != nil {
			return err
		}
		subjectDigest := findingSubjectDigest(row)
		err = tx.QueryRow(ctx, `
SELECT request_id
  FROM audit_review_requests
 WHERE audit_id = $1 AND finding_id = $2 AND kind = 'finding-triage'
   AND subject_revision = $3 AND subject_digest = $4 AND state = 'pending'
 ORDER BY created_at LIMIT 1`, params.AuditID, params.FindingID, row.revision, subjectDigest).Scan(&requestID)
		if err == nil {
			replayed = true
			return nil
		}
		if !errors.Is(err, pgx.ErrNoRows) {
			return err
		}
		expiresAt := now.Add(defaultReviewTTL)
		if params.ExpiresAt != nil {
			expiresAt = params.ExpiresAt.UTC()
		}
		actions, _ := json.Marshal([]AnalystVerdict{
			VerdictTruePositive, VerdictFalsePositive, VerdictDuplicate,
			VerdictReopen, VerdictNeedsEvidence,
		})
		requestID = params.RequestID
		if _, err := tx.Exec(ctx, `
INSERT INTO audit_review_requests (
    request_id, audit_id, finding_id, kind, subject_revision, subject_digest,
    requested_actions, expires_at, idempotency_key, request_digest
) VALUES ($1, $2, $3, 'finding-triage', $4, $5, $6, $7, $8, $9)`,
			requestID, params.AuditID, params.FindingID, row.revision, subjectDigest,
			actions, expiresAt, params.IdempotencyKey, params.RequestDigest); err != nil {
			if persistencepostgres.SQLState(err) == "23505" {
				return auditstore.ErrConflict
			}
			return err
		}
		return appendAuditReviewEvent(ctx, tx, params.AuditID, "review.requested", requestID, nil,
			map[string]any{"findingId": params.FindingID, "kind": FindingReviewKind})
	})
	if err != nil {
		return FindingReviewResult{}, err
	}
	request, err := s.GetReview(ctx, params.OwnerID, params.AuditID, requestID)
	return FindingReviewResult{Request: request, Replayed: replayed}, err
}

func (s *Service) DecideFinding(
	ctx context.Context, params DecideFindingParams,
) (FindingDecisionResult, error) {
	if err := validateFindingDecision(params); err != nil {
		return FindingDecisionResult{}, err
	}
	var decisionID, requestID, findingID string
	var replayed bool
	var expired bool
	err := persistencepostgres.InTx(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if _, err := tx.Exec(ctx, `SELECT pg_advisory_xact_lock(hashtextextended($1, 0))`,
			"audit-finding-review:"+params.AuditID); err != nil {
			return err
		}
		var storedDigest string
		err := tx.QueryRow(ctx, `
SELECT decision_id, request_id, finding_id, request_digest
  FROM audit_review_decisions
 WHERE audit_id = $1 AND idempotency_key = $2`, params.AuditID, params.IdempotencyKey).Scan(
			&decisionID, &requestID, &findingID, &storedDigest,
		)
		if err == nil {
			if storedDigest != params.RequestDigest || requestID != params.RequestID {
				return auditstore.ErrConflict
			}
			replayed = true
			return nil
		}
		if !errors.Is(err, pgx.ErrNoRows) {
			return err
		}

		var auditState auditstore.AuditState
		var request ReviewRequest
		var row findingRow
		var requestedActionsJSON []byte
		var requestRevision int64
		var findingRevision int64
		err = tx.QueryRow(ctx, `
SELECT audit.state,
       request.request_id, request.audit_id, request.finding_id, request.kind,
       request.subject_revision, request.subject_digest, request.requested_actions,
       request.state, request.expires_at, request.revision,
       request.created_at, request.updated_at,
       `+findingRowColumns+`
  FROM audit_review_requests AS request
  JOIN audit_findings AS finding
    ON finding.finding_id = request.finding_id AND finding.audit_id = request.audit_id
  JOIN audits AS audit ON audit.audit_id = request.audit_id
 WHERE audit.owner_id = $1 AND request.audit_id = $2 AND request.request_id = $3
 FOR UPDATE OF audit, request, finding`, params.OwnerID, params.AuditID, params.RequestID).Scan(
			&auditState,
			&request.RequestID, &request.AuditID, &request.FindingID, &request.Kind,
			&request.SubjectRevision, &request.SubjectDigest, &requestedActionsJSON,
			&request.State, &request.ExpiresAt, &requestRevision,
			&request.CreatedAt, &request.UpdatedAt,
			&row.findingID, &row.auditID, &row.firstReceiptID, &row.state,
			&row.rejectionReason, &row.duplicateTargetID, &row.currentAssessmentID,
			&row.currentDecisionID, &findingRevision, &row.createdAt, &row.updatedAt,
		)
		if errors.Is(err, pgx.ErrNoRows) {
			return auditstore.ErrNotFound
		}
		if err != nil {
			return err
		}
		request.Revision, row.revision = uint64(requestRevision), uint64(findingRevision)
		if auditState == auditstore.AuditDeleting || request.State != ReviewPending ||
			request.Revision != params.ExpectedRequestRevision || request.Kind != FindingReviewKind {
			return auditstore.ErrPrecondition
		}
		if request.ExpiresAt != nil && !s.now().UTC().Before(*request.ExpiresAt) {
			if _, updateErr := tx.Exec(ctx, `
UPDATE audit_review_requests SET state = 'expired', revision = revision + 1,
 updated_at = GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
 WHERE request_id = $1 AND state = 'pending'`, request.RequestID); updateErr != nil {
				return updateErr
			}
			// Commit the expiry transition before reporting the failed decision.
			// Returning ErrPrecondition from inside the transaction would roll the
			// update back and leave the request permanently pending.
			expired = true
			return nil
		}
		if request.SubjectRevision != row.revision || request.SubjectDigest != findingSubjectDigest(row) {
			return auditstore.ErrPrecondition
		}
		if params.Verdict == VerdictDuplicate {
			if err := validateDuplicateTarget(ctx, tx, params.AuditID, row.findingID, *params.DuplicateTargetID); err != nil {
				return err
			}
		}
		decisionID, requestID, findingID = params.DecisionID, request.RequestID, row.findingID
		var severity, target *string
		if params.Severity != nil {
			value := string(*params.Severity)
			severity = &value
		}
		if params.DuplicateTargetID != nil {
			target = params.DuplicateTargetID
		}
		if _, err := tx.Exec(ctx, `
INSERT INTO audit_review_decisions (
    decision_id, request_id, audit_id, finding_id, actor_id, verdict, severity,
    rationale, duplicate_target_id, subject_revision, subject_digest,
    idempotency_key, request_digest
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)`,
			decisionID, requestID, params.AuditID, findingID, params.OwnerID,
			string(params.Verdict), severity, params.Rationale, target,
			request.SubjectRevision, request.SubjectDigest, params.IdempotencyKey,
			params.RequestDigest); err != nil {
			if persistencepostgres.SQLState(err) == "23505" {
				return auditstore.ErrConflict
			}
			return err
		}
		if _, err := tx.Exec(ctx, `
UPDATE audit_review_requests
   SET state = 'decided', revision = revision + 1,
       updated_at = GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
 WHERE request_id = $1 AND state = 'pending'`, requestID); err != nil {
			return err
		}
		state, rejection, duplicate, effectiveDecision := decisionProjection(params.Verdict, decisionID, target)
		if _, err := tx.Exec(ctx, `
UPDATE audit_findings
   SET state = $2, rejection_reason = $3, duplicate_target_id = $4,
       current_decision_id = $5, revision = revision + 1,
       updated_at = GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
 WHERE finding_id = $1`, findingID, state, rejection, duplicate, effectiveDecision); err != nil {
			return err
		}
		return appendAuditReviewEvent(ctx, tx, params.AuditID, "review.decided", decisionID, nil,
			map[string]any{"findingId": findingID, "verdict": params.Verdict})
	})
	if err != nil {
		return FindingDecisionResult{}, err
	}
	if expired {
		return FindingDecisionResult{}, auditstore.ErrPrecondition
	}
	request, err := s.GetReview(ctx, params.OwnerID, params.AuditID, requestID)
	if err != nil {
		return FindingDecisionResult{}, err
	}
	finding, err := s.GetFinding(ctx, params.OwnerID, params.AuditID, findingID)
	if err != nil {
		return FindingDecisionResult{}, err
	}
	if request.Decision == nil || request.Decision.DecisionID != decisionID {
		return FindingDecisionResult{}, errors.New("stored finding review decision is missing")
	}
	return FindingDecisionResult{
		Finding: finding, Request: request, Decision: *request.Decision, Replayed: replayed,
	}, nil
}

func (s *Service) GetReview(
	ctx context.Context, ownerID, auditID, requestID string,
) (ReviewRequest, error) {
	if !validReviewIdentity(ownerID, 256) || !validReviewIdentity(auditID, 256) ||
		!validReviewIdentity(requestID, 256) {
		return ReviewRequest{}, auditstore.ErrInvalid
	}
	request, err := scanReviewRequest(s.pool.QueryRow(ctx, reviewSelect+`
 WHERE audit.owner_id = $1 AND request.audit_id = $2 AND request.request_id = $3`,
		ownerID, auditID, requestID))
	if errors.Is(err, pgx.ErrNoRows) {
		return ReviewRequest{}, auditstore.ErrNotFound
	}
	if err != nil {
		return ReviewRequest{}, err
	}
	return request, nil
}

func (s *Service) ListReviews(
	ctx context.Context, params ReviewListParams,
) ([]ReviewRequest, error) {
	if err := validateReviewList(params); err != nil {
		return nil, err
	}
	var state *string
	if params.State != nil {
		value := string(*params.State)
		state = &value
	}
	rows, err := s.pool.Query(ctx, reviewSelect+`
 WHERE audit.owner_id = $1 AND request.audit_id = $2
   AND ($3::text IS NULL OR request.finding_id = $3)
   AND ($4::text IS NULL OR request.state = $4)
   AND ($5::timestamptz IS NULL OR
        (request.created_at, request.request_id) > ($5, $6))
 ORDER BY request.created_at, request.request_id LIMIT $7`,
		params.OwnerID, params.AuditID, params.FindingID, state,
		params.AfterCreatedAt, params.AfterRequestID, params.Limit)
	if err != nil {
		return nil, fmt.Errorf("list Audit reviews: %w", err)
	}
	defer rows.Close()
	result := make([]ReviewRequest, 0, params.Limit)
	for rows.Next() {
		request, scanErr := scanReviewRequest(rows)
		if scanErr != nil {
			return nil, scanErr
		}
		result = append(result, request)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	if len(result) == 0 {
		if _, err := auditstore.NewPostgresStore(s.pool).Get(ctx, params.OwnerID, params.AuditID); err != nil {
			return nil, err
		}
	}
	return result, nil
}

func (s *Service) ListFindingProvenance(
	ctx context.Context, params ProvenanceListParams,
) ([]FindingProvenance, error) {
	if err := validateProvenanceList(params); err != nil {
		return nil, err
	}
	finding, err := s.GetFinding(ctx, params.OwnerID, params.AuditID, params.FindingID)
	if err != nil {
		return nil, err
	}
	rows, err := s.pool.Query(ctx, `
WITH records AS (
    SELECT contribution.created_at, 'proposal:' || contribution.receipt_id AS record_id,
           'source-proposal'::text AS kind, contribution.receipt_id,
           contribution.relation, contribution.proposal_ref,
           NULL::text AS assessment_id, NULL::text AS semantic_assessment,
           NULL::jsonb AS result_ref, NULL::text AS result_digest,
           NULL::text AS item_id, NULL::text AS execution_item_id,
           NULL::text AS collection_receipt_id, false AS direct_verification,
           NULL::jsonb AS contract_ref, NULL::text AS contract_digest
      FROM audit_finding_contributions AS contribution
     WHERE contribution.audit_id = $1 AND contribution.finding_id = $2
    UNION ALL
    SELECT assessment.accepted_at, 'assessment:' || assessment.assessment_id,
           CASE WHEN assessment.direct_verification THEN 'direct-verification'
                ELSE 'check-attempt' END,
           assessment.receipt_id, ''::text, contribution.proposal_ref,
           assessment.assessment_id, assessment.semantic_assessment,
           assessment.result_ref, assessment.result_digest,
           assessment.item_id, assessment.execution_item_id,
           assessment.collection_receipt_id, assessment.direct_verification,
           assessment.contract_ref, assessment.contract_digest
      FROM audit_finding_assessments AS assessment
      JOIN audit_finding_contributions AS contribution
        ON contribution.finding_id = assessment.finding_id
       AND contribution.receipt_id = assessment.receipt_id
     WHERE assessment.audit_id = $1 AND assessment.finding_id = $2
)
SELECT created_at, record_id, kind, receipt_id, relation, proposal_ref,
       assessment_id, semantic_assessment, result_ref, result_digest,
       item_id, execution_item_id, collection_receipt_id, direct_verification,
       contract_ref, contract_digest
  FROM records
 WHERE ($3::timestamptz IS NULL OR (created_at, record_id) > ($3, $4))
 ORDER BY created_at, record_id LIMIT $5`, params.AuditID, params.FindingID,
		params.AfterCreatedAt, params.AfterRecordID, params.Limit)
	if err != nil {
		return nil, fmt.Errorf("list finding provenance: %w", err)
	}
	defer rows.Close()
	result := make([]FindingProvenance, 0, params.Limit)
	for rows.Next() {
		var value FindingProvenance
		var kind string
		var proposalJSON, resultJSON, contractJSON []byte
		var assessmentID, semantic, resultDigest, itemID, executionItemID, collectionID, contractDigest *string
		var direct bool
		if err := rows.Scan(
			&value.CreatedAt, &value.RecordID, &kind, &value.ReceiptID, &value.Relation,
			&proposalJSON, &assessmentID, &semantic, &resultJSON, &resultDigest,
			&itemID, &executionItemID, &collectionID, &direct, &contractJSON, &contractDigest,
		); err != nil {
			return nil, fmt.Errorf("scan finding provenance: %w", err)
		}
		value.Kind = FindingProvenanceKind(kind)
		if json.Unmarshal(proposalJSON, &value.Proposal) != nil || value.Proposal.Ref.ValidateExact() != nil {
			return nil, errors.New("stored finding proposal provenance is invalid")
		}
		receipt, err := s.findings.GetAuditReceipt(ctx, params.OwnerID, params.AuditID, value.ReceiptID)
		if err != nil {
			return nil, err
		}
		value.Origin = receipt.Origin
		if assessmentID != nil {
			assessment := &FindingAssessment{
				AssessmentID: *assessmentID, SemanticAssessment: *semantic,
				ReceiptID: value.ReceiptID, ItemID: itemID, ExecutionItemID: executionItemID,
				CollectionReceiptID: collectionID, DirectVerification: direct,
				AcceptedAt: value.CreatedAt,
			}
			if resultDigest == nil || json.Unmarshal(resultJSON, &assessment.Result.Ref) != nil {
				return nil, errors.New("stored finding assessment provenance is invalid")
			}
			assessment.Result.Digest = *resultDigest
			if contractJSON != nil {
				if contractDigest == nil {
					return nil, errors.New("stored direct verification contract is invalid")
				}
				assessment.Contract = &auditstore.ExactArtifact{Digest: *contractDigest}
				if json.Unmarshal(contractJSON, &assessment.Contract.Ref) != nil {
					return nil, errors.New("stored direct verification contract is invalid")
				}
			}
			value.Assessment = assessment
			value.SupportsCurrent = finding.CurrentAssessment != nil &&
				finding.CurrentAssessment.AssessmentID == assessment.AssessmentID
		}
		result = append(result, value)
	}
	return result, rows.Err()
}

func (s *Service) hydrateFinding(
	ctx context.Context, ownerID string, row findingRow,
) (Finding, error) {
	receipt, err := s.findings.GetAuditReceipt(ctx, ownerID, row.auditID, row.firstReceiptID)
	if err != nil {
		return Finding{}, err
	}
	result := Finding{
		FindingID: row.findingID, AuditID: row.auditID, State: row.state,
		RejectionReason: row.rejectionReason, DuplicateTargetID: row.duplicateTargetID,
		FirstProposal: receipt, Revision: row.revision,
		CreatedAt: row.createdAt, UpdatedAt: row.updatedAt,
	}
	if row.currentDecisionID != nil {
		decision, err := readDecision(s.pool.QueryRow(ctx, decisionSelect+` WHERE decision.decision_id = $1`, *row.currentDecisionID))
		if err != nil {
			return Finding{}, err
		}
		result.AnalystDecision = &decision
		verdict := decision.Verdict
		result.AnalystVerdict = &verdict
		result.AnalystSeverity = decision.Severity
	}
	if row.currentAssessmentID != nil {
		assessment, err := readAssessment(s.pool.QueryRow(ctx, assessmentSelect+` WHERE assessment.assessment_id = $1`, *row.currentAssessmentID))
		if err != nil {
			return Finding{}, err
		}
		result.CurrentAssessment = &assessment
	}
	return result, nil
}

func lockFinding(
	ctx context.Context, tx pgx.Tx, ownerID, auditID, findingID string,
) (findingRow, auditstore.AuditState, error) {
	var state auditstore.AuditState
	var row findingRow
	var revision int64
	err := tx.QueryRow(ctx, `
SELECT audit.state, `+findingRowColumns+`
  FROM audit_findings AS finding
  JOIN audits AS audit USING (audit_id)
 WHERE audit.owner_id = $1 AND finding.audit_id = $2 AND finding.finding_id = $3
 FOR UPDATE OF audit, finding`, ownerID, auditID, findingID).Scan(
		&state, &row.findingID, &row.auditID, &row.firstReceiptID, &row.state,
		&row.rejectionReason, &row.duplicateTargetID, &row.currentAssessmentID,
		&row.currentDecisionID, &revision, &row.createdAt, &row.updatedAt,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return findingRow{}, "", auditstore.ErrNotFound
	}
	if err != nil {
		return findingRow{}, "", err
	}
	row.revision = uint64(revision)
	return row, state, nil
}

func scanFindingRow(row pgx.Row) (findingRow, error) {
	var result findingRow
	var revision int64
	err := row.Scan(
		&result.findingID, &result.auditID, &result.firstReceiptID, &result.state,
		&result.rejectionReason, &result.duplicateTargetID, &result.currentAssessmentID,
		&result.currentDecisionID, &revision, &result.createdAt, &result.updatedAt,
	)
	if err != nil {
		return findingRow{}, err
	}
	if revision < 1 || !result.state.Valid() {
		return findingRow{}, errors.New("stored Audit finding is invalid")
	}
	result.revision = uint64(revision)
	return result, nil
}

const reviewSelect = `
SELECT request.request_id, request.audit_id, request.finding_id, request.kind,
       request.subject_revision, request.subject_digest, request.requested_actions,
       request.state, request.expires_at, request.revision,
       request.created_at, request.updated_at,
       decision.decision_id, decision.actor_id, decision.verdict, decision.severity,
       decision.rationale, decision.duplicate_target_id, decision.created_at
  FROM audit_review_requests AS request
  JOIN audits AS audit USING (audit_id)
  LEFT JOIN audit_review_decisions AS decision
    ON decision.request_id = request.request_id
   AND decision.audit_id = request.audit_id
   AND decision.finding_id = request.finding_id`

func scanReviewRequest(row pgx.Row) (ReviewRequest, error) {
	var result ReviewRequest
	var actionsJSON []byte
	var revision int64
	var decisionID, actor, verdict, severity, rationale, duplicate *string
	var decisionAt *time.Time
	err := row.Scan(
		&result.RequestID, &result.AuditID, &result.FindingID, &result.Kind,
		&result.SubjectRevision, &result.SubjectDigest, &actionsJSON,
		&result.State, &result.ExpiresAt, &revision, &result.CreatedAt, &result.UpdatedAt,
		&decisionID, &actor, &verdict, &severity, &rationale, &duplicate, &decisionAt,
	)
	if err != nil {
		return ReviewRequest{}, err
	}
	if revision < 1 || !result.State.Valid() || json.Unmarshal(actionsJSON, &result.RequestedActions) != nil {
		return ReviewRequest{}, errors.New("stored Audit review request is invalid")
	}
	result.Revision = uint64(revision)
	if decisionID != nil {
		if actor == nil || verdict == nil || rationale == nil || decisionAt == nil {
			return ReviewRequest{}, errors.New("stored Audit review decision is incomplete")
		}
		decision := ReviewDecision{
			DecisionID: *decisionID, RequestID: result.RequestID, AuditID: result.AuditID,
			FindingID: result.FindingID, ActorID: *actor, Verdict: AnalystVerdict(*verdict),
			Rationale: *rationale, DuplicateTargetID: duplicate,
			SubjectRevision: result.SubjectRevision, SubjectDigest: result.SubjectDigest,
			CreatedAt: *decisionAt,
		}
		if severity != nil {
			value := FindingSeverity(*severity)
			decision.Severity = &value
		}
		result.Decision = &decision
	}
	return result, nil
}

const decisionSelect = `
SELECT decision.decision_id, decision.request_id, decision.audit_id,
       decision.finding_id, decision.actor_id, decision.verdict,
       decision.severity, decision.rationale, decision.duplicate_target_id,
       decision.subject_revision, decision.subject_digest, decision.created_at
  FROM audit_review_decisions AS decision`

func readDecision(row pgx.Row) (ReviewDecision, error) {
	var result ReviewDecision
	var severity *string
	err := row.Scan(&result.DecisionID, &result.RequestID, &result.AuditID,
		&result.FindingID, &result.ActorID, &result.Verdict, &severity,
		&result.Rationale, &result.DuplicateTargetID, &result.SubjectRevision,
		&result.SubjectDigest, &result.CreatedAt)
	if err != nil {
		return ReviewDecision{}, err
	}
	if severity != nil {
		value := FindingSeverity(*severity)
		result.Severity = &value
	}
	return result, nil
}

const assessmentSelect = `
SELECT assessment.assessment_id, assessment.semantic_assessment,
       assessment.result_ref, assessment.result_digest, assessment.receipt_id,
       assessment.item_id, assessment.execution_item_id,
       assessment.collection_receipt_id, assessment.direct_verification,
       assessment.contract_ref, assessment.contract_digest, assessment.accepted_at
  FROM audit_finding_assessments AS assessment`

func readAssessment(row pgx.Row) (FindingAssessment, error) {
	var result FindingAssessment
	var resultRef, contractRef []byte
	var contractDigest *string
	err := row.Scan(&result.AssessmentID, &result.SemanticAssessment,
		&resultRef, &result.Result.Digest, &result.ReceiptID,
		&result.ItemID, &result.ExecutionItemID, &result.CollectionReceiptID,
		&result.DirectVerification, &contractRef, &contractDigest, &result.AcceptedAt)
	if err != nil {
		return FindingAssessment{}, err
	}
	if json.Unmarshal(resultRef, &result.Result.Ref) != nil || result.Result.Ref.ValidateExact() != nil {
		return FindingAssessment{}, errors.New("stored finding assessment is invalid")
	}
	if contractRef != nil {
		if contractDigest == nil {
			return FindingAssessment{}, errors.New("stored finding assessment contract is invalid")
		}
		result.Contract = &auditstore.ExactArtifact{Digest: *contractDigest}
		if json.Unmarshal(contractRef, &result.Contract.Ref) != nil {
			return FindingAssessment{}, errors.New("stored finding assessment contract is invalid")
		}
	}
	return result, nil
}

func findingSubjectDigest(row findingRow) string {
	value := struct {
		Schema              string  `json:"schema"`
		FindingID           string  `json:"findingId"`
		AuditID             string  `json:"auditId"`
		Revision            uint64  `json:"revision"`
		State               string  `json:"state"`
		FirstReceiptID      string  `json:"firstReceiptId"`
		CurrentAssessmentID *string `json:"currentAssessmentId"`
		CurrentDecisionID   *string `json:"currentDecisionId"`
		DuplicateTargetID   *string `json:"duplicateTargetId"`
	}{
		Schema: "contractor.audit.finding-subject.v1", FindingID: row.findingID,
		AuditID: row.auditID, Revision: row.revision, State: string(row.state),
		FirstReceiptID: row.firstReceiptID, CurrentAssessmentID: row.currentAssessmentID,
		CurrentDecisionID: row.currentDecisionID, DuplicateTargetID: row.duplicateTargetID,
	}
	encoded, _ := json.Marshal(value)
	digest := sha256.Sum256(encoded)
	return "sha256:" + hex.EncodeToString(digest[:])
}

func decisionProjection(
	verdict AnalystVerdict, decisionID string, target *string,
) (string, *string, *string, *string) {
	switch verdict {
	case VerdictTruePositive:
		return string(FindingConfirmed), nil, nil, &decisionID
	case VerdictFalsePositive:
		reason := "false-positive"
		return string(FindingRejected), &reason, nil, &decisionID
	case VerdictDuplicate:
		return string(FindingDuplicate), nil, target, nil
	case VerdictNeedsEvidence:
		return string(FindingNeedsEvidence), nil, nil, nil
	default:
		return string(FindingProposed), nil, nil, nil
	}
}

func validateDuplicateTarget(
	ctx context.Context, tx pgx.Tx, auditID, findingID, targetID string,
) error {
	if targetID == findingID || !validReviewIdentity(targetID, 256) {
		return auditstore.ErrInvalid
	}
	var exists, cycle bool
	err := tx.QueryRow(ctx, `
WITH RECURSIVE chain(finding_id, duplicate_target_id) AS (
    SELECT finding_id, duplicate_target_id
      FROM audit_findings WHERE audit_id = $1 AND finding_id = $2
    UNION
    SELECT next.finding_id, next.duplicate_target_id
      FROM chain JOIN audit_findings AS next
        ON next.audit_id = $1 AND next.finding_id = chain.duplicate_target_id
     WHERE chain.duplicate_target_id IS NOT NULL
)
SELECT EXISTS (SELECT 1 FROM audit_findings WHERE audit_id = $1 AND finding_id = $2),
       EXISTS (SELECT 1 FROM chain WHERE finding_id = $3)`, auditID, targetID, findingID).Scan(&exists, &cycle)
	if err != nil {
		return err
	}
	if !exists {
		return auditstore.ErrNotFound
	}
	if cycle {
		return auditstore.ErrConflict
	}
	return nil
}

func appendAuditReviewEvent(
	ctx context.Context, tx pgx.Tx, auditID, kind, entityID string,
	entityRevision *uint64, summary map[string]any,
) error {
	var sequence int64
	if err := tx.QueryRow(ctx, `
UPDATE audits
   SET revision = revision + 1, next_event_sequence = next_event_sequence + 1,
       updated_at = GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
 WHERE audit_id = $1
RETURNING next_event_sequence - 1`, auditID).Scan(&sequence); err != nil {
		return err
	}
	encoded, _ := json.Marshal(summary)
	var revision any
	if entityRevision != nil {
		revision = int64(*entityRevision)
	}
	_, err := tx.Exec(ctx, `
INSERT INTO audit_events (
    audit_id, sequence_number, kind, entity_id, entity_revision, summary
) VALUES ($1, $2, $3, $4, $5, $6)`, auditID, sequence, kind, entityID, revision, encoded)
	return err
}

func validateFindingList(params FindingListParams) error {
	if !validReviewIdentity(params.OwnerID, 256) || !validReviewIdentity(params.AuditID, 256) ||
		params.Limit < 1 || params.Limit > MaxFindingPageSize ||
		(params.AfterCreatedAt == nil) != (params.AfterFindingID == "") ||
		(params.State != nil && !params.State.Valid()) ||
		(params.Severity != nil && !params.Severity.Valid()) {
		return auditstore.ErrInvalid
	}
	if params.Verdict != nil && *params.Verdict != VerdictTruePositive && *params.Verdict != VerdictFalsePositive {
		return auditstore.ErrInvalid
	}
	if params.Unreviewed && (params.Verdict != nil || params.Severity != nil) {
		return auditstore.ErrInvalid
	}
	return nil
}

func validateReviewList(params ReviewListParams) error {
	if !validReviewIdentity(params.OwnerID, 256) || !validReviewIdentity(params.AuditID, 256) ||
		params.Limit < 1 || params.Limit > MaxFindingPageSize ||
		(params.AfterCreatedAt == nil) != (params.AfterRequestID == "") ||
		(params.State != nil && !params.State.Valid()) ||
		(params.FindingID != nil && !validReviewIdentity(*params.FindingID, 256)) {
		return auditstore.ErrInvalid
	}
	return nil
}

func validateProvenanceList(params ProvenanceListParams) error {
	if !validReviewIdentity(params.OwnerID, 256) || !validReviewIdentity(params.AuditID, 256) ||
		!validReviewIdentity(params.FindingID, 256) || params.Limit < 1 ||
		params.Limit > MaxFindingPageSize ||
		(params.AfterCreatedAt == nil) != (params.AfterRecordID == "") {
		return auditstore.ErrInvalid
	}
	return nil
}

func validateCreateFindingReview(params CreateFindingReviewParams, now time.Time) error {
	if !validReviewIdentity(params.OwnerID, 256) || !validReviewIdentity(params.AuditID, 256) ||
		!validReviewIdentity(params.FindingID, 256) || !validReviewIdentity(params.RequestID, 256) ||
		params.ExpectedRevision < 1 || !validIdempotencyKey(params.IdempotencyKey) ||
		!validDigest(params.RequestDigest) {
		return auditstore.ErrInvalid
	}
	if params.ExpiresAt != nil && (!params.ExpiresAt.After(now) || params.ExpiresAt.After(now.Add(maximumReviewTTL))) {
		return auditstore.ErrInvalid
	}
	return nil
}

func validateFindingDecision(params DecideFindingParams) error {
	if !validReviewIdentity(params.OwnerID, 256) || !validReviewIdentity(params.AuditID, 256) ||
		!validReviewIdentity(params.RequestID, 256) || !validReviewIdentity(params.DecisionID, 256) ||
		params.ExpectedRequestRevision < 1 || !params.Verdict.Valid() ||
		!validIdempotencyKey(params.IdempotencyKey) || !validDigest(params.RequestDigest) ||
		!validRationale(params.Rationale) {
		return auditstore.ErrInvalid
	}
	if params.Verdict == VerdictTruePositive {
		if params.Severity == nil || !params.Severity.Valid() {
			return auditstore.ErrInvalid
		}
	} else if params.Severity != nil {
		return auditstore.ErrInvalid
	}
	if (params.Verdict == VerdictDuplicate) != (params.DuplicateTargetID != nil) {
		return auditstore.ErrInvalid
	}
	return nil
}

func validReviewIdentity(value string, maximum int) bool {
	return strings.TrimSpace(value) != "" && utf8.ValidString(value) &&
		!strings.ContainsRune(value, 0) && len([]byte(value)) <= maximum
}

func validRationale(value string) bool {
	return strings.TrimSpace(value) != "" && utf8.ValidString(value) &&
		!strings.ContainsRune(value, 0) && len([]byte(value)) <= MaxReviewRationale
}

func validIdempotencyKey(value string) bool {
	if value == "" || len(value) > 128 {
		return false
	}
	for index, character := range value {
		if (character >= 'A' && character <= 'Z') || (character >= 'a' && character <= 'z') ||
			(character >= '0' && character <= '9') || (index > 0 && strings.ContainsRune("._:-", character)) {
			continue
		}
		return false
	}
	return true
}

func sortProvenance(values []FindingProvenance) {
	sort.Slice(values, func(i, j int) bool {
		if values[i].CreatedAt.Equal(values[j].CreatedAt) {
			return values[i].RecordID < values[j].RecordID
		}
		return values[i].CreatedAt.Before(values[j].CreatedAt)
	})
}
