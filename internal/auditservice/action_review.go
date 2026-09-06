package auditservice

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// DecideActionReview applies non-finding human authority through the same
// immutable review ledger as finding triage. The exact subject is revalidated
// while both the request and its target are locked; execution intent creation
// performs an independent final authorization check.
func (s *Service) DecideActionReview(
	ctx context.Context, params DecideActionReviewParams,
) (ActionReviewDecisionResult, error) {
	if err := validateActionDecision(params); err != nil {
		return ActionReviewDecisionResult{}, err
	}
	var decisionID, requestID string
	var replayed, expired bool
	err := persistencepostgres.InTx(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if _, err := tx.Exec(ctx, `SELECT pg_advisory_xact_lock(hashtextextended($1, 0))`,
			"audit-action-review:"+params.AuditID); err != nil {
			return err
		}
		var storedDigest string
		err := tx.QueryRow(ctx, `
SELECT decision_id, request_id, request_digest
  FROM audit_review_decisions
 WHERE audit_id = $1 AND idempotency_key = $2`,
			params.AuditID, params.IdempotencyKey,
		).Scan(&decisionID, &requestID, &storedDigest)
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
		var subjectKind ReviewSubjectKind
		var subjectID, kind, subjectDigest string
		var subjectRevision, requestRevision int64
		var requestState ReviewState
		var expiresAt *time.Time
		var requestedJSON []byte
		err = tx.QueryRow(ctx, `
SELECT audit.state, request.subject_kind, request.subject_id, request.kind,
       request.subject_revision, request.subject_digest,
       request.requested_actions, request.state, request.expires_at,
       request.revision
  FROM audit_review_requests AS request
  JOIN audits AS audit USING (audit_id)
 WHERE audit.owner_id = $1 AND request.audit_id = $2
   AND request.request_id = $3
 FOR UPDATE OF audit, request`,
			params.OwnerID, params.AuditID, params.RequestID,
		).Scan(&auditState, &subjectKind, &subjectID, &kind, &subjectRevision,
			&subjectDigest, &requestedJSON, &requestState, &expiresAt, &requestRevision)
		if errors.Is(err, pgx.ErrNoRows) {
			return auditstore.ErrNotFound
		}
		if err != nil {
			return err
		}
		if requestState != ReviewPending || requestRevision != int64(params.ExpectedRequestRevision) ||
			subjectRevision < 1 || auditState == auditstore.AuditDeleting ||
			auditState == auditstore.AuditFinalizing {
			return auditstore.ErrPrecondition
		}
		var requested []ReviewAction
		if json.Unmarshal(requestedJSON, &requested) != nil || !containsReviewAction(requested, params.Action) {
			return auditstore.ErrPrecondition
		}
		if expiresAt != nil && !s.now().UTC().Before(*expiresAt) {
			if _, err := tx.Exec(ctx, `
UPDATE audit_review_requests
   SET state = 'expired', revision = revision + 1,
       updated_at = GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
 WHERE request_id = $1 AND state = 'pending'`, params.RequestID); err != nil {
				return err
			}
			expired = true
			return appendAuditReviewEvent(ctx, tx, params.AuditID, "review.expired",
				params.RequestID, nil, map[string]any{
					"subjectKind": subjectKind,
					"subjectId":   subjectID,
					"kind":        kind,
				})
		}

		switch subjectKind {
		case ReviewSubjectItemAction:
			if auditState != auditstore.AuditActive && auditState != auditstore.AuditWaitingReview &&
				auditState != auditstore.AuditPaused {
				return auditstore.ErrPrecondition
			}
			if err := validateAndApplyItemDecision(
				ctx, tx, params.AuditID, subjectID, kind, subjectDigest, params.Action,
			); err != nil {
				return err
			}
		case ReviewSubjectReport:
			if auditState != auditstore.AuditWaitingReview || kind != ReportAcceptanceReviewKind {
				return auditstore.ErrPrecondition
			}
			if err := validateAndApplyReportDecision(
				ctx, tx, params.AuditID, params.RequestID, subjectID,
				subjectRevision, subjectDigest, params.Action,
			); err != nil {
				return err
			}
		default:
			return auditstore.ErrPrecondition
		}

		decisionID, requestID = params.DecisionID, params.RequestID
		if _, err := tx.Exec(ctx, `
INSERT INTO audit_review_decisions (
    decision_id, request_id, audit_id, finding_id, actor_id, action, verdict,
    severity, rationale, duplicate_target_id, subject_revision,
    subject_digest, idempotency_key, request_digest
) VALUES ($1, $2, $3, NULL, $4, $5, NULL, NULL, $6, NULL, $7, $8, $9, $10)`,
			decisionID, requestID, params.AuditID, params.OwnerID, params.Action,
			params.Rationale, subjectRevision, subjectDigest,
			params.IdempotencyKey, params.RequestDigest); err != nil {
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
		if subjectKind == ReviewSubjectItemAction {
			if _, err := tx.Exec(ctx, `
UPDATE audits AS audit
   SET state = CASE
           WHEN audit.state = 'waiting_review' AND NOT EXISTS (
               SELECT 1 FROM audit_items AS item
                WHERE item.audit_id = audit.audit_id
                  AND item.state = 'awaiting_review'
           ) THEN 'active'
           ELSE audit.state
       END
 WHERE audit.audit_id = $1`, params.AuditID); err != nil {
				return err
			}
		}
		return appendAuditReviewEvent(ctx, tx, params.AuditID, "review.decided", decisionID, nil,
			map[string]any{
				"subjectKind": subjectKind, "subjectId": subjectID,
				"kind": kind, "action": params.Action,
			})
	})
	if err != nil {
		return ActionReviewDecisionResult{}, err
	}
	if expired {
		return ActionReviewDecisionResult{}, auditstore.ErrPrecondition
	}
	request, err := s.GetReview(ctx, params.OwnerID, params.AuditID, requestID)
	if err != nil {
		return ActionReviewDecisionResult{}, err
	}
	if request.Decision == nil || request.Decision.DecisionID != decisionID {
		return ActionReviewDecisionResult{}, errors.New("stored Audit action decision is missing")
	}
	return ActionReviewDecisionResult{
		Request: request, Decision: *request.Decision, Replayed: replayed,
	}, nil
}

func validateAndApplyItemDecision(
	ctx context.Context, tx pgx.Tx, auditID, itemID, kind, digest string,
	action ReviewAction,
) error {
	var state auditstore.ItemState
	var approvalKind auditstore.ItemApprovalKind
	var approvalDigest *string
	err := tx.QueryRow(ctx, `
SELECT state, approval_kind, approval_subject_digest
  FROM audit_items
 WHERE audit_id = $1 AND item_id = $2
 FOR UPDATE`, auditID, itemID).Scan(&state, &approvalKind, &approvalDigest)
	if errors.Is(err, pgx.ErrNoRows) {
		return auditstore.ErrNotFound
	}
	if err != nil {
		return err
	}
	if state != auditstore.ItemAwaitingReview || approvalDigest == nil ||
		string(approvalKind) != kind || *approvalDigest != digest {
		return auditstore.ErrPrecondition
	}
	if action == ReviewApprove {
		tag, err := tx.Exec(ctx, `
UPDATE audit_items
   SET state = 'ready',
       updated_at = GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
 WHERE audit_id = $1 AND item_id = $2 AND state = 'awaiting_review'`, auditID, itemID)
		if err != nil {
			return err
		}
		if tag.RowsAffected() != 1 {
			return auditstore.ErrPrecondition
		}
		return nil
	}
	if action != ReviewReject {
		return auditstore.ErrInvalid
	}
	tag, err := tx.Exec(ctx, `
UPDATE audit_items
   SET state = 'settled', final_disposition = 'excluded',
       updated_at = GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
	WHERE audit_id = $1 AND item_id = $2 AND state = 'awaiting_review'`, auditID, itemID)
	if err != nil {
		return err
	}
	if tag.RowsAffected() != 1 {
		return auditstore.ErrPrecondition
	}
	_, err = tx.Exec(ctx, `
UPDATE audit_coverage_rows
   SET status = 'excluded',
       gaps = CASE WHEN gaps ? 'human-approval-rejected' THEN gaps
                   ELSE gaps || '["human-approval-rejected"]'::jsonb END,
       rationale = 'The exact proposed action was rejected by its owner.',
       updated_at = GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
 WHERE audit_id = $1 AND item_id = $2`, auditID, itemID)
	return err
}

func validateAndApplyReportDecision(
	ctx context.Context, tx pgx.Tx, auditID, requestID, subjectID string,
	subjectRevision int64, subjectDigest string, action ReviewAction,
) error {
	if subjectID != auditID {
		return auditstore.ErrPrecondition
	}
	var candidateRevision int64
	var candidateDigest string
	var machineJSON, summaryJSON []byte
	err := tx.QueryRow(ctx, `
SELECT subject_revision, subject_digest, machine_link, summary_link
  FROM audit_report_candidates
 WHERE audit_id = $1 AND request_id = $2
 FOR UPDATE`, auditID, requestID).Scan(
		&candidateRevision, &candidateDigest, &machineJSON, &summaryJSON,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return auditstore.ErrNotFound
	}
	if err != nil {
		return err
	}
	if candidateRevision != subjectRevision || candidateDigest != subjectDigest {
		return auditstore.ErrPrecondition
	}
	if action == ReviewReject {
		tag, err := tx.Exec(ctx, `
UPDATE audits
   SET state = 'failed', dispatch_state = 'closed', hold_state = 'released',
       stop_reason_code = 'report_rejected',
       stop_reason_message = 'The exact proposed Audit report was rejected by its owner.',
       finished_at = clock_timestamp()
 WHERE audit_id = $1 AND state = 'waiting_review'`, auditID)
		if err != nil {
			return err
		}
		if tag.RowsAffected() != 1 {
			return auditstore.ErrPrecondition
		}
		return nil
	}
	if action != ReviewApprove {
		return auditstore.ErrInvalid
	}
	var machine, summary auditstore.ArtifactLink
	if json.Unmarshal(machineJSON, &machine) != nil || json.Unmarshal(summaryJSON, &summary) != nil {
		return fmt.Errorf("stored Audit report candidate is invalid")
	}
	for _, link := range []auditstore.ArtifactLink{machine, summary} {
		refJSON, _ := json.Marshal(link.Artifact.Ref)
		if _, err := tx.Exec(ctx, `
INSERT INTO audit_artifact_links (
    audit_id, logical_key, artifact_ref, artifact_digest,
    media_type, size_bytes, source_provenance, display_ref
) VALUES ($1, $2, $3::jsonb, $4, $5, $6, $7::jsonb, $8)`,
			auditID, link.LogicalKey, refJSON, link.Artifact.Digest,
			link.Artifact.MediaType, link.Artifact.SizeBytes,
			link.SourceProvenance, link.DisplayRef); err != nil {
			return err
		}
	}
	tag, err := tx.Exec(ctx, `
UPDATE audits
   SET state = 'completed', dispatch_state = 'closed', hold_state = 'released',
       stop_reason_code = CASE WHEN stop_reason_code = 'round_complete' THEN NULL ELSE stop_reason_code END,
       stop_reason_message = CASE WHEN stop_reason_code = 'round_complete' THEN NULL ELSE stop_reason_message END,
       finished_at = clock_timestamp()
 WHERE audit_id = $1 AND state = 'waiting_review'`, auditID)
	if err != nil {
		return err
	}
	if tag.RowsAffected() != 1 {
		return auditstore.ErrPrecondition
	}
	return nil
}

func containsReviewAction(values []ReviewAction, target ReviewAction) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}

func validateActionDecision(params DecideActionReviewParams) error {
	if !validReviewIdentity(params.OwnerID, 256) || !validReviewIdentity(params.AuditID, 256) ||
		!validReviewIdentity(params.RequestID, 256) || !validReviewIdentity(params.DecisionID, 256) ||
		params.ExpectedRequestRevision == 0 || !params.Action.Valid() ||
		!validReviewIdentity(params.IdempotencyKey, 128) || !validDigest(params.RequestDigest) ||
		!validRationale(params.Rationale) {
		return auditstore.ErrInvalid
	}
	return nil
}
