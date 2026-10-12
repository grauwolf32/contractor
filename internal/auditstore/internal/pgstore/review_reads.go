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

// GetReview reads one review request of an owner's Audit with its decision.
func (s *QueryStore) GetReview(
	ctx context.Context, ownerID, auditID, requestID string,
) (auditstore.ReviewRequest, error) {
	if !auditstore.ValidReviewIdentity(ownerID, 256) || !auditstore.ValidReviewIdentity(auditID, 256) ||
		!auditstore.ValidReviewIdentity(requestID, 256) {
		return auditstore.ReviewRequest{}, auditstore.ErrInvalid
	}
	request, err := scanReviewRequest(s.db.QueryRow(ctx, reviewSelect+`
 WHERE audit.owner_id = $1 AND request.audit_id = $2 AND request.request_id = $3`,
		ownerID, auditID, requestID))
	if errors.Is(err, pgx.ErrNoRows) {
		return auditstore.ReviewRequest{}, auditstore.ErrNotFound
	}
	if err != nil {
		return auditstore.ReviewRequest{}, err
	}
	return request, nil
}

// ListReviews reads one keyset page of the review requests of an owner's
// Audit in creation order. An empty page of a missing or foreign Audit is
// auditstore.ErrNotFound.
func (s *QueryStore) ListReviews(
	ctx context.Context, params auditstore.ReviewListParams,
) ([]auditstore.ReviewRequest, error) {
	if err := validateReviewList(params); err != nil {
		return nil, err
	}
	var state *string
	if params.State != nil {
		value := string(*params.State)
		state = &value
	}
	rows, err := s.db.Query(ctx, listReviewsSQL,
		params.OwnerID, params.AuditID, params.FindingID, state,
		params.AfterCreatedAt, params.AfterRequestID, params.Limit)
	if err != nil {
		return nil, fmt.Errorf("list Audit reviews: %w", err)
	}
	defer rows.Close()
	result := make([]auditstore.ReviewRequest, 0, params.Limit)
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
	rows.Close()
	if len(result) == 0 {
		if _, err := s.get(ctx, params.OwnerID, params.AuditID); err != nil {
			return nil, err
		}
	}
	return result, nil
}

// ListOwnerReviews lists the owner's review requests newest first. The
// pending filter lists only requests that can still be decided: a stored
// pending request past its expiry at the PostgreSQL clock is left out before
// anything has expired it, as in the workspace's pending count. The decided
// and expired filters and the unfiltered list use the stored state.
func (s *QueryStore) ListOwnerReviews(
	ctx context.Context, params auditstore.OwnerReviewListParams,
) ([]auditstore.ReviewRequest, error) {
	if err := validateOwnerPage(params.OwnerID, params.Limit, params.BeforeCreatedAt, params.BeforeRequestID, params.AuditStates); err != nil {
		return nil, err
	}
	if params.State != nil && !params.State.Valid() {
		return nil, auditstore.ErrInvalid
	}
	rows, err := s.db.Query(ctx, listOwnerReviewsSQL,
		params.OwnerID, params.State, params.BeforeCreatedAt, params.BeforeRequestID, params.Limit, params.AuditStates)
	if err != nil {
		return nil, fmt.Errorf("list owner reviews: %w", err)
	}
	defer rows.Close()
	result := make([]auditstore.ReviewRequest, 0, params.Limit)
	for rows.Next() {
		request, err := scanReviewRequest(rows)
		if err != nil {
			return nil, err
		}
		result = append(result, request)
	}
	return result, rows.Err()
}

// ReviewPageBasis reads the revision of an owner's Audit, the statement time
// and the number of the Audit's review requests that match the page's
// filters, in one statement.
func (s *QueryStore) ReviewPageBasis(
	ctx context.Context, params auditstore.ReviewListParams,
) (auditstore.PageBasis, error) {
	var result auditstore.PageBasis
	if err := validateReviewList(params); err != nil {
		return result, err
	}
	err := s.db.QueryRow(ctx, reviewPageBasisSQL, params.OwnerID, params.AuditID, params.FindingID, params.State).
		Scan(&result.AuditRevision, &result.AsOf, &result.Total)
	if errors.Is(err, pgx.ErrNoRows) {
		return result, auditstore.ErrNotFound
	}
	return result, err
}

func scanReviewRequest(row scanner) (auditstore.ReviewRequest, error) {
	var result auditstore.ReviewRequest
	var actionsJSON []byte
	var revision int64
	var findingID *string
	var decisionID, actor, action, verdict, severity, rationale, duplicate *string
	var decisionAt *time.Time
	err := row.Scan(
		&result.RequestID, &result.AuditID, &findingID,
		&result.SubjectKind, &result.SubjectID, &result.Kind,
		&result.SubjectRevision, &result.SubjectDigest, &actionsJSON,
		&result.State, &result.ExpiresAt, &revision, &result.CreatedAt, &result.UpdatedAt,
		&decisionID, &actor, &action, &verdict, &severity, &rationale, &duplicate, &decisionAt,
	)
	if err != nil {
		return auditstore.ReviewRequest{}, err
	}
	if revision < 1 || !result.State.Valid() || !result.SubjectKind.Valid() || result.SubjectID == "" ||
		(result.SubjectKind == auditstore.ReviewSubjectFinding) != (findingID != nil) ||
		json.Unmarshal(actionsJSON, &result.RequestedActions) != nil {
		return auditstore.ReviewRequest{}, errors.New("stored Audit review request is invalid")
	}
	if len(result.RequestedActions) == 0 {
		return auditstore.ReviewRequest{}, errors.New("stored Audit review request has no actions")
	}
	for _, action := range result.RequestedActions {
		if !action.Valid() {
			return auditstore.ReviewRequest{}, errors.New("stored Audit review request action is invalid")
		}
	}
	result.Revision = uint64(revision)
	if findingID != nil {
		result.FindingID = *findingID
	}
	if decisionID != nil {
		if actor == nil || action == nil || rationale == nil || decisionAt == nil {
			return auditstore.ReviewRequest{}, errors.New("stored Audit review decision is incomplete")
		}
		decision := auditstore.ReviewDecision{
			DecisionID: *decisionID, RequestID: result.RequestID, AuditID: result.AuditID,
			FindingID: result.FindingID, ActorID: *actor,
			Rationale: *rationale, DuplicateTargetID: duplicate,
			SubjectRevision: result.SubjectRevision, SubjectDigest: result.SubjectDigest,
			CreatedAt: *decisionAt,
		}
		if verdict != nil {
			decision.Verdict = auditstore.AnalystVerdict(*verdict)
		}
		if findingID == nil {
			decision.Action = auditstore.ReviewAction(*action)
			if !decision.Action.Valid() || decision.Verdict != "" {
				return auditstore.ReviewRequest{}, errors.New("stored Audit action review decision is invalid")
			}
		} else if !decision.Verdict.Valid() {
			return auditstore.ReviewRequest{}, errors.New("stored Audit finding review decision is invalid")
		}
		if severity != nil {
			value := auditstore.FindingSeverity(*severity)
			decision.Severity = &value
		}
		result.Decision = &decision
	}
	return result, nil
}
