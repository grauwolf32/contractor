package auditservice

import (
	"context"
	"fmt"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/pagelimit"
	"github.com/grauwolf32/contractor/internal/projectstore"
)

// visibleAuditProjectSQL keeps an owner-wide list to the Audits of Projects
// listed to their owner, by the rule the owner-wide Audit list applies.
var visibleAuditProjectSQL = `
 AND ` + projectstore.OwnerListedProjectSQL("audit.project_id", "audit.owner_id")

func validateOwnerPage(ownerID string, limit int, before *time.Time, id string, states []auditstore.AuditState) error {
	if !validReviewIdentity(ownerID, 256) || !pagelimit.ValidFetch(limit) ||
		(before == nil) != (id == "") ||
		(before != nil && (before.IsZero() || !validReviewIdentity(id, 256))) || len(states) > 10 {
		return auditstore.ErrInvalid
	}
	for _, state := range states {
		if !state.Valid() {
			return auditstore.ErrInvalid
		}
	}
	return nil
}

func (s *Service) ListOwnerFindings(ctx context.Context, params OwnerFindingListParams) ([]Finding, error) {
	if err := validateOwnerPage(params.OwnerID, params.Limit, params.BeforeCreatedAt, params.BeforeFindingID, params.AuditStates); err != nil {
		return nil, err
	}
	if (params.State != nil && !params.State.Valid()) || (params.Severity != nil && !params.Severity.Valid()) ||
		(params.Verdict != nil && *params.Verdict != VerdictTruePositive && *params.Verdict != VerdictFalsePositive) ||
		(params.Unreviewed && (params.Verdict != nil || params.Severity != nil)) {
		return nil, auditstore.ErrInvalid
	}
	rows, err := s.pool.Query(ctx, `SELECT `+findingRowColumns+findingFilterSQL+visibleAuditProjectSQL+`
   AND ($7::timestamptz IS NULL OR (finding.created_at, finding.finding_id) < ($7, $8))
   AND ($10::text[] IS NULL OR audit.state = ANY($10))
 ORDER BY finding.created_at DESC, finding.finding_id DESC LIMIT $9`,
		params.OwnerID, nil, params.State, params.Verdict, params.Unreviewed, params.Severity,
		params.BeforeCreatedAt, params.BeforeFindingID, params.Limit, params.AuditStates)
	if err != nil {
		return nil, fmt.Errorf("list owner findings: %w", err)
	}
	defer rows.Close()
	base := make([]findingRow, 0, params.Limit)
	for rows.Next() {
		row, err := scanFindingRow(rows)
		if err != nil {
			return nil, err
		}
		base = append(base, row)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	rows.Close()
	// Hydration reads every receipt through its finding's own Audit hold, for
	// all Audits of the page at once; no unowned receipt can enter the
	// projection.
	return s.hydrateFindings(ctx, params.OwnerID, base)
}

// ListOwnerReviews lists the owner's review requests newest first. The
// pending filter lists only requests that can still be decided: a stored
// pending request past its expiry at the PostgreSQL clock is left out before
// anything has expired it, as in the workspace's pending count. The decided
// and expired filters and the unfiltered list use the stored state.
func (s *Service) ListOwnerReviews(ctx context.Context, params OwnerReviewListParams) ([]ReviewRequest, error) {
	if err := validateOwnerPage(params.OwnerID, params.Limit, params.BeforeCreatedAt, params.BeforeRequestID, params.AuditStates); err != nil {
		return nil, err
	}
	if params.State != nil && !params.State.Valid() {
		return nil, auditstore.ErrInvalid
	}
	rows, err := s.pool.Query(ctx, reviewSelect+`
 WHERE audit.owner_id = $1`+visibleAuditProjectSQL+`
   AND ($2::text IS NULL OR request.state = $2)
   AND ($2::text IS DISTINCT FROM 'pending' OR `+reviewUnexpiredSQL+`)
   AND ($3::timestamptz IS NULL OR (request.created_at, request.request_id) < ($3, $4))
   AND ($6::text[] IS NULL OR audit.state = ANY($6))
 ORDER BY request.created_at DESC, request.request_id DESC LIMIT $5`,
		params.OwnerID, params.State, params.BeforeCreatedAt, params.BeforeRequestID, params.Limit, params.AuditStates)
	if err != nil {
		return nil, fmt.Errorf("list owner reviews: %w", err)
	}
	defer rows.Close()
	result := make([]ReviewRequest, 0, params.Limit)
	for rows.Next() {
		request, err := scanReviewRequest(rows)
		if err != nil {
			return nil, err
		}
		result = append(result, request)
	}
	return result, rows.Err()
}
