package auditservice

import (
	"context"
	"fmt"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
)

const visibleAuditProjectSQL = `
 AND EXISTS (SELECT 1 FROM projects AS project
   WHERE project.project_id = audit.project_id AND project.owner_id = audit.owner_id
     AND project.kind = 'project' AND project.lifecycle_state = 'active'
     AND NOT EXISTS (SELECT 1 FROM eval_submissions AS submission
       JOIN eval_experiments AS experiment USING (experiment_id)
       WHERE experiment.owner_id = project.owner_id
         AND submission.execution_project_id = project.project_id))`

func validateOwnerPage(ownerID string, limit int, before *time.Time, id string, states []auditstore.AuditState) error {
	if !validReviewIdentity(ownerID, 256) || limit < 1 || limit > maxFindingListRows ||
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
	byAudit := make(map[string][]findingRow)
	for rows.Next() {
		row, err := scanFindingRow(rows)
		if err != nil {
			return nil, err
		}
		base = append(base, row)
		byAudit[row.auditID] = append(byAudit[row.auditID], row)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	rows.Close()
	// Receipt reads enforce each Audit hold. Batch by Audit, then restore the
	// global keyset order; no unowned receipt can enter the projection.
	byID := make(map[string]Finding, len(base))
	for auditID, group := range byAudit {
		findings, err := s.hydrateFindings(ctx, params.OwnerID, auditID, group)
		if err != nil {
			return nil, err
		}
		for _, finding := range findings {
			byID[finding.FindingID] = finding
		}
	}
	result := make([]Finding, len(base))
	for i, row := range base {
		result[i] = byID[row.findingID]
	}
	return result, nil
}

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
