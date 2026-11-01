package pgstore

import (
	"context"
	"errors"
	"fmt"
	"time"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/pagelimit"
	"github.com/jackc/pgx/v5"
)

// ListFindings reads one keyset page of the findings of an owner's Audit in
// creation order. An empty page of a missing or foreign Audit is
// auditstore.ErrNotFound.
func (s *QueryStore) ListFindings(
	ctx context.Context, params auditstore.FindingListParams,
) ([]auditstore.ProjectFindingSummary, error) {
	if err := validateFindingList(params); err != nil {
		return nil, err
	}
	rows, err := s.db.Query(ctx, listFindingsSQL, append(findingFilterArgs(params),
		params.AfterCreatedAt, params.AfterFindingID, params.Limit)...)
	if err != nil {
		return nil, fmt.Errorf("list Audit findings: %w", err)
	}
	result, err := pgx.AppendRows(make([]auditstore.ProjectFindingSummary, 0, params.Limit), rows, func(row pgx.CollectableRow) (auditstore.ProjectFindingSummary, error) {
		return scanProjectFindingSummary(row)
	})
	if err != nil {
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

// findingFilterArgs binds the findingFilterSQL parameters $1 to $6 of a
// finding page, the count and the page alike.
func findingFilterArgs(params auditstore.FindingListParams) []any {
	return []any{params.OwnerID, params.AuditID, optionalText(params.State), optionalText(params.Verdict),
		params.Unreviewed, optionalText(params.Severity)}
}

// optionalText binds an optional enumerated value as text.
func optionalText[T ~string](value *T) *string {
	if value == nil {
		return nil
	}
	text := string(*value)
	return &text
}

// ListOwnerFindings reads one keyset page of every finding of an owner's
// Audits in listed Projects, newest first.
func (s *QueryStore) ListOwnerFindings(
	ctx context.Context, params auditstore.OwnerFindingListParams,
) ([]auditstore.ProjectFindingSummary, error) {
	if err := validateOwnerPage(params.OwnerID, params.Limit, params.BeforeCreatedAt, params.BeforeFindingID, params.AuditStates); err != nil {
		return nil, err
	}
	if (params.State != nil && !params.State.Valid()) || (params.Severity != nil && !params.Severity.Valid()) ||
		(params.Verdict != nil && *params.Verdict != auditstore.VerdictTruePositive && *params.Verdict != auditstore.VerdictFalsePositive) ||
		(params.Unreviewed && (params.Verdict != nil || params.Severity != nil)) {
		return nil, auditstore.ErrInvalid
	}
	query := listOwnerFindingsSQL
	var project any
	if params.ProjectID != "" {
		if !contracts.ValidResourceID(params.ProjectID) || len(params.AuditStates) != 0 {
			return nil, auditstore.ErrInvalid
		}
		query = listProjectFindingsSQL
		project = params.ProjectID
	}
	rows, err := s.db.Query(ctx, query,
		params.OwnerID, project, params.State, params.Verdict, params.Unreviewed, params.Severity,
		params.BeforeCreatedAt, params.BeforeFindingID, params.Limit, params.AuditStates)
	if err != nil {
		return nil, fmt.Errorf("list owner findings: %w", err)
	}
	defer rows.Close()
	result := make([]auditstore.ProjectFindingSummary, 0, params.Limit)
	for rows.Next() {
		row, err := scanProjectFindingSummary(rows)
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
) (auditstore.ProjectFinding, error) {
	if !auditstore.ValidReviewIdentity(ownerID, 256) || !auditstore.ValidReviewIdentity(auditID, 256) ||
		!auditstore.ValidReviewIdentity(findingID, 256) {
		return auditstore.ProjectFinding{}, auditstore.ErrInvalid
	}
	row, err := scanProjectFinding(s.db.QueryRow(ctx, getAuditProjectFindingSQL, ownerID, auditID, findingID))
	if errors.Is(err, pgx.ErrNoRows) {
		return auditstore.ProjectFinding{}, auditstore.ErrNotFound
	}
	if err != nil {
		return auditstore.ProjectFinding{}, fmt.Errorf("read Audit finding: %w", err)
	}
	return row, nil
}

// FindingPageBasis reads the revision of an owner's Audit, the statement
// time and the number of the Audit's findings that match the page's
// filters, in one statement.
func (s *QueryStore) FindingPageBasis(
	ctx context.Context, params auditstore.FindingListParams,
) (auditstore.FindingPageBasis, error) {
	var result auditstore.FindingPageBasis
	if err := validateFindingList(params); err != nil {
		return result, err
	}
	err := s.db.QueryRow(ctx, findingPageBasisSQL, params.OwnerID, params.AuditID).
		Scan(&result.AuditRevision, &result.ProjectRevision, &result.AsOf)
	if errors.Is(err, pgx.ErrNoRows) {
		return result, auditstore.ErrNotFound
	}
	return result, err
}

// ReadFindingDecisions reads analyst decisions by their (Audit, decision)
// pairs, each through its own Audit of the owner, keyed by decision.

// ReadFindingAssessments reads finding assessments by their (Audit,
// assessment) pairs, each through its own Audit of the owner, keyed by
// assessment.

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
		(params.State != nil && !params.State.Valid()) {
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

func scanProjectFindingSummary(row pgx.Row) (auditstore.ProjectFindingSummary, error) {
	var value auditstore.ProjectFindingSummary
	if err := row.Scan(&value.FindingID, &value.ProjectID, &value.Title, &value.TitleTruncated,
		&value.State, &value.AnalystVerdict, &value.AnalystSeverity, &value.DuplicateTargetID, &value.Revision, &value.CreatedAt, &value.UpdatedAt); err != nil {
		return value, err
	}
	if !contracts.ValidResourceID(value.FindingID) || !contracts.ValidResourceID(value.ProjectID) || value.Revision < 1 || value.Revision > contracts.MaxExactInteger ||
		!utf8.ValidString(value.Title) || len(value.Title) > 1024 || utf8.RuneCountInString(value.Title) > 256 ||
		(value.State != nil && !value.State.Valid()) || (value.AnalystVerdict != nil && !value.AnalystVerdict.Valid()) ||
		(value.AnalystSeverity != nil && !value.AnalystSeverity.Valid()) || (value.State == nil) != (value.AnalystVerdict == nil) {
		return value, errors.New("stored Project Finding summary is invalid")
	}
	return value, nil
}
