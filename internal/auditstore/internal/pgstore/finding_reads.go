package pgstore

import (
	"context"
	"errors"
	"fmt"
	"time"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/pagelimit"
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
	rows, err := s.db.Query(ctx, findingPageSQL(listFindingsSQL, params.State, params.Verdict, params.Unreviewed, params.Severity), append(findingFilterArgs(params),
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

// ListOwnerFindings examines one bounded window of canonical Findings in the
// owner's Projects, newest first. Filtered-out candidates still advance the cursor.
func (s *QueryStore) ListOwnerFindings(
	ctx context.Context, params auditstore.OwnerFindingListParams,
) (auditstore.FindingWindow, error) {
	if err := validateOwnerPage(params.OwnerID, params.Limit, params.BeforeCreatedAt, params.BeforeFindingID, nil); err != nil || params.Limit < 2 {
		if err == nil {
			err = auditstore.ErrInvalid
		}
		return auditstore.FindingWindow{}, err
	}
	if (params.State != nil && !params.State.Valid()) || (params.Severity != nil && !params.Severity.Valid()) ||
		(params.Verdict != nil && *params.Verdict != auditstore.VerdictTruePositive && *params.Verdict != auditstore.VerdictFalsePositive) ||
		(params.Unreviewed && (params.Verdict != nil || params.Severity != nil)) {
		return auditstore.FindingWindow{}, auditstore.ErrInvalid
	}
	query := listOwnerFindingsSQL
	var project any
	if params.ProjectID != "" {
		if !contracts.ValidResourceID(params.ProjectID) {
			return auditstore.FindingWindow{}, auditstore.ErrInvalid
		}
		query = listProjectFindingsSQL
		project = params.ProjectID
	}
	rows, err := s.db.Query(ctx, query,
		params.OwnerID, project, params.State, params.Verdict, params.Unreviewed, params.Severity,
		params.BeforeCreatedAt, params.BeforeFindingID, params.Limit)
	if err != nil {
		return auditstore.FindingWindow{}, fmt.Errorf("list owner findings: %w", err)
	}
	defer rows.Close()
	result := auditstore.FindingWindow{Items: make([]auditstore.ProjectFindingSummary, 0, params.Limit)}
	examined := 0
	var previous *auditstore.ProjectFindingSummary
	for rows.Next() {
		var visible bool
		row, err := scanProjectFindingSummary(rows, &visible)
		if err != nil {
			return auditstore.FindingWindow{}, err
		}
		examined++
		if examined == params.Limit {
			result.ContinueBefore = previous
			continue
		}
		previous = &row
		if visible && row.Matches(params.State, params.Verdict, params.Unreviewed, params.Severity) {
			result.Items = append(result.Items, row)
		}
	}
	return result, rows.Err()
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

func scanProjectFindingSummary(row pgx.Row, extra ...any) (auditstore.ProjectFindingSummary, error) {
	var value auditstore.ProjectFindingSummary
	destinations := []any{&value.FindingID, &value.ProjectID, &value.Title, &value.TitleTruncated,
		&value.SubjectKind, &value.SubjectKey, &value.SubjectTruncated, &value.SeveritySuggestion,
		&value.State, &value.AnalystVerdict, &value.AnalystSeverity, &value.DuplicateTargetID, &value.Revision, &value.CreatedAt, &value.UpdatedAt}
	if err := row.Scan(append(destinations, extra...)...); err != nil {
		return value, err
	}
	if !contracts.ValidResourceID(value.FindingID) || !contracts.ValidResourceID(value.ProjectID) || value.Revision < 1 || value.Revision > contracts.MaxExactInteger ||
		!utf8.ValidString(value.SubjectKind) || len(value.SubjectKind) > 256 || utf8.RuneCountInString(value.SubjectKind) > 64 ||
		!utf8.ValidString(value.SubjectKey) || len(value.SubjectKey) > 1024 || utf8.RuneCountInString(value.SubjectKey) > 256 ||
		(value.SeveritySuggestion != "" && !auditstore.FindingSeverity(value.SeveritySuggestion).Valid()) ||
		!utf8.ValidString(value.Title) || len(value.Title) > 1024 || utf8.RuneCountInString(value.Title) > 256 ||
		(value.State != nil && !value.State.Valid()) || (value.AnalystVerdict != nil && !value.AnalystVerdict.Valid()) ||
		(value.AnalystSeverity != nil && !value.AnalystSeverity.Valid()) || (value.State == nil) != (value.AnalystVerdict == nil) {
		return value, errors.New("stored Project Finding summary is invalid")
	}
	if value.DuplicateTargetID != nil && (!contracts.ValidResourceID(*value.DuplicateTargetID) || *value.DuplicateTargetID == value.FindingID) {
		return value, errors.New("stored Finding duplicate target is invalid")
	}
	if (auditstore.ProjectFinding{State: value.State, AnalystVerdict: value.AnalystVerdict, AnalystSeverity: value.AnalystSeverity, DuplicateTargetID: value.DuplicateTargetID}).AssessmentShape() != nil {
		return value, errors.New("stored Project Finding summary assessment is invalid")
	}
	return value, nil
}
