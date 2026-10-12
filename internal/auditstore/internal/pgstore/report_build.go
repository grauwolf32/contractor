package pgstore

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/jackc/pgx/v5"
)

// findingSection reports whether section selects findings: every valid
// report section except the items of the Rounds.
func findingSection(section auditstore.ReportSection) bool {
	return section.Valid() && section != auditstore.ReportSectionItems
}

// coverageCountsTotal is the number of items counts covers, and false when
// a count is negative.
func coverageCountsTotal(counts auditstore.ReportCoverageCounts) (int, bool) {
	total := 0
	for _, count := range []int{
		counts.NotTested, counts.Inconclusive, counts.Satisfied, counts.Violated, counts.NotApplicable,
		counts.Blocked, counts.Excluded, counts.TracedComplete, counts.TracedPartial, counts.Unmapped,
	} {
		if count < 0 {
			return 0, false
		}
		total += count
	}
	return total, true
}

// ValidateRecordReportPage enforces the page contract of RecordReportPage,
// so a store that skips PostgreSQL cannot accept a page it would reject.
func ValidateRecordReportPage(params auditstore.RecordReportPageParams) error {
	if err := validateClaimIdentity(params.Claim); err != nil {
		return err
	}
	page := params.Page
	if params.ExpectedAuditRevision == 0 || params.ExpectedAuditRevision > math.MaxInt64 ||
		page.Ordinal < 0 || page.Ordinal >= auditstore.MaxReportPages || !page.Section.Valid() ||
		page.EntryCount < 0 || page.EntryCount > auditstore.MaxReportPageRows ||
		(page.EntryCount == 0) != (len(params.Content) == 0) ||
		page.EntryCount == 0 && !page.Exhausted || len(params.Content) > artifacts.MaxPayloadSize {
		return invalidf("report page is invalid")
	}
	if page.Section == auditstore.ReportSectionItems {
		if page.RoundID == nil || validateID("report page round", *page.RoundID) != nil ||
			page.LastFinding != nil || page.CoverageCounts == nil ||
			(page.LastItemOrdinal == nil) != (page.EntryCount == 0) ||
			page.LastItemOrdinal != nil && *page.LastItemOrdinal < 0 {
			return invalidf("report items page is invalid")
		}
		if total, ok := coverageCountsTotal(*page.CoverageCounts); !ok || total != page.EntryCount {
			return invalidf("report items page coverage counts are invalid")
		}
	} else if page.RoundID != nil || page.LastItemOrdinal != nil || page.CoverageCounts != nil ||
		(page.LastFinding == nil) != (page.EntryCount == 0) ||
		page.LastFinding != nil && (page.LastFinding.CreatedAt.IsZero() ||
			validateID("report page finding", page.LastFinding.FindingID) != nil) {
		return invalidf("report findings page is invalid")
	}
	if page.EntryCount != 0 {
		var entries []json.RawMessage
		bracketed := make([]byte, 0, len(params.Content)+2)
		bracketed = append(append(append(bracketed, '['), params.Content...), ']')
		if json.Unmarshal(bracketed, &entries) != nil || len(entries) != page.EntryCount {
			return invalidf("report page content is invalid")
		}
	}
	return nil
}

// RecordReportPage appends the next page of a report build. It requires a
// live claim, the finalizing Audit at the expected revision and the page
// ordinal right after the recorded ones, and discards pages of any other
// revision. A failed precondition records nothing.
func (s *ClaimStore) RecordReportPage(ctx context.Context, params auditstore.RecordReportPageParams) error {
	if err := ValidateRecordReportPage(params); err != nil {
		return err
	}
	page := params.Page
	var lastCreatedAt *time.Time
	var lastFindingID *string
	if page.LastFinding != nil {
		createdAt := page.LastFinding.CreatedAt.UTC()
		lastCreatedAt, lastFindingID = &createdAt, &page.LastFinding.FindingID
	}
	var coverageCounts []byte
	if page.CoverageCounts != nil {
		coverageCounts, _ = json.Marshal(page.CoverageCounts)
	}
	content := params.Content
	if content == nil {
		content = []byte{}
	}
	var recorded int
	err := s.db.QueryRow(ctx, recordReportPageSQL,
		params.Claim.AuditID, params.Claim.HolderID, params.Claim.Epoch,
		params.ExpectedAuditRevision, page.Ordinal, string(page.Section), page.RoundID,
		page.LastItemOrdinal, lastCreatedAt, lastFindingID, page.Exhausted,
		page.EntryCount, content, coverageCounts,
	).Scan(&recorded)
	if err == nil {
		return nil
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		return fmt.Errorf("record Audit report page: %w", err)
	}
	if live, liveErr := s.claimLive(ctx, params.Claim); liveErr != nil {
		return liveErr
	} else if !live {
		return auditstore.ErrClaimLost
	}
	return auditstore.ErrPrecondition
}

// ListReportPages returns the metadata of the pages recorded for one Audit
// revision, in page order, without their content.
func (s *QueryStore) ListReportPages(ctx context.Context, auditID string, revision uint64) ([]auditstore.ReportPage, error) {
	if err := validateID("auditID", auditID); err != nil {
		return nil, err
	}
	if revision == 0 || revision > math.MaxInt64 {
		return nil, invalidf("report revision is invalid")
	}
	rows, err := s.db.Query(ctx, listReportPagesSQL, auditID, int64(revision), auditstore.MaxReportPages+1)
	if err != nil {
		return nil, fmt.Errorf("list Audit report pages: %w", err)
	}
	defer rows.Close()
	result := make([]auditstore.ReportPage, 0)
	for rows.Next() {
		page, scanErr := scanReportPage(rows)
		if scanErr != nil {
			return nil, scanErr
		}
		result = append(result, page)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate Audit report pages: %w", err)
	}
	if len(result) > auditstore.MaxReportPages {
		return nil, errors.New("stored Audit report pages exceed their bound")
	}
	return result, nil
}

func scanReportPage(row scanner) (auditstore.ReportPage, error) {
	var page auditstore.ReportPage
	var section string
	var lastCreatedAt *time.Time
	var lastFindingID *string
	var coverageCounts []byte
	if err := row.Scan(
		&page.Ordinal, &section, &page.RoundID, &page.LastItemOrdinal, &lastCreatedAt,
		&lastFindingID, &page.Exhausted, &page.EntryCount, &page.ContentBytes, &coverageCounts,
	); err != nil {
		return auditstore.ReportPage{}, fmt.Errorf("scan Audit report page: %w", err)
	}
	page.Section = auditstore.ReportSection(section)
	if lastCreatedAt != nil && lastFindingID != nil {
		page.LastFinding = &auditstore.ReportFindingCursor{CreatedAt: *lastCreatedAt, FindingID: *lastFindingID}
	} else if lastCreatedAt != nil || lastFindingID != nil {
		return auditstore.ReportPage{}, errors.New("stored Audit report page cursor is invalid")
	}
	if coverageCounts != nil {
		page.CoverageCounts = &auditstore.ReportCoverageCounts{}
		decoder := json.NewDecoder(bytes.NewReader(coverageCounts))
		decoder.DisallowUnknownFields()
		if decoder.Decode(page.CoverageCounts) != nil {
			return auditstore.ReportPage{}, errors.New("stored Audit report page coverage counts are invalid")
		}
	}
	if !page.Section.Valid() || page.ContentBytes < 0 ||
		page.Section == auditstore.ReportSectionItems && page.CoverageCounts == nil {
		return auditstore.ReportPage{}, errors.New("stored Audit report page is invalid")
	}
	if page.CoverageCounts != nil {
		if total, ok := coverageCountsTotal(*page.CoverageCounts); !ok || total != page.EntryCount {
			return auditstore.ReportPage{}, errors.New("stored Audit report page coverage counts are invalid")
		}
	}
	return page, nil
}

// ReadReportSections returns the recorded content of every section of one
// Audit revision's report build. Recording bounds the total content by the
// machine report bound, so the result is bounded too.
func (s *QueryStore) ReadReportSections(
	ctx context.Context, auditID string, revision uint64,
) (map[auditstore.ReportSection]auditstore.ReportSectionContent, error) {
	if err := validateID("auditID", auditID); err != nil {
		return nil, err
	}
	if revision == 0 || revision > math.MaxInt64 {
		return nil, invalidf("report revision is invalid")
	}
	rows, err := s.db.Query(ctx, readReportSectionsSQL, auditID, int64(revision))
	if err != nil {
		return nil, fmt.Errorf("read Audit report sections: %w", err)
	}
	defer rows.Close()
	result := make(map[auditstore.ReportSection]auditstore.ReportSectionContent)
	for rows.Next() {
		var value auditstore.ReportSectionContent
		var section string
		if err := rows.Scan(&section, &value.EntryCount, &value.Content); err != nil {
			return nil, fmt.Errorf("scan Audit report section: %w", err)
		}
		value.Section = auditstore.ReportSection(section)
		if !value.Section.Valid() || value.EntryCount < 0 || (value.EntryCount == 0) != (len(value.Content) == 0) {
			return nil, errors.New("stored Audit report section is invalid")
		}
		result[value.Section] = value
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate Audit report sections: %w", err)
	}
	return result, nil
}

// ListReportItems reads one bounded page of a Round's items with their
// coverage, in item ordinal order after afterOrdinal (-1 starts the Round).
func (s *QueryStore) ListReportItems(
	ctx context.Context, auditID, roundID string, afterOrdinal, limit int,
) ([]auditstore.ReportItem, error) {
	if err := validateID("auditID", auditID); err != nil {
		return nil, err
	}
	if err := validateID("roundID", roundID); err != nil {
		return nil, err
	}
	if afterOrdinal < -1 || limit < 1 || limit > auditstore.MaxReportPageRows+1 {
		return nil, invalidf("report item page is invalid")
	}
	rows, err := s.db.Query(ctx, listReportItemsSQL, auditID, roundID, afterOrdinal, limit)
	if err != nil {
		return nil, fmt.Errorf("list Audit report items: %w", err)
	}
	defer rows.Close()
	result := make([]auditstore.ReportItem, 0, limit)
	for rows.Next() {
		item, scanErr := scanReportItem(rows)
		if scanErr != nil {
			return nil, scanErr
		}
		result = append(result, item)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate Audit report items: %w", err)
	}
	return result, nil
}

func scanReportItem(row scanner) (auditstore.ReportItem, error) {
	var item auditstore.ReportItem
	var taskRef, requested, completed, gaps, resultRef []byte
	var state, status string
	var final, resultDigest *string
	if err := row.Scan(
		&item.ItemID, &item.RoundID, &item.ItemKey, &item.Ordinal, &item.Kind, &item.SubjectKey,
		&taskRef, &item.Task.Digest, &state, &final, &status, &requested, &completed, &gaps,
		&item.Coverage.Rationale, &resultRef, &resultDigest,
	); err != nil {
		return auditstore.ReportItem{}, fmt.Errorf("scan Audit report item: %w", err)
	}
	if json.Unmarshal(taskRef, &item.Task.Ref) != nil || item.Task.Ref.ValidateExact() != nil ||
		validateDigest("stored report item task digest", item.Task.Digest) != nil {
		return auditstore.ReportItem{}, errors.New("stored Audit report item task is invalid")
	}
	item.State, item.Coverage.Status = auditstore.ItemState(state), auditstore.CoverageStatus(status)
	if !item.State.Valid() || !item.Coverage.Status.Valid() ||
		json.Unmarshal(requested, &item.Coverage.Requested) != nil ||
		json.Unmarshal(completed, &item.Coverage.Completed) != nil ||
		json.Unmarshal(gaps, &item.Coverage.Gaps) != nil {
		return auditstore.ReportItem{}, errors.New("stored Audit report item coverage is invalid")
	}
	if final != nil {
		value := auditstore.FinalDisposition(*final)
		if !value.Valid() {
			return auditstore.ReportItem{}, errors.New("stored Audit report item disposition is invalid")
		}
		item.FinalDisposition = &value
	}
	if (resultRef == nil) != (resultDigest == nil) {
		return auditstore.ReportItem{}, errors.New("stored Audit report item result shape is invalid")
	}
	if resultRef != nil {
		item.Result = &auditstore.ExactArtifact{Digest: *resultDigest}
		if json.Unmarshal(resultRef, &item.Result.Ref) != nil || item.Result.Ref.ValidateExact() != nil ||
			validateDigest("stored report item result digest", *resultDigest) != nil {
			return auditstore.ReportItem{}, errors.New("stored Audit report item result is invalid")
		}
	}
	return item, nil
}

// ListReportFindings reads one bounded page of report findings with their
// current assessment and decision and the report projection stored at
// intake. more reports whether the section continues after the page.
func (s *QueryStore) ListReportFindings(
	ctx context.Context, params auditstore.ReportFindingPageParams,
) (findings []auditstore.ReportFinding, more bool, err error) {
	if err := validateID("auditID", params.AuditID); err != nil {
		return nil, false, err
	}
	if !findingSection(params.Section) || params.Limit < 1 || params.Limit > auditstore.MaxReportPageRows ||
		params.ProjectionBytes < 1 || params.ProjectionBytes > artifacts.MaxPayloadSize ||
		params.After != nil && (params.After.CreatedAt.IsZero() || validateID("finding cursor", params.After.FindingID) != nil) {
		return nil, false, invalidf("report finding page is invalid")
	}
	var afterCreatedAt *time.Time
	var afterFindingID *string
	if params.After != nil {
		createdAt := params.After.CreatedAt.UTC()
		afterCreatedAt, afterFindingID = &createdAt, &params.After.FindingID
	}
	rows, err := s.db.Query(ctx, listReportFindingsSQL, params.AuditID, string(params.Section),
		afterCreatedAt, afterFindingID, params.Limit+1, params.ProjectionBytes)
	if err != nil {
		return nil, false, fmt.Errorf("list Audit report findings: %w", err)
	}
	defer rows.Close()
	findings = make([]auditstore.ReportFinding, 0, params.Limit)
	for rows.Next() {
		value, included, scanErr := scanReportFinding(rows)
		if scanErr != nil {
			return nil, false, scanErr
		}
		if !included || len(findings) == params.Limit {
			more = true
			continue
		}
		if value.State != string(params.Section) {
			return nil, false, errors.New("stored Audit report finding state is invalid")
		}
		findings = append(findings, value)
	}
	if err := rows.Err(); err != nil {
		return nil, false, fmt.Errorf("iterate Audit report findings: %w", err)
	}
	return findings, more, nil
}

func scanReportFinding(row scanner) (auditstore.ReportFinding, bool, error) {
	var value auditstore.ReportFinding
	var proposalJSON, projection []byte
	var included bool
	var revision int64
	var assessmentID, semantic, resultDigest, contractDigest *string
	var resultJSON, contractJSON []byte
	var direct *bool
	var assessmentAt *time.Time
	var decisionID, actor, verdict, severity, rationale, subjectDigest *string
	var subjectRevision *int64
	var decisionAt *time.Time
	if err := row.Scan(
		&value.FindingID, &value.State, &proposalJSON, &value.DuplicateTargetID, &revision,
		&value.CreatedAt, &included, &projection,
		&assessmentID, &semantic, &resultJSON, &resultDigest, &direct,
		&contractJSON, &contractDigest, &assessmentAt,
		&decisionID, &actor, &verdict, &severity, &rationale, &subjectRevision,
		&subjectDigest, &decisionAt,
	); err != nil {
		return auditstore.ReportFinding{}, false, fmt.Errorf("scan Audit report finding: %w", err)
	}
	if !included {
		return auditstore.ReportFinding{}, false, nil
	}
	if revision < 1 || json.Unmarshal(proposalJSON, &value.FirstProposal) != nil ||
		validateExactArtifact("first proposal", value.FirstProposal, true) != nil {
		return auditstore.ReportFinding{}, false, errors.New("stored Audit report finding is invalid")
	}
	value.Revision = uint64(revision)
	var err error
	if value.Projection, err = auditdomain.DecodeFindingReportProjection(projection); err != nil {
		return auditstore.ReportFinding{}, false, errors.New("stored Audit report finding projection is invalid")
	}
	if assessmentID != nil {
		if semantic == nil || resultDigest == nil || direct == nil || assessmentAt == nil {
			return auditstore.ReportFinding{}, false, errors.New("stored Audit report assessment is incomplete")
		}
		assessment := &auditstore.ReportFindingAssessment{
			AssessmentID: *assessmentID, SemanticAssessment: *semantic,
			Result: auditstore.ExactArtifact{Digest: *resultDigest}, DirectVerification: *direct,
			AcceptedAt: *assessmentAt,
		}
		if json.Unmarshal(resultJSON, &assessment.Result.Ref) != nil ||
			validateExactArtifact("assessment result", assessment.Result, false) != nil {
			return auditstore.ReportFinding{}, false, errors.New("stored Audit report assessment result is invalid")
		}
		if contractJSON != nil {
			if contractDigest == nil {
				return auditstore.ReportFinding{}, false, errors.New("stored Audit report assessment contract is incomplete")
			}
			assessment.Contract = &auditstore.ExactArtifact{Digest: *contractDigest}
			if json.Unmarshal(contractJSON, &assessment.Contract.Ref) != nil ||
				validateExactArtifact("assessment contract", *assessment.Contract, false) != nil {
				return auditstore.ReportFinding{}, false, errors.New("stored Audit report assessment contract is invalid")
			}
		}
		value.Assessment = assessment
	}
	if decisionID != nil {
		if actor == nil || verdict == nil || rationale == nil || subjectRevision == nil ||
			*subjectRevision < 1 || subjectDigest == nil || decisionAt == nil {
			return auditstore.ReportFinding{}, false, errors.New("stored Audit report decision is incomplete")
		}
		value.Decision = &auditstore.ReportFindingDecision{
			DecisionID: *decisionID, ActorID: *actor, Verdict: *verdict,
			Severity: severity, Rationale: *rationale,
			SubjectRevision: uint64(*subjectRevision), SubjectDigest: *subjectDigest,
			CreatedAt: *decisionAt,
		}
	}
	return value, true, nil
}
