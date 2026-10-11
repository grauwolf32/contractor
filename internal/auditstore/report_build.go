package auditstore

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
	"github.com/jackc/pgx/v5"
)

// ReportSection names one ordered section of a resumable report build. The
// items of every Round come first, then one section per finding state, in
// ReportFindingSections order.
type ReportSection string

const (
	ReportSectionItems         ReportSection = "items"
	ReportSectionConfirmed     ReportSection = "confirmed"
	ReportSectionProposed      ReportSection = "proposed"
	ReportSectionRejected      ReportSection = "rejected"
	ReportSectionDuplicate     ReportSection = "duplicate"
	ReportSectionNeedsEvidence ReportSection = "needs-evidence"
)

// ReportFindingSections are the finding sections in report order. Each
// selects the findings in the state of the same name.
var ReportFindingSections = []ReportSection{
	ReportSectionConfirmed, ReportSectionProposed, ReportSectionRejected,
	ReportSectionDuplicate, ReportSectionNeedsEvidence,
}

func (s ReportSection) Valid() bool {
	return s == ReportSectionItems || s.finding()
}

func (s ReportSection) finding() bool {
	switch s {
	case ReportSectionConfirmed, ReportSectionProposed, ReportSectionRejected,
		ReportSectionDuplicate, ReportSectionNeedsEvidence:
		return true
	}
	return false
}

const (
	// MaxReportFindings bounds the findings of one Audit report (LIM-REV-7).
	MaxReportFindings = 100_000
	// MaxReportPageRows bounds the entries of one recorded report page.
	MaxReportPageRows = 1_000
	// MaxReportPages bounds the recorded pages of one report build. At
	// MaxReportPageRows entries a page, it covers MaxAuditItemsTotal items
	// and MaxReportFindings findings many times over.
	MaxReportPages = 4_096
)

// ReportCoverageCounts counts the items of one report page per coverage
// status. The JSON names are those of the report's coverage counts.
type ReportCoverageCounts struct {
	NotTested      int `json:"notTested"`
	Inconclusive   int `json:"inconclusive"`
	Satisfied      int `json:"satisfied"`
	Violated       int `json:"violated"`
	NotApplicable  int `json:"notApplicable"`
	Blocked        int `json:"blocked"`
	Excluded       int `json:"excluded"`
	TracedComplete int `json:"tracedComplete"`
	TracedPartial  int `json:"tracedPartial"`
	Unmapped       int `json:"unmapped"`
}

// Add counts one item with the given coverage status.
func (c *ReportCoverageCounts) Add(status CoverageStatus) {
	switch status {
	case CoverageNotTested:
		c.NotTested++
	case CoverageInconclusive:
		c.Inconclusive++
	case CoverageSatisfied:
		c.Satisfied++
	case CoverageViolated:
		c.Violated++
	case CoverageNotApplicable:
		c.NotApplicable++
	case CoverageBlocked:
		c.Blocked++
	case CoverageExcluded:
		c.Excluded++
	case CoverageTracedComplete:
		c.TracedComplete++
	case CoverageTracedPartial:
		c.TracedPartial++
	case CoverageUnmapped:
		c.Unmapped++
	}
}

// Merge adds another page's counts.
func (c *ReportCoverageCounts) Merge(other ReportCoverageCounts) {
	c.NotTested += other.NotTested
	c.Inconclusive += other.Inconclusive
	c.Satisfied += other.Satisfied
	c.Violated += other.Violated
	c.NotApplicable += other.NotApplicable
	c.Blocked += other.Blocked
	c.Excluded += other.Excluded
	c.TracedComplete += other.TracedComplete
	c.TracedPartial += other.TracedPartial
	c.Unmapped += other.Unmapped
}

func (c ReportCoverageCounts) total() (int, bool) {
	total := 0
	for _, count := range []int{
		c.NotTested, c.Inconclusive, c.Satisfied, c.Violated, c.NotApplicable,
		c.Blocked, c.Excluded, c.TracedComplete, c.TracedPartial, c.Unmapped,
	} {
		if count < 0 {
			return 0, false
		}
		total += count
	}
	return total, true
}

// ReportFindingCursor is the last finding of a findings page, in report
// order (creation time, then finding ID).
type ReportFindingCursor struct {
	CreatedAt time.Time
	FindingID string
}

// ReportPage is the metadata of one recorded report page. A page that ends
// its section (or, for items, its Round) is Exhausted; otherwise its last
// entry is the cursor where the next page of the section starts.
type ReportPage struct {
	Ordinal         int
	Section         ReportSection
	RoundID         *string
	LastItemOrdinal *int
	LastFinding     *ReportFindingCursor
	Exhausted       bool
	EntryCount      int
	ContentBytes    int
	CoverageCounts  *ReportCoverageCounts
}

// RecordReportPageParams records Content, the page's report entries as
// comma-separated compact JSON values, under a live claim for the Audit
// revision the build belongs to.
type RecordReportPageParams struct {
	Claim                 ControllerClaim
	ExpectedAuditRevision uint64
	Page                  ReportPage
	Content               []byte
}

// ValidateRecordReportPage enforces the page contract of RecordReportPage,
// so a store that skips PostgreSQL cannot accept a page it would reject.
func ValidateRecordReportPage(params RecordReportPageParams) error {
	if err := validateClaimIdentity(params.Claim); err != nil {
		return err
	}
	page := params.Page
	if params.ExpectedAuditRevision == 0 || params.ExpectedAuditRevision > math.MaxInt64 ||
		page.Ordinal < 0 || page.Ordinal >= MaxReportPages || !page.Section.Valid() ||
		page.EntryCount < 0 || page.EntryCount > MaxReportPageRows ||
		(page.EntryCount == 0) != (len(params.Content) == 0) ||
		page.EntryCount == 0 && !page.Exhausted || len(params.Content) > artifacts.MaxPayloadSize {
		return invalidf("report page is invalid")
	}
	if page.Section == ReportSectionItems {
		if page.RoundID == nil || validateID("report page round", *page.RoundID) != nil ||
			page.LastFinding != nil || page.CoverageCounts == nil ||
			(page.LastItemOrdinal == nil) != (page.EntryCount == 0) ||
			page.LastItemOrdinal != nil && *page.LastItemOrdinal < 0 {
			return invalidf("report items page is invalid")
		}
		if total, ok := page.CoverageCounts.total(); !ok || total != page.EntryCount {
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
func (s *PostgresStore) RecordReportPage(ctx context.Context, params RecordReportPageParams) error {
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
		return ErrClaimLost
	}
	return ErrPrecondition
}

// ListReportPages returns the metadata of the pages recorded for one Audit
// revision, in page order, without their content.
func (s *PostgresStore) ListReportPages(ctx context.Context, auditID string, revision uint64) ([]ReportPage, error) {
	if err := validateID("auditID", auditID); err != nil {
		return nil, err
	}
	if revision == 0 || revision > math.MaxInt64 {
		return nil, invalidf("report revision is invalid")
	}
	rows, err := s.db.Query(ctx, listReportPagesSQL, auditID, int64(revision), MaxReportPages+1)
	if err != nil {
		return nil, fmt.Errorf("list Audit report pages: %w", err)
	}
	defer rows.Close()
	result := make([]ReportPage, 0)
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
	if len(result) > MaxReportPages {
		return nil, errors.New("stored Audit report pages exceed their bound")
	}
	return result, nil
}

func scanReportPage(row scanner) (ReportPage, error) {
	var page ReportPage
	var section string
	var lastCreatedAt *time.Time
	var lastFindingID *string
	var coverageCounts []byte
	if err := row.Scan(
		&page.Ordinal, &section, &page.RoundID, &page.LastItemOrdinal, &lastCreatedAt,
		&lastFindingID, &page.Exhausted, &page.EntryCount, &page.ContentBytes, &coverageCounts,
	); err != nil {
		return ReportPage{}, fmt.Errorf("scan Audit report page: %w", err)
	}
	page.Section = ReportSection(section)
	if lastCreatedAt != nil && lastFindingID != nil {
		page.LastFinding = &ReportFindingCursor{CreatedAt: *lastCreatedAt, FindingID: *lastFindingID}
	} else if lastCreatedAt != nil || lastFindingID != nil {
		return ReportPage{}, errors.New("stored Audit report page cursor is invalid")
	}
	if coverageCounts != nil {
		page.CoverageCounts = &ReportCoverageCounts{}
		decoder := json.NewDecoder(bytes.NewReader(coverageCounts))
		decoder.DisallowUnknownFields()
		if decoder.Decode(page.CoverageCounts) != nil {
			return ReportPage{}, errors.New("stored Audit report page coverage counts are invalid")
		}
	}
	if !page.Section.Valid() || page.ContentBytes < 0 ||
		page.Section == ReportSectionItems && page.CoverageCounts == nil {
		return ReportPage{}, errors.New("stored Audit report page is invalid")
	}
	if page.CoverageCounts != nil {
		if total, ok := page.CoverageCounts.total(); !ok || total != page.EntryCount {
			return ReportPage{}, errors.New("stored Audit report page coverage counts are invalid")
		}
	}
	return page, nil
}

// ReportSectionContent is the recorded entries of one report section, the
// comma-joined content of its pages in page order.
type ReportSectionContent struct {
	Section    ReportSection
	EntryCount int
	Content    []byte
}

// ReadReportSections returns the recorded content of every section of one
// Audit revision's report build. Recording bounds the total content by the
// machine report bound, so the result is bounded too.
func (s *PostgresStore) ReadReportSections(
	ctx context.Context, auditID string, revision uint64,
) (map[ReportSection]ReportSectionContent, error) {
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
	result := make(map[ReportSection]ReportSectionContent)
	for rows.Next() {
		var value ReportSectionContent
		var section string
		if err := rows.Scan(&section, &value.EntryCount, &value.Content); err != nil {
			return nil, fmt.Errorf("scan Audit report section: %w", err)
		}
		value.Section = ReportSection(section)
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

// ReportItem is one item of a report page with its coverage.
type ReportItem struct {
	ItemID           string
	RoundID          string
	ItemKey          string
	Ordinal          int
	Kind             string
	SubjectKey       string
	Task             ExactArtifact
	State            ItemState
	FinalDisposition *FinalDisposition
	Coverage         Coverage
	Result           *ExactArtifact
}

// ListReportItems reads one bounded page of a Round's items with their
// coverage, in item ordinal order after afterOrdinal (-1 starts the Round).
func (s *PostgresStore) ListReportItems(
	ctx context.Context, auditID, roundID string, afterOrdinal, limit int,
) ([]ReportItem, error) {
	if err := validateID("auditID", auditID); err != nil {
		return nil, err
	}
	if err := validateID("roundID", roundID); err != nil {
		return nil, err
	}
	if afterOrdinal < -1 || limit < 1 || limit > MaxReportPageRows+1 {
		return nil, invalidf("report item page is invalid")
	}
	rows, err := s.db.Query(ctx, listReportItemsSQL, auditID, roundID, afterOrdinal, limit)
	if err != nil {
		return nil, fmt.Errorf("list Audit report items: %w", err)
	}
	defer rows.Close()
	result := make([]ReportItem, 0, limit)
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

func scanReportItem(row scanner) (ReportItem, error) {
	var item ReportItem
	var taskRef, requested, completed, gaps, resultRef []byte
	var state, status string
	var final, resultDigest *string
	if err := row.Scan(
		&item.ItemID, &item.RoundID, &item.ItemKey, &item.Ordinal, &item.Kind, &item.SubjectKey,
		&taskRef, &item.Task.Digest, &state, &final, &status, &requested, &completed, &gaps,
		&item.Coverage.Rationale, &resultRef, &resultDigest,
	); err != nil {
		return ReportItem{}, fmt.Errorf("scan Audit report item: %w", err)
	}
	if json.Unmarshal(taskRef, &item.Task.Ref) != nil || item.Task.Ref.ValidateExact() != nil ||
		validateDigest("stored report item task digest", item.Task.Digest) != nil {
		return ReportItem{}, errors.New("stored Audit report item task is invalid")
	}
	item.State, item.Coverage.Status = ItemState(state), CoverageStatus(status)
	if !item.State.Valid() || !item.Coverage.Status.Valid() ||
		json.Unmarshal(requested, &item.Coverage.Requested) != nil ||
		json.Unmarshal(completed, &item.Coverage.Completed) != nil ||
		json.Unmarshal(gaps, &item.Coverage.Gaps) != nil {
		return ReportItem{}, errors.New("stored Audit report item coverage is invalid")
	}
	if final != nil {
		value := FinalDisposition(*final)
		if !value.Valid() {
			return ReportItem{}, errors.New("stored Audit report item disposition is invalid")
		}
		item.FinalDisposition = &value
	}
	if (resultRef == nil) != (resultDigest == nil) {
		return ReportItem{}, errors.New("stored Audit report item result shape is invalid")
	}
	if resultRef != nil {
		item.Result = &ExactArtifact{Digest: *resultDigest}
		if json.Unmarshal(resultRef, &item.Result.Ref) != nil || item.Result.Ref.ValidateExact() != nil ||
			validateDigest("stored report item result digest", *resultDigest) != nil {
			return ReportItem{}, errors.New("stored Audit report item result is invalid")
		}
	}
	return item, nil
}

// ReportFindingPageParams selects one bounded page of the findings in one
// state, after an optional cursor. A page holds at most Limit findings and
// stops before the finding whose report projection would take the page's
// projections past ProjectionBytes; its first finding is always included.
type ReportFindingPageParams struct {
	AuditID         string
	Section         ReportSection
	After           *ReportFindingCursor
	Limit           int
	ProjectionBytes int
}

// ListReportFindings reads one bounded page of report findings with their
// current assessment and decision and the report projection stored at
// intake. more reports whether the section continues after the page.
func (s *PostgresStore) ListReportFindings(
	ctx context.Context, params ReportFindingPageParams,
) (findings []ReportFinding, more bool, err error) {
	if err := validateID("auditID", params.AuditID); err != nil {
		return nil, false, err
	}
	if !params.Section.finding() || params.Limit < 1 || params.Limit > MaxReportPageRows ||
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
	findings = make([]ReportFinding, 0, params.Limit)
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

func scanReportFinding(row scanner) (ReportFinding, bool, error) {
	var value ReportFinding
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
		return ReportFinding{}, false, fmt.Errorf("scan Audit report finding: %w", err)
	}
	if !included {
		return ReportFinding{}, false, nil
	}
	if revision < 1 || json.Unmarshal(proposalJSON, &value.FirstProposal) != nil ||
		validateExactArtifact("first proposal", value.FirstProposal, true) != nil {
		return ReportFinding{}, false, errors.New("stored Audit report finding is invalid")
	}
	value.Revision = uint64(revision)
	var err error
	if value.Projection, err = auditdomain.DecodeFindingReportProjection(projection); err != nil {
		return ReportFinding{}, false, errors.New("stored Audit report finding projection is invalid")
	}
	if assessmentID != nil {
		if semantic == nil || resultDigest == nil || direct == nil || assessmentAt == nil {
			return ReportFinding{}, false, errors.New("stored Audit report assessment is incomplete")
		}
		assessment := &ReportFindingAssessment{
			AssessmentID: *assessmentID, SemanticAssessment: *semantic,
			Result: ExactArtifact{Digest: *resultDigest}, DirectVerification: *direct,
			AcceptedAt: *assessmentAt,
		}
		if json.Unmarshal(resultJSON, &assessment.Result.Ref) != nil ||
			validateExactArtifact("assessment result", assessment.Result, false) != nil {
			return ReportFinding{}, false, errors.New("stored Audit report assessment result is invalid")
		}
		if contractJSON != nil {
			if contractDigest == nil {
				return ReportFinding{}, false, errors.New("stored Audit report assessment contract is incomplete")
			}
			assessment.Contract = &ExactArtifact{Digest: *contractDigest}
			if json.Unmarshal(contractJSON, &assessment.Contract.Ref) != nil ||
				validateExactArtifact("assessment contract", *assessment.Contract, false) != nil {
				return ReportFinding{}, false, errors.New("stored Audit report assessment contract is invalid")
			}
		}
		value.Assessment = assessment
	}
	if decisionID != nil {
		if actor == nil || verdict == nil || rationale == nil || subjectRevision == nil ||
			*subjectRevision < 1 || subjectDigest == nil || decisionAt == nil {
			return ReportFinding{}, false, errors.New("stored Audit report decision is incomplete")
		}
		value.Decision = &ReportFindingDecision{
			DecisionID: *decisionID, ActorID: *actor, Verdict: *verdict,
			Severity: severity, Rationale: *rationale,
			SubjectRevision: uint64(*subjectRevision), SubjectDigest: *subjectDigest,
			CreatedAt: *decisionAt,
		}
	}
	return value, true, nil
}
