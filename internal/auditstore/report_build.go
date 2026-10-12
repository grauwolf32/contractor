package auditstore

import (
	"time"
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

// ReportSectionContent is the recorded entries of one report section, the
// comma-joined content of its pages in page order.
type ReportSectionContent struct {
	Section    ReportSection
	EntryCount int
	Content    []byte
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
