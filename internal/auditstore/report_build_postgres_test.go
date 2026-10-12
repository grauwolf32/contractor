package auditstore

import (
	"encoding/json"
	"errors"
	"sort"
	"testing"
	"time"
)

// finalizingReportFixture collects the fixture's child Run with three
// admitted findings and closes the Audit into finalizing.
func finalizingReportFixture(t *testing.T, name string) (collectingAuditFixture, Audit, []string) {
	t.Helper()
	f := newCollectingAuditFixture(t, name, 1<<20)
	findings := []string{collectPurgeFinding(t, f, "receipt-"+name+"-a")}
	for _, suffix := range []string{"b", "c"} {
		proposal := testExact("audit-findings", "candidate-"+name+"-"+suffix, "proposal-r1")
		proposal.MediaType, proposal.SizeBytes = "application/json", 128
		findings = append(findings, insertAuditChildFinding(t, f, "receipt-"+name+"-"+suffix, proposal))
	}
	current, err := f.store.Get(f.ctx, f.audit.OwnerID, f.audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	finalizing, err := f.store.TransitionClaimed(f.ctx, ClaimedTransitionParams{
		Claim: f.claim, ExpectedRevision: current.Revision,
		ExpectedState: AuditActive, TargetState: AuditFinalizing,
	})
	if err != nil {
		t.Fatal(err)
	}
	return f, finalizing, findings
}

func reportItemsPage(f collectingAuditFixture, revision uint64, ordinal int) RecordReportPageParams {
	roundID, last := *f.audit.CurrentRoundID, 0
	counts := ReportCoverageCounts{Satisfied: 1}
	return RecordReportPageParams{
		Claim: f.claim, ExpectedAuditRevision: revision,
		Page: ReportPage{
			Ordinal: ordinal, Section: ReportSectionItems, RoundID: &roundID,
			LastItemOrdinal: &last, Exhausted: true, EntryCount: 1, CoverageCounts: &counts,
		},
		Content: []byte(`{"itemId":"` + f.itemID + `"}`),
	}
}

func TestPostgresReportPagesFollowOneRevisionAndLeaveWithFinalizing(t *testing.T) {
	f, audit, _ := finalizingReportFixture(t, "report-pages")
	page := reportItemsPage(f, audit.Revision, 1)
	if err := f.store.RecordReportPage(f.ctx, page); !errors.Is(err, ErrPrecondition) {
		t.Fatalf("page after a gap = %v, want precondition failure", err)
	}
	page.Page.Ordinal = 0
	if err := f.store.RecordReportPage(f.ctx, page); err != nil {
		t.Fatal(err)
	}
	if err := f.store.RecordReportPage(f.ctx, page); !errors.Is(err, ErrPrecondition) {
		t.Fatalf("recorded page again = %v, want precondition failure", err)
	}
	stale := reportItemsPage(f, audit.Revision+1, 1)
	if err := f.store.RecordReportPage(f.ctx, stale); !errors.Is(err, ErrPrecondition) {
		t.Fatalf("page of another revision = %v, want precondition failure", err)
	}
	lost := reportItemsPage(f, audit.Revision, 1)
	lost.Claim.Epoch++
	if err := f.store.RecordReportPage(f.ctx, lost); !errors.Is(err, ErrClaimLost) {
		t.Fatalf("page under a stale claim = %v, want claim lost", err)
	}
	findings := RecordReportPageParams{
		Claim: f.claim, ExpectedAuditRevision: audit.Revision,
		Page: ReportPage{Ordinal: 1, Section: ReportSectionProposed, Exhausted: true},
	}
	if err := f.store.RecordReportPage(f.ctx, findings); err != nil {
		t.Fatal(err)
	}
	pages, err := f.store.ListReportPages(f.ctx, audit.AuditID, audit.Revision)
	if err != nil || len(pages) != 2 || pages[0].Section != ReportSectionItems ||
		*pages[0].RoundID != *f.audit.CurrentRoundID || *pages[0].LastItemOrdinal != 0 ||
		pages[0].CoverageCounts == nil || pages[0].CoverageCounts.Satisfied != 1 ||
		pages[0].ContentBytes != len(page.Content) || pages[1].Section != ReportSectionProposed ||
		pages[1].EntryCount != 0 || !pages[1].Exhausted || pages[1].LastFinding != nil {
		t.Fatalf("recorded pages = (%+v, %v)", pages, err)
	}
	sections, err := f.store.ReadReportSections(f.ctx, audit.AuditID, audit.Revision)
	if err != nil || len(sections) != 2 || string(sections[ReportSectionItems].Content) != string(page.Content) ||
		sections[ReportSectionItems].EntryCount != 1 || len(sections[ReportSectionProposed].Content) != 0 {
		t.Fatalf("report sections = (%+v, %v)", sections, err)
	}

	// A revision change makes the build stale: the next revision starts over
	// and its first page discards the stale pages.
	if _, err := f.pool.Exec(f.ctx, `UPDATE audits SET revision = revision + 1 WHERE audit_id = $1`, audit.AuditID); err != nil {
		t.Fatal(err)
	}
	if err := f.store.RecordReportPage(f.ctx, reportItemsPage(f, audit.Revision+1, 0)); err != nil {
		t.Fatal(err)
	}
	if count := reportPageRows(t, f); count != 1 {
		t.Fatalf("stale pages remain: %d rows", count)
	}
	if pages, err := f.store.ListReportPages(f.ctx, audit.AuditID, audit.Revision); err != nil || len(pages) != 0 {
		t.Fatalf("stale revision pages = (%+v, %v)", pages, err)
	}

	// Every exit from finalizing discards the build in its transaction.
	if _, err := f.store.TransitionClaimed(f.ctx, ClaimedTransitionParams{
		Claim: f.claim, ExpectedRevision: audit.Revision + 1,
		ExpectedState: AuditFinalizing, TargetState: AuditFailed,
		Reason: &StopReason{Code: "report-contract-invalid", Message: "Report pages test."},
	}); err != nil {
		t.Fatal(err)
	}
	if count := reportPageRows(t, f); count != 0 {
		t.Fatalf("failed Audit keeps %d report pages", count)
	}
	failed, err := f.store.Get(f.ctx, f.audit.OwnerID, f.audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.store.RecordReportPage(f.ctx, reportItemsPage(f, failed.Revision, 0)); !errors.Is(err, ErrPrecondition) {
		t.Fatalf("page of a failed Audit = %v, want precondition failure", err)
	}
}

func reportPageRows(t *testing.T, f collectingAuditFixture) int {
	t.Helper()
	var count int
	if err := f.pool.QueryRow(f.ctx, `SELECT count(*) FROM audit_report_pages WHERE audit_id = $1`,
		f.audit.AuditID).Scan(&count); err != nil {
		t.Fatal(err)
	}
	return count
}

func TestPostgresReportFindingPagesBoundRowsAndProjectionBytes(t *testing.T) {
	f, audit, findings := finalizingReportFixture(t, "report-findings")
	if _, err := f.pool.Exec(f.ctx, `
UPDATE audit_findings
   SET created_at = timestamptz '2026-10-01 00:00:00+00'
 WHERE audit_id = $1`, audit.AuditID); err != nil {
		t.Fatal(err)
	}
	var projectionBytes int
	if err := f.pool.QueryRow(f.ctx, `
SELECT octet_length(receipt.report_projection)
  FROM audit_findings AS finding
  JOIN finding_proposal_receipts AS receipt ON receipt.receipt_id = finding.first_receipt_id
 WHERE finding.finding_id = $1`, findings[0]).Scan(&projectionBytes); err != nil {
		t.Fatal(err)
	}
	read := func(after *ReportFindingCursor, limit, bytes int) ([]ReportFinding, bool) {
		t.Helper()
		page, more, err := f.store.ListReportFindings(f.ctx, ReportFindingPageParams{
			AuditID: audit.AuditID, Section: ReportSectionProposed, After: after,
			Limit: limit, ProjectionBytes: bytes,
		})
		if err != nil {
			t.Fatal(err)
		}
		return page, more
	}
	// Equal creation times order by finding ID, the cursor's tie breaker.
	ordered := append([]string(nil), findings...)
	sort.Strings(ordered)
	for _, test := range []struct {
		limit, bytes, want int
		more               bool
	}{
		{10, projectionBytes, 1, true},      // the first finding always fits
		{10, projectionBytes + 1, 2, true},  // the third would pass the budget
		{10, 3 * projectionBytes, 3, false}, // the whole section
		{2, 3 * projectionBytes, 2, true},   // the row bound
	} {
		page, more := read(nil, test.limit, test.bytes)
		if len(page) != test.want || more != test.more {
			t.Fatalf("page(limit=%d, bytes=%d) = %d findings, more=%t", test.limit, test.bytes, len(page), more)
		}
		for index, finding := range page {
			if finding.FindingID != ordered[index] || finding.Projection.Title != "Test finding." ||
				finding.State != string(ReportSectionProposed) || finding.FirstProposal.Ref.Revision == nil {
				t.Fatalf("finding %d = %+v", index, finding)
			}
		}
	}
	first, _ := read(nil, 1, projectionBytes)
	cursor := &ReportFindingCursor{CreatedAt: first[0].CreatedAt, FindingID: first[0].FindingID}
	rest, more := read(cursor, 10, 3*projectionBytes)
	if more || len(rest) != 2 || rest[0].FindingID != ordered[1] || rest[1].FindingID != ordered[2] {
		t.Fatalf("page after %s = %+v, more=%t", cursor.FindingID, rest, more)
	}
	if page, more, err := f.store.ListReportFindings(f.ctx, ReportFindingPageParams{
		AuditID: audit.AuditID, Section: ReportSectionConfirmed, Limit: 10, ProjectionBytes: 1 << 20,
	}); err != nil || more || len(page) != 0 {
		t.Fatalf("empty section = (%+v, %t, %v)", page, more, err)
	}
	if _, _, err := f.store.ListReportFindings(f.ctx, ReportFindingPageParams{
		AuditID: audit.AuditID, Section: ReportSectionItems, Limit: 10, ProjectionBytes: 1,
	}); !errors.Is(err, ErrInvalid) {
		t.Fatalf("items section as findings = %v, want invalid", err)
	}
	items, err := f.store.ListReportItems(f.ctx, audit.AuditID, *f.audit.CurrentRoundID, -1, 10)
	if err != nil || len(items) != 1 || items[0].ItemID != f.itemID || items[0].State != ItemSettled ||
		items[0].FinalDisposition == nil || items[0].Coverage.Status != CoverageSatisfied || items[0].Result == nil {
		t.Fatalf("report items = (%+v, %v)", items, err)
	}
}

func TestValidateRecordReportPageMatchesTheStoredShape(t *testing.T) {
	claim := ControllerClaim{AuditID: "audit-pages", HolderID: "holder", Epoch: 1}
	roundID, last := "round-pages", 3
	counts := ReportCoverageCounts{NotTested: 1, Blocked: 1}
	valid := RecordReportPageParams{
		Claim: claim, ExpectedAuditRevision: 2,
		Page: ReportPage{
			Section: ReportSectionItems, RoundID: &roundID, LastItemOrdinal: &last,
			EntryCount: 2, CoverageCounts: &counts,
		},
		Content: []byte(`{"itemId":"a"},{"itemId":"b"}`),
	}
	if err := ValidateRecordReportPage(valid); err != nil {
		t.Fatal(err)
	}
	finding := RecordReportPageParams{
		Claim: claim, ExpectedAuditRevision: 2,
		Page: ReportPage{
			Section: ReportSectionDuplicate, EntryCount: 1,
			LastFinding: &ReportFindingCursor{CreatedAt: time.Unix(10, 0), FindingID: "finding-a"},
		},
		Content: []byte(`{"findingId":"finding-a"}`),
	}
	if err := ValidateRecordReportPage(finding); err != nil {
		t.Fatal(err)
	}
	for name, mutate := range map[string]func(*RecordReportPageParams){
		"entry count":        func(p *RecordReportPageParams) { p.Page.EntryCount = 3 },
		"content not JSON":   func(p *RecordReportPageParams) { p.Content = []byte(`{"itemId":"a"},`) },
		"empty unexhausted":  func(p *RecordReportPageParams) { p.Page.EntryCount, p.Content, p.Page.LastItemOrdinal = 0, nil, nil },
		"coverage total":     func(p *RecordReportPageParams) { p.Page.CoverageCounts = &ReportCoverageCounts{NotTested: 1} },
		"missing round":      func(p *RecordReportPageParams) { p.Page.RoundID = nil },
		"finding cursor":     func(p *RecordReportPageParams) { p.Page.LastFinding = finding.Page.LastFinding },
		"unknown section":    func(p *RecordReportPageParams) { p.Page.Section = "items-v2" },
		"too many entries":   func(p *RecordReportPageParams) { p.Page.EntryCount = MaxReportPageRows + 1 },
		"page beyond bound":  func(p *RecordReportPageParams) { p.Page.Ordinal = MaxReportPages },
		"stale revision":     func(p *RecordReportPageParams) { p.ExpectedAuditRevision = 0 },
		"items cursor shape": func(p *RecordReportPageParams) { p.Page.LastItemOrdinal = nil },
	} {
		params := valid
		counts := *valid.Page.CoverageCounts
		params.Page.CoverageCounts = &counts
		mutate(&params)
		if err := ValidateRecordReportPage(params); !errors.Is(err, ErrInvalid) {
			t.Errorf("%s: %v, want invalid", name, err)
		}
	}
	findingWithCounts := finding
	findingWithCounts.Page.CoverageCounts = &counts
	if err := ValidateRecordReportPage(findingWithCounts); !errors.Is(err, ErrInvalid) {
		t.Errorf("findings page with coverage counts: %v", err)
	}
	encoded, err := json.Marshal(ReportCoverageCounts{TracedPartial: 2})
	if err != nil || string(encoded) != `{"notTested":0,"inconclusive":0,"satisfied":0,"violated":0,"notApplicable":0,"blocked":0,"excluded":0,"tracedComplete":0,"tracedPartial":2,"unmapped":0}` {
		t.Fatalf("coverage counts encoding = %s, %v", encoded, err)
	}
}
