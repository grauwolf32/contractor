package auditimport

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/pagelimit"
)

// reportBudget bounds one Finalize attempt (LIM-REV-9). A page reads at most
// pageRows report entries, and a findings page ends with the finding whose
// report projection brings its projections to pageBytes; an attempt records
// at most pagesPerAttempt pages and returns, and the next attempt resumes
// from the recorded pages. Assembly then reads the recorded entries once,
// bounded by the machine report bound.
type reportBudget struct {
	pageRows        int
	pageBytes       int
	pagesPerAttempt int
}

var defaultReportBudget = reportBudget{
	pageRows: pagelimit.MaxRows, pageBytes: 8 << 20, pagesPerAttempt: 32,
}

// errReportPagesInconsistent means recorded pages do not follow the build
// order of the current Audit revision. Only a defect can cause it.
var errReportPagesInconsistent = errors.New("recorded Audit report pages are inconsistent")

// reportPosition is one section of the build order: the items of one Round,
// or the findings of one state.
type reportPosition struct {
	section auditstore.ReportSection
	round   *auditstore.Round
}

// reportBuild is the progress of one Audit revision's report build, replayed
// from its recorded pages.
type reportBuild struct {
	positions []reportPosition
	index     int
	ordinal   int
	// afterItem and afterFinding are the cursor inside positions[index].
	afterItem    int
	afterFinding *auditstore.ReportFindingCursor
	roundItems   int
	items        int
	findings     int
	contentBytes int
	coverage     auditstore.ReportCoverageCounts
}

func newReportBuild(rounds []auditstore.Round) reportBuild {
	positions := make([]reportPosition, 0, len(rounds)+len(auditstore.ReportFindingSections))
	for index := range rounds {
		positions = append(positions, reportPosition{section: auditstore.ReportSectionItems, round: &rounds[index]})
	}
	for _, section := range auditstore.ReportFindingSections {
		positions = append(positions, reportPosition{section: section})
	}
	return reportBuild{positions: positions, afterItem: -1}
}

func (b *reportBuild) complete() bool { return b.index == len(b.positions) }

// apply advances the build past one recorded page. The page must be the next
// one of the current section; an exhausted Round must hold exactly its
// expected items.
func (b *reportBuild) apply(page auditstore.ReportPage) error {
	if b.complete() || page.Ordinal != b.ordinal {
		return errReportPagesInconsistent
	}
	position := b.positions[b.index]
	if page.Section != position.section ||
		position.round != nil && (page.RoundID == nil || *page.RoundID != position.round.RoundID) {
		return errReportPagesInconsistent
	}
	b.ordinal++
	b.contentBytes += page.ContentBytes
	if position.round != nil {
		if page.CoverageCounts == nil {
			return errReportPagesInconsistent
		}
		b.coverage.Merge(*page.CoverageCounts)
		b.items += page.EntryCount
		b.roundItems += page.EntryCount
		if page.LastItemOrdinal != nil {
			b.afterItem = *page.LastItemOrdinal
		}
		if page.Exhausted && b.roundItems != position.round.ExpectedItemCount {
			return fmt.Errorf("%w: report coverage barrier is incomplete", ErrPermanent)
		}
	} else {
		b.findings += page.EntryCount
		if page.LastFinding != nil {
			cursor := *page.LastFinding
			b.afterFinding = &cursor
		}
	}
	if page.Exhausted {
		b.index++
		b.afterItem, b.afterFinding, b.roundItems = -1, nil, 0
	}
	return nil
}

// resumeReport replays the pages recorded for the snapshot's Audit revision.
func (i *Importer) resumeReport(
	ctx context.Context, audit auditstore.Audit, rounds []auditstore.Round,
) (reportBuild, error) {
	build := newReportBuild(rounds)
	pages, err := i.reads.ListReportPages(ctx, audit.AuditID, audit.Revision)
	if err != nil {
		return reportBuild{}, err
	}
	for _, page := range pages {
		if err := build.apply(page); err != nil {
			return reportBuild{}, err
		}
	}
	return build, nil
}

// recordNextPage reads, validates and records the next page of the build.
func (i *Importer) recordNextPage(
	ctx context.Context, claim auditstore.ControllerClaim, audit auditstore.Audit, build *reportBuild,
) error {
	position := build.positions[build.index]
	page := auditstore.ReportPage{Ordinal: build.ordinal, Section: position.section}
	var content []byte
	var err error
	if position.round != nil {
		content, err = i.itemsPage(ctx, audit, position.round, build, &page)
	} else {
		content, err = i.findingsPage(ctx, audit, build, &page)
	}
	if err != nil {
		return err
	}
	// The recorded entries become the report's items and finding sections,
	// so they must leave room in the machine report bound.
	if build.contentBytes+len(content) > artifacts.MaxPayloadSize {
		return fmt.Errorf("%w: machine report exceeds its bound", ErrPermanent)
	}
	page.ContentBytes = len(content)
	if err := i.store.RecordReportPage(ctx, auditstore.RecordReportPageParams{
		Claim: claim, ExpectedAuditRevision: audit.Revision, Page: page, Content: content,
	}); err != nil {
		return err
	}
	return build.apply(page)
}

func (i *Importer) itemsPage(
	ctx context.Context, audit auditstore.Audit, round *auditstore.Round,
	build *reportBuild, page *auditstore.ReportPage,
) ([]byte, error) {
	rows, err := i.reads.ListReportItems(ctx, audit.AuditID, round.RoundID, build.afterItem, i.budget.pageRows+1)
	if err != nil {
		return nil, err
	}
	page.Exhausted = len(rows) <= i.budget.pageRows
	rows = rows[:min(len(rows), i.budget.pageRows)]
	counts := auditstore.ReportCoverageCounts{}
	content := make([]byte, 0)
	for _, row := range rows {
		if row.RoundID != round.RoundID || row.State != auditstore.ItemSettled || row.FinalDisposition == nil {
			return nil, fmt.Errorf("%w: report item barrier is incomplete", ErrPermanent)
		}
		entry, err := json.Marshal(reportItem{
			ItemID: row.ItemID, RoundID: row.RoundID, ItemKey: row.ItemKey, Ordinal: row.Ordinal,
			Kind: row.Kind, SubjectKey: row.SubjectKey, FinalDisposition: *row.FinalDisposition,
			Coverage: row.Coverage, Result: row.Result, Task: row.Task,
		})
		if err != nil {
			return nil, err
		}
		content = appendReportEntry(content, entry)
		counts.Add(row.Coverage.Status)
	}
	if page.Exhausted && build.roundItems+len(rows) != round.ExpectedItemCount {
		return nil, fmt.Errorf("%w: report coverage barrier is incomplete", ErrPermanent)
	}
	page.RoundID, page.EntryCount, page.CoverageCounts = &round.RoundID, len(rows), &counts
	if len(rows) != 0 {
		last := rows[len(rows)-1].Ordinal
		page.LastItemOrdinal = &last
	}
	return content, nil
}

func (i *Importer) findingsPage(
	ctx context.Context, audit auditstore.Audit, build *reportBuild, page *auditstore.ReportPage,
) ([]byte, error) {
	rows, more, err := i.reads.ListReportFindings(ctx, auditstore.ReportFindingPageParams{
		AuditID: audit.AuditID, Section: page.Section, After: build.afterFinding,
		Limit: i.budget.pageRows, ProjectionBytes: i.budget.pageBytes,
	})
	if err != nil {
		return nil, err
	}
	if build.findings+len(rows) > auditstore.MaxReportFindings {
		return nil, fmt.Errorf("%w: report findings exceed their bound", ErrPermanent)
	}
	content := make([]byte, 0)
	for _, row := range rows {
		entry, err := json.Marshal(newReportFinding(row))
		if err != nil {
			return nil, err
		}
		content = appendReportEntry(content, entry)
	}
	page.Exhausted, page.EntryCount = !more, len(rows)
	if len(rows) != 0 {
		last := rows[len(rows)-1]
		page.LastFinding = &auditstore.ReportFindingCursor{CreatedAt: last.CreatedAt, FindingID: last.FindingID}
	}
	return content, nil
}

func appendReportEntry(content, entry []byte) []byte {
	if len(content) != 0 {
		content = append(content, ',')
	}
	return append(content, entry...)
}

func newReportFinding(row auditstore.ReportFinding) reportFinding {
	projection := row.Projection
	value := reportFinding{
		FindingID: row.FindingID, State: row.State, Revision: row.Revision,
		FirstProposal: row.FirstProposal, Title: projection.Title, Description: projection.Description,
		Subject: projection.Subject, Hypothesis: projection.Hypothesis,
		Locations: projection.Locations, HTTPExchange: projection.HTTPExchange,
		SeveritySuggestion: projection.SeveritySuggestion,
		StandardRefs:       append([]auditdomain.StandardReference{}, projection.StandardRefs...),
		Limitations:        append([]string{}, projection.Limitations...),
		DuplicateTargetID:  row.DuplicateTargetID,
	}
	if row.Assessment != nil {
		value.Assessment = &reportFindingAssessment{
			AssessmentID:       row.Assessment.AssessmentID,
			SemanticAssessment: row.Assessment.SemanticAssessment,
			Result:             row.Assessment.Result,
			DirectVerification: row.Assessment.DirectVerification,
			Contract:           row.Assessment.Contract,
			AcceptedAt:         row.Assessment.AcceptedAt,
		}
	}
	if row.Decision != nil {
		value.AnalystDecision = &reportFindingDecision{
			DecisionID: row.Decision.DecisionID, ActorID: row.Decision.ActorID,
			Verdict: row.Decision.Verdict, Severity: row.Decision.Severity,
			Rationale: row.Decision.Rationale, SubjectRevision: row.Decision.SubjectRevision,
			SubjectDigest: row.Decision.SubjectDigest, CreatedAt: row.Decision.CreatedAt,
		}
	}
	return value
}

// readReportSections reads the recorded entries of a complete build as JSON
// arrays and checks them against the replayed page metadata.
func (i *Importer) readReportSections(
	ctx context.Context, audit auditstore.Audit, build reportBuild,
) (json.RawMessage, reportFindingSections, reportFindingCounts, error) {
	sections, err := i.reads.ReadReportSections(ctx, audit.AuditID, audit.Revision)
	if err != nil {
		return nil, reportFindingSections{}, reportFindingCounts{}, err
	}
	array := func(section auditstore.ReportSection) (json.RawMessage, int) {
		value := sections[section]
		result := make([]byte, 0, len(value.Content)+2)
		return append(append(append(result, '['), value.Content...), ']'), value.EntryCount
	}
	items, itemCount := array(auditstore.ReportSectionItems)
	var findings reportFindingSections
	var counts reportFindingCounts
	findings.Confirmed, counts.confirmed = array(auditstore.ReportSectionConfirmed)
	findings.Proposed, counts.proposed = array(auditstore.ReportSectionProposed)
	findings.Rejected, counts.rejected = array(auditstore.ReportSectionRejected)
	findings.Duplicates, counts.duplicates = array(auditstore.ReportSectionDuplicate)
	findings.NeedsEvidence, counts.needsEvidence = array(auditstore.ReportSectionNeedsEvidence)
	if itemCount != build.items || counts.total() != build.findings {
		return nil, reportFindingSections{}, reportFindingCounts{}, errReportPagesInconsistent
	}
	return items, findings, counts, nil
}
