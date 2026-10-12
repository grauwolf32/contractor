package auditimport

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditbaseline"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/clone"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/contracts"
)

const ReportSchema = "contractor.audit.report.v1"

type reportProfile struct {
	Name              string                         `json:"name"`
	Version           string                         `json:"version"`
	Digest            string                         `json:"digest"`
	Mode              config.AuditProfileMode        `json:"mode"`
	Standards         []config.AuditStandardRef      `json:"standards"`
	StandardSelection *config.AuditStandardSelection `json:"standardSelection,omitempty"`
}

type reportRound struct {
	RoundID  string                   `json:"roundId"`
	Ordinal  int                      `json:"ordinal"`
	Manifest auditstore.ExactArtifact `json:"manifest"`
}

type reportBaseline struct {
	Digest                   string                              `json:"digest"`
	Inputs                   map[string]auditstore.ExactArtifact `json:"inputs"`
	Scope                    map[string]string                   `json:"scope"`
	SourceContentDigest      string                              `json:"sourceContentDigest"`
	CanonicalInventoryDigest string                              `json:"canonicalInventoryDigest"`
	Worklist                 auditstore.ExactArtifact            `json:"worklist"`
	InventoryGaps            []string                            `json:"inventoryGaps"`
	Standards                []auditstandards.PinnedPackage      `json:"standards"`
	StandardSelection        *config.AuditStandardSelection      `json:"standardSelection,omitempty"`
}

type reportStopReason struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

type reportCoverageCounts struct {
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

type reportCoverageSummary struct {
	Counts                reportCoverageCounts `json:"counts"`
	SelectedItems         int                  `json:"selectedItems"`
	ApplicableDenominator int                  `json:"applicableDenominator"`
	AssessedApplicable    int                  `json:"assessedApplicable"`
	AssessedPercent       *float64             `json:"assessedPercent"`
	ZeroDenominator       bool                 `json:"zeroDenominator"`
}

type reportItem struct {
	ItemID           string                      `json:"itemId"`
	RoundID          string                      `json:"roundId"`
	ItemKey          string                      `json:"itemKey"`
	Ordinal          int                         `json:"ordinal"`
	Kind             string                      `json:"kind"`
	SubjectKey       string                      `json:"subjectKey"`
	FinalDisposition auditstore.FinalDisposition `json:"finalDisposition"`
	Coverage         auditstore.Coverage         `json:"coverage"`
	Result           *auditstore.ExactArtifact   `json:"result,omitempty"`
	Task             auditstore.ExactArtifact    `json:"task"`
}

type reportFindingDecision struct {
	DecisionID      string    `json:"decisionId"`
	ActorID         string    `json:"actorId"`
	Verdict         string    `json:"verdict"`
	Severity        *string   `json:"severity,omitempty"`
	Rationale       string    `json:"rationale"`
	SubjectRevision uint64    `json:"subjectRevision"`
	SubjectDigest   string    `json:"subjectDigest"`
	CreatedAt       time.Time `json:"createdAt"`
}

type reportFindingAssessment struct {
	AssessmentID       string                    `json:"assessmentId"`
	SemanticAssessment string                    `json:"semanticAssessment"`
	Result             auditstore.ExactArtifact  `json:"result"`
	DirectVerification bool                      `json:"directVerification"`
	Contract           *auditstore.ExactArtifact `json:"contract,omitempty"`
	AcceptedAt         time.Time                 `json:"acceptedAt"`
}

type reportFinding struct {
	FindingID          string                           `json:"findingId"`
	State              string                           `json:"state"`
	Revision           uint64                           `json:"revision"`
	FirstProposal      auditstore.ExactArtifact         `json:"firstProposal"`
	Title              string                           `json:"title"`
	Description        string                           `json:"description"`
	Subject            *auditdomain.FindingSubject      `json:"subject"`
	Locations          []auditdomain.FindingLocation    `json:"locations,omitempty"`
	HTTPExchange       *auditdomain.FindingHTTPExchange `json:"httpExchange,omitempty"`
	Hypothesis         string                           `json:"hypothesis,omitempty"`
	SeveritySuggestion string                           `json:"severitySuggestion,omitempty"`
	StandardRefs       []auditdomain.StandardReference  `json:"standardRefs"`
	Limitations        []string                         `json:"limitations"`
	Assessment         *reportFindingAssessment         `json:"assessment,omitempty"`
	AnalystDecision    *reportFindingDecision           `json:"analystDecision,omitempty"`
	DuplicateTargetID  *string                          `json:"duplicateTargetId,omitempty"`
}

// reportFindingSections holds each finding section as the JSON array of its
// reportFinding entries, assembled from the recorded report pages.
type reportFindingSections struct {
	Confirmed     json.RawMessage `json:"confirmed"`
	Proposed      json.RawMessage `json:"proposed"`
	Rejected      json.RawMessage `json:"rejected"`
	Duplicates    json.RawMessage `json:"duplicates"`
	NeedsEvidence json.RawMessage `json:"needsEvidence"`
}

// reportFindingCounts counts the findings of each report section.
type reportFindingCounts struct {
	confirmed, proposed, rejected, duplicates, needsEvidence int
}

func (c reportFindingCounts) total() int {
	return c.confirmed + c.proposed + c.rejected + c.duplicates + c.needsEvidence
}

type machineReport struct {
	Schema              string                                 `json:"schema"`
	AuditID             string                                 `json:"auditId"`
	ProjectID           string                                 `json:"projectId"`
	GeneratedFrom       time.Time                              `json:"generatedFrom"`
	Profile             reportProfile                          `json:"profile"`
	Baseline            reportBaseline                         `json:"baseline"`
	Round               reportRound                            `json:"round"`
	Limits              auditstore.Limits                      `json:"limits"`
	StopReason          *reportStopReason                      `json:"stopReason,omitempty"`
	AttemptDispositions auditstore.CollectionDispositionCounts `json:"attemptDispositions"`
	Coverage            reportCoverageSummary                  `json:"coverage"`
	// Items is the JSON array of reportItem entries in Round and item order.
	Items      json.RawMessage       `json:"items"`
	Findings   reportFindingSections `json:"findings"`
	Conclusion string                `json:"conclusion"`
}

// Finalize builds deterministic exact report artifacts and commits their two
// accepted links together with the final Audit transition. The report entries
// are recorded page by page under reportBudget: an attempt that reaches its
// budget returns true with its progress recorded, and the next attempt
// resumes from the recorded pages instead of rebuilding the report. A failed
// attempt returns false; the pages it recorded stay for the next attempt.
func (i *Importer) Finalize(
	ctx context.Context,
	claim auditstore.ControllerClaim,
	snapshot auditstore.ReconcileSnapshot,
) (bool, error) {
	if snapshot.Audit.AuditID != claim.AuditID || snapshot.Audit.State != auditstore.AuditFinalizing ||
		snapshot.Round == nil || snapshot.Round.State != auditstore.RoundClosed ||
		snapshot.Audit.OutstandingRunCount != 0 || len(snapshot.Items) != 0 || len(snapshot.Executions) != 0 ||
		snapshot.MoreItems || snapshot.MoreExecutions {
		return false, nil
	}
	profile, err := config.DecodeResolvedAuditProfileSnapshot(snapshot.Audit.ProfileSnapshot)
	if err != nil || profile.Ref.Name != snapshot.Audit.Profile.Name ||
		profile.Ref.Version != snapshot.Audit.Profile.Version || profile.Ref.Digest != snapshot.Audit.Profile.Digest {
		return false, fmt.Errorf("%w: report profile snapshot is invalid", ErrPermanent)
	}
	var baseline auditbaseline.ReportProjection
	if json.Unmarshal(snapshot.Audit.BaselineSnapshot, &baseline) != nil ||
		baseline.Schema != auditbaseline.Schema {
		return false, fmt.Errorf("%w: report baseline snapshot is invalid", ErrPermanent)
	}
	if profile.HasPreparation() && baseline.Inventory.Gaps != nil {
		return false, fmt.Errorf("%w: preparation mutated its original baseline", ErrPermanent)
	}
	if !profile.HasPreparation() && baseline.Inventory.Gaps == nil {
		return false, fmt.Errorf("%w: report inventory gaps are absent", ErrPermanent)
	}
	rounds, err := i.reads.ListRounds(ctx, snapshot.Audit.AuditID)
	if err != nil {
		return false, err
	}
	if len(rounds) == 0 || rounds[len(rounds)-1].RoundID != snapshot.Round.RoundID {
		return false, fmt.Errorf("%w: report round barrier is incomplete", ErrPermanent)
	}
	build, err := i.resumeReport(ctx, snapshot.Audit, rounds)
	if err != nil {
		return false, err
	}
	for recorded := 0; !build.complete(); recorded++ {
		if recorded == i.budget.pagesPerAttempt {
			return true, nil
		}
		if err := i.recordNextPage(ctx, claim, snapshot.Audit, &build); err != nil {
			if errors.Is(err, auditstore.ErrPrecondition) &&
				profile.Interaction.ReportAcceptance == config.AuditReportHumanRequired {
				return i.replayedProposal(ctx, snapshot.Audit, err)
			}
			return false, err
		}
	}
	if profile.HasPreparation() {
		link, err := i.reads.GetArtifactLink(ctx, snapshot.Audit.AuditID, auditstore.InitialInventoryLogicalKey)
		if err != nil {
			return false, err
		}
		data, err := i.artifacts.ReadProjectExact(ctx, snapshot.Audit.ProjectID, link.Artifact)
		if err != nil {
			return false, err
		}
		var derived auditbaseline.DerivedInventory
		if len(data) > auditstore.MaxSnapshotBytes || json.Unmarshal(data, &derived) != nil || derived.Schema != auditbaseline.DerivedInventorySchema ||
			!derived.Inventory.Worklist.Ref.SameExact(rounds[0].Manifest.Ref) || derived.Inventory.Worklist.Digest != rounds[0].Manifest.Digest ||
			!contentdigest.Valid(derived.Inventory.SourceContentDigest) || !contentdigest.Valid(derived.Inventory.CanonicalInventoryDigest) ||
			auditdomain.ValidateDispatchExecutionManifest(derived.Inventory.ExecutionManifest) != nil {
			return false, fmt.Errorf("%w: derived initial inventory is invalid", ErrPermanent)
		}
		if derived.Inventory.Gaps == nil {
			return false, fmt.Errorf("%w: report inventory gaps are absent", ErrPermanent)
		}
		baseline.Inventory = derived.Inventory
	}
	return i.publishReport(ctx, claim, snapshot, profile, baseline, build)
}

// replayedProposal recognizes a lost response of an earlier attempt that
// already froze this revision's candidate: proposing it left finalizing and
// discarded the build. A report build is deterministic for one Audit
// revision, so the candidate pinned to the snapshot revision is this
// attempt's candidate. Any other precondition failure is returned.
func (i *Importer) replayedProposal(ctx context.Context, audit auditstore.Audit, cause error) (bool, error) {
	candidate, err := i.reads.GetReportCandidate(ctx, audit.AuditID)
	if errors.Is(err, auditstore.ErrNotFound) {
		return false, cause
	}
	if err != nil {
		return false, err
	}
	if candidate.SubjectRevision != audit.Revision {
		return false, cause
	}
	return true, nil
}

// publishReport assembles the machine report from the recorded entries of a
// complete build, writes both report artifacts and commits or proposes them.
func (i *Importer) publishReport(
	ctx context.Context,
	claim auditstore.ControllerClaim,
	snapshot auditstore.ReconcileSnapshot,
	profile config.ResolvedAuditProfile,
	baseline auditbaseline.ReportProjection,
	build reportBuild,
) (bool, error) {
	items, findings, findingCounts, err := i.readReportSections(ctx, snapshot.Audit, build)
	if err != nil {
		return false, err
	}
	attempts, err := i.reads.CollectionDispositionCounts(ctx, snapshot.Audit.AuditID)
	if err != nil {
		return false, err
	}
	coverageSummary := summarizeCoverage(build.coverage, build.items)
	conclusion := "completed"
	if build.items == 0 || len(baseline.Inventory.Gaps) != 0 || hasIncompleteCoverage(coverageSummary.Counts) ||
		snapshot.Audit.StopReason != nil && snapshot.Audit.StopReason.Code != "round_complete" {
		conclusion = "completed-with-gaps"
	}
	var stopReason *reportStopReason
	if snapshot.Audit.StopReason != nil && snapshot.Audit.StopReason.Code != "round_complete" {
		stopReason = &reportStopReason{Code: snapshot.Audit.StopReason.Code, Message: snapshot.Audit.StopReason.Message}
	}
	report := machineReport{
		Schema: ReportSchema, AuditID: snapshot.Audit.AuditID, ProjectID: snapshot.Audit.ProjectID,
		GeneratedFrom: snapshot.Audit.UpdatedAt.UTC().Round(0),
		Profile: reportProfile{
			Name: snapshot.Audit.Profile.Name, Version: snapshot.Audit.Profile.Version,
			Digest: snapshot.Audit.Profile.Digest, Mode: profile.Mode,
			Standards:         append([]config.AuditStandardRef{}, profile.Standards...),
			StandardSelection: cloneReportStandardSelection(profile.Inventory.StandardSelection),
		},
		Baseline: reportBaseline{
			Digest: auditdomain.DigestBytes(snapshot.Audit.BaselineSnapshot),
			Inputs: cloneExactArtifactMap(baseline.Inputs), Scope: clone.Map(baseline.Scope),
			SourceContentDigest:      baseline.Inventory.SourceContentDigest,
			CanonicalInventoryDigest: baseline.Inventory.CanonicalInventoryDigest,
			Worklist:                 baseline.Inventory.Worklist,
			InventoryGaps:            append([]string{}, baseline.Inventory.Gaps...),
			Standards:                append([]auditstandards.PinnedPackage{}, baseline.Standards...),
			StandardSelection:        cloneReportStandardSelection(baseline.Inventory.StandardSelection),
		},
		Round: reportRound{
			RoundID: snapshot.Round.RoundID, Ordinal: snapshot.Round.Ordinal,
			Manifest: snapshot.Round.Manifest,
		},
		Limits: snapshot.Audit.Limits, StopReason: stopReason,
		AttemptDispositions: attempts, Coverage: coverageSummary,
		Items: items, Findings: findings, Conclusion: conclusion,
	}
	machineBytes, err := json.Marshal(report)
	if err != nil || len(machineBytes) > artifacts.MaxPayloadSize {
		return false, fmt.Errorf("%w: machine report exceeds its bound", ErrPermanent)
	}
	summaryBytes := []byte(humanSummary(report, findingCounts))
	if len(summaryBytes) > auditstore.MaxSummaryBytes {
		return false, fmt.Errorf("%w: human report exceeds its bound", ErrPermanent)
	}
	namespace := auditdomain.ArtifactNamespace(snapshot.Audit.AuditID)
	machineArtifact, err := i.artifacts.PutImmutableProject(
		ctx, snapshot.Audit.ProjectID,
		contracts.ArtifactRef{Namespace: namespace, Name: "report.json"},
		artifacts.Payload{MediaType: "application/json", Data: machineBytes},
	)
	if err != nil {
		return false, err
	}
	summaryArtifact, err := i.artifacts.PutImmutableProject(
		ctx, snapshot.Audit.ProjectID,
		contracts.ArtifactRef{Namespace: namespace, Name: "report.md"},
		artifacts.Payload{MediaType: "text/markdown", Data: summaryBytes},
	)
	if err != nil {
		return false, err
	}
	provenance, err := json.Marshal(struct {
		Schema         string                   `json:"schema"`
		AuditID        string                   `json:"auditId"`
		ProfileDigest  string                   `json:"profileDigest"`
		BaselineDigest string                   `json:"baselineDigest"`
		RoundManifest  auditstore.ExactArtifact `json:"roundManifest"`
	}{
		Schema: "contractor.audit.report-provenance.v1", AuditID: snapshot.Audit.AuditID,
		ProfileDigest: snapshot.Audit.Profile.Digest, BaselineDigest: report.Baseline.Digest,
		RoundManifest: snapshot.Round.Manifest,
	})
	if err != nil {
		return false, err
	}
	machineLink := auditstore.ArtifactLink{
		LogicalKey: auditstore.ReportMachineLogicalKey, Artifact: machineArtifact,
		SourceProvenance: provenance, DisplayRef: "machine-readable Audit report",
	}
	summaryLink := auditstore.ArtifactLink{
		LogicalKey: auditstore.ReportSummaryLogicalKey, Artifact: summaryArtifact,
		SourceProvenance: provenance, DisplayRef: "bounded human-readable Audit summary",
	}
	identity, err := json.Marshal(struct {
		Schema   string                   `json:"schema"`
		AuditID  string                   `json:"auditId"`
		Revision uint64                   `json:"revision"`
		Round    string                   `json:"round"`
		Machine  auditstore.ExactArtifact `json:"machine"`
		Summary  auditstore.ExactArtifact `json:"summary"`
	}{
		Schema: "contractor.audit.report-commit.v1", AuditID: snapshot.Audit.AuditID,
		Revision: snapshot.Audit.Revision, Round: snapshot.Round.RoundID,
		Machine: machineArtifact, Summary: summaryArtifact,
	})
	if err != nil {
		return false, err
	}
	commit := auditstore.CommitReportParams{
		Claim: claim, ExpectedAuditRevision: snapshot.Audit.Revision,
		RoundID: snapshot.Round.RoundID, ExpectedRoundRevision: snapshot.Round.Revision,
		Machine: machineLink, Summary: summaryLink, RequestDigest: auditdomain.DigestBytes(identity),
	}
	if profile.Interaction.ReportAcceptance == config.AuditReportHumanRequired {
		_, _, err = i.store.ProposeReport(ctx, auditstore.ProposeReportParams(commit))
		return err == nil, err
	}
	_, err = i.store.CommitReport(ctx, commit)
	return err == nil, err
}

// summarizeCoverage derives the report coverage summary from the per-status
// counts of the selected items.
func summarizeCoverage(counts auditstore.ReportCoverageCounts, selectedItems int) reportCoverageSummary {
	result := reportCoverageSummary{Counts: reportCoverageCounts(counts), SelectedItems: selectedItems}
	result.ApplicableDenominator = selectedItems - result.Counts.NotApplicable - result.Counts.Excluded
	result.AssessedApplicable = result.Counts.Satisfied + result.Counts.Violated + result.Counts.TracedComplete
	result.ZeroDenominator = result.ApplicableDenominator == 0
	if result.ApplicableDenominator != 0 {
		value := 100 * float64(result.AssessedApplicable) / float64(result.ApplicableDenominator)
		result.AssessedPercent = &value
	}
	return result
}

// coverageStatusCount is one coverage status with its report count.
type coverageStatusCount struct {
	status auditstore.CoverageStatus
	count  int
}

// byStatus pairs every coverage status with its count, in the order of the
// report's coverage table.
func (c reportCoverageCounts) byStatus() []coverageStatusCount {
	return []coverageStatusCount{
		{auditstore.CoverageSatisfied, c.Satisfied}, {auditstore.CoverageViolated, c.Violated},
		{auditstore.CoverageInconclusive, c.Inconclusive}, {auditstore.CoverageNotTested, c.NotTested},
		{auditstore.CoverageBlocked, c.Blocked}, {auditstore.CoverageNotApplicable, c.NotApplicable},
		{auditstore.CoverageExcluded, c.Excluded}, {auditstore.CoverageTracedComplete, c.TracedComplete},
		{auditstore.CoverageTracedPartial, c.TracedPartial}, {auditstore.CoverageUnmapped, c.Unmapped},
	}
}

// hasIncompleteCoverage reports whether any selected item is outside the
// completed coverage bucket, the classification the Audit workspace counts
// with: a gap or unchecked work makes the report completed-with-gaps.
func hasIncompleteCoverage(counts reportCoverageCounts) bool {
	for _, entry := range counts.byStatus() {
		if entry.count != 0 && entry.status.Bucket() != auditdomain.CoverageBucketCompleted {
			return true
		}
	}
	return false
}

func humanSummary(report machineReport, findings reportFindingCounts) string {
	const qualification = "\nCompletion describes the bounded Audit process; it is not a security or compliance certification.\n"
	const gapPrefix = "- Inventory gap: "
	var builder strings.Builder
	fmt.Fprintf(&builder, "# Audit %s\n\n", markdownText(report.AuditID))
	fmt.Fprintf(&builder, "- Profile: %s@%s (%s)\n", markdownText(report.Profile.Name), markdownText(report.Profile.Version), markdownText(string(report.Profile.Mode)))
	fmt.Fprintf(&builder, "- Conclusion: %s\n", markdownText(report.Conclusion))
	fmt.Fprintf(&builder, "- Selected items: %d\n", report.Coverage.SelectedItems)
	if selection := report.Baseline.StandardSelection; selection != nil {
		fmt.Fprintf(&builder, "- Standard selection: %s\n", markdownText(selection.Scope))
		fmt.Fprintf(&builder, "- Selected levels: %s\n", markdownText(strings.Join(selection.Levels, ", ")))
		fmt.Fprintf(&builder, "- Exact selected requirements: %d\n", len(selection.EntryIDs))
	}
	if report.Coverage.AssessedPercent == nil {
		builder.WriteString("- Applicable coverage: N/A (zero denominator)\n")
	} else {
		fmt.Fprintf(&builder, "- Applicable coverage: %.2f%%\n", *report.Coverage.AssessedPercent)
	}
	type countRow struct {
		label string
		count int
	}
	writeCounts := func(title string, rows []countRow) {
		fmt.Fprintf(&builder, "\n## %s\n\n| Status | Count |\n| --- | ---: |\n", title)
		for _, row := range rows {
			fmt.Fprintf(&builder, "| %s | %d |\n", row.label, row.count)
		}
	}
	coverageRows := []countRow{}
	for _, entry := range report.Coverage.Counts.byStatus() {
		coverageRows = append(coverageRows, countRow{string(entry.status), entry.count})
	}
	writeCounts("Coverage", coverageRows)
	writeCounts("Findings", []countRow{
		{"confirmed", findings.confirmed}, {"proposed", findings.proposed},
		{"rejected", findings.rejected}, {"duplicate", findings.duplicates},
		{"needs-evidence", findings.needsEvidence},
	})
	writeCounts("Attempts", []countRow{
		{"accepted", report.AttemptDispositions.AcceptedResult}, {"missing-output", report.AttemptDispositions.MissingOutput},
		{"invalid-result", report.AttemptDispositions.InvalidResult}, {"failed", report.AttemptDispositions.ExecutionFailed},
		{"cancelled", report.AttemptDispositions.ExecutionCancelled}, {"collection-contract-invalid", report.AttemptDispositions.ContractInvalid},
	})
	builder.WriteString("\n## Limitations\n\n")
	if report.StopReason != nil {
		fmt.Fprintf(&builder, "- Stop reason: %s — %s\n", markdownText(report.StopReason.Code), markdownText(report.StopReason.Message))
	}
	for index, gap := range report.Baseline.InventoryGaps {
		remaining := len(report.Baseline.InventoryGaps) - index - 1
		reserve := len(qualification)
		if remaining > 0 {
			reserve += len(omittedInventoryGaps(remaining))
		}
		// Escaping only grows the text. Skip an obviously oversized gap before
		// allocating its escaped representation.
		if builder.Len()+len(gapPrefix)+len(gap)+1+reserve <= auditstore.MaxSummaryBytes {
			escaped := markdownText(gap)
			if builder.Len()+len(gapPrefix)+len(escaped)+1+reserve <= auditstore.MaxSummaryBytes {
				builder.WriteString(gapPrefix)
				builder.WriteString(escaped)
				builder.WriteByte('\n')
				continue
			}
		}
		builder.WriteString(omittedInventoryGaps(len(report.Baseline.InventoryGaps) - index))
		break
	}
	builder.WriteString(qualification)
	return builder.String()
}

func omittedInventoryGaps(count int) string {
	return fmt.Sprintf("- … and %d more inventory gaps (see report.json)\n", count)
}

// Dynamic report fields are prose, never executable Markdown or raw HTML.
func markdownText(value string) string {
	return strings.NewReplacer(
		"&", "&amp;", "<", "&lt;", ">", "&gt;", "\\", "\\\\",
		"`", "\\`", "*", "\\*", "_", "\\_", "[", "\\[", "]", "\\]",
		"#", "\\#", "|", "\\|", "\r", " ", "\n", " ",
	).Replace(value)
}

func cloneExactArtifactMap(values map[string]auditstore.ExactArtifact) map[string]auditstore.ExactArtifact {
	result := make(map[string]auditstore.ExactArtifact, len(values))
	for key, value := range values {
		result[key] = value
	}
	return result
}

func cloneReportStandardSelection(
	source *config.AuditStandardSelection,
) *config.AuditStandardSelection {
	if source == nil {
		return nil
	}
	result := *source
	result.Levels = append([]string(nil), source.Levels...)
	result.EntryIDs = append([]string(nil), source.EntryIDs...)
	return &result
}
