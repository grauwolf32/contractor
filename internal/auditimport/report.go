package auditimport

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
)

const ReportSchema = "contractor.audit.report.v1"

type reportProfile struct {
	Name      string                    `json:"name"`
	Version   string                    `json:"version"`
	Digest    string                    `json:"digest"`
	Mode      config.AuditProfileMode   `json:"mode"`
	Standards []config.AuditStandardRef `json:"standards"`
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
	FindingID          string                          `json:"findingId"`
	State              string                          `json:"state"`
	Revision           uint64                          `json:"revision"`
	FirstProposal      auditstore.ExactArtifact        `json:"firstProposal"`
	Title              string                          `json:"title"`
	Description        string                          `json:"description"`
	Subject            auditdomain.FindingSubject      `json:"subject"`
	Hypothesis         string                          `json:"hypothesis,omitempty"`
	SeveritySuggestion string                          `json:"severitySuggestion,omitempty"`
	StandardRefs       []auditdomain.StandardReference `json:"standardRefs"`
	Limitations        []string                        `json:"limitations"`
	Assessment         *reportFindingAssessment        `json:"assessment,omitempty"`
	AnalystDecision    *reportFindingDecision          `json:"analystDecision,omitempty"`
	DuplicateTargetID  *string                         `json:"duplicateTargetId,omitempty"`
}

type reportFindingSections struct {
	Confirmed     []reportFinding `json:"confirmed"`
	Proposed      []reportFinding `json:"proposed"`
	Rejected      []reportFinding `json:"rejected"`
	Duplicates    []reportFinding `json:"duplicates"`
	NeedsEvidence []reportFinding `json:"needsEvidence"`
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
	Items               []reportItem                           `json:"items"`
	Findings            reportFindingSections                  `json:"findings"`
	Conclusion          string                                 `json:"conclusion"`
}

// Finalize builds deterministic exact report artifacts and commits their two
// accepted links together with the final Audit transition.
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
	var baseline struct {
		Schema    string                              `json:"schema"`
		Inputs    map[string]auditstore.ExactArtifact `json:"inputs"`
		Scope     map[string]string                   `json:"scope"`
		Standards []auditstandards.PinnedPackage      `json:"standards"`
		Inventory struct {
			SourceContentDigest      string                   `json:"sourceContentDigest"`
			CanonicalInventoryDigest string                   `json:"canonicalInventoryDigest"`
			Worklist                 auditstore.ExactArtifact `json:"worklist"`
			Gaps                     []string                 `json:"gaps"`
		} `json:"inventory"`
	}
	if json.Unmarshal(snapshot.Audit.BaselineSnapshot, &baseline) != nil ||
		baseline.Schema != "contractor.audit.baseline.v1" || baseline.Inventory.Gaps == nil {
		return false, fmt.Errorf("%w: report baseline snapshot is invalid", ErrPermanent)
	}
	items, err := i.store.ListItems(ctx, snapshot.Audit.AuditID)
	if err != nil {
		return false, err
	}
	coverage, err := i.allCoverage(ctx, snapshot.Audit.AuditID, snapshot.Round.RoundID)
	if err != nil {
		return false, err
	}
	if len(items) != snapshot.Round.ExpectedItemCount || len(coverage) != len(items) {
		return false, fmt.Errorf("%w: report coverage barrier is incomplete", ErrPermanent)
	}
	rowsByItem := make(map[string]auditstore.CoverageRow, len(coverage))
	for _, row := range coverage {
		rowsByItem[row.ItemID] = row
	}
	reportItems := make([]reportItem, len(items))
	for index, item := range items {
		row, exists := rowsByItem[item.ItemID]
		if !exists || item.State != auditstore.ItemSettled || item.FinalDisposition == nil {
			return false, fmt.Errorf("%w: report item barrier is incomplete", ErrPermanent)
		}
		reportItems[index] = reportItem{
			ItemID: item.ItemID, ItemKey: item.ItemKey, Ordinal: item.Ordinal,
			Kind: item.Kind, SubjectKey: item.SubjectKey,
			FinalDisposition: *item.FinalDisposition, Coverage: row.Coverage,
			Result: row.Result, Task: item.Task,
		}
	}
	sort.Slice(reportItems, func(left, right int) bool {
		return reportItems[left].Ordinal < reportItems[right].Ordinal
	})
	attempts, err := i.store.CollectionDispositionCounts(ctx, snapshot.Audit.AuditID)
	if err != nil {
		return false, err
	}
	findings, err := i.reportFindings(ctx, snapshot.Audit)
	if err != nil {
		return false, err
	}
	coverageSummary := summarizeCoverage(coverage)
	conclusion := "completed"
	if len(items) == 0 || len(baseline.Inventory.Gaps) != 0 || hasIncompleteCoverage(coverageSummary.Counts) ||
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
			Standards: append([]config.AuditStandardRef{}, profile.Standards...),
		},
		Baseline: reportBaseline{
			Digest: digestBytes(snapshot.Audit.BaselineSnapshot),
			Inputs: cloneExactArtifactMap(baseline.Inputs), Scope: cloneStringMap(baseline.Scope),
			SourceContentDigest:      baseline.Inventory.SourceContentDigest,
			CanonicalInventoryDigest: baseline.Inventory.CanonicalInventoryDigest,
			Worklist:                 baseline.Inventory.Worklist,
			InventoryGaps:            append([]string{}, baseline.Inventory.Gaps...),
			Standards:                append([]auditstandards.PinnedPackage{}, baseline.Standards...),
		},
		Round: reportRound{
			RoundID: snapshot.Round.RoundID, Ordinal: snapshot.Round.Ordinal,
			Manifest: snapshot.Round.Manifest,
		},
		Limits: snapshot.Audit.Limits, StopReason: stopReason,
		AttemptDispositions: attempts, Coverage: coverageSummary,
		Items: reportItems, Findings: findings, Conclusion: conclusion,
	}
	machineBytes, err := json.Marshal(report)
	if err != nil || len(machineBytes) > artifacts.MaxPayloadSize {
		return false, fmt.Errorf("%w: machine report exceeds its bound", ErrPermanent)
	}
	summaryBytes := []byte(humanSummary(report))
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
		contracts.ArtifactRef{Namespace: namespace, Name: "report.txt"},
		artifacts.Payload{MediaType: "text/plain", Data: summaryBytes},
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
		Machine: machineLink, Summary: summaryLink, RequestDigest: digestBytes(identity),
	}
	if profile.Interaction.ReportAcceptance == config.AuditReportHumanRequired {
		_, _, err = i.store.ProposeReport(ctx, auditstore.ProposeReportParams(commit))
		return err == nil, err
	}
	_, err = i.store.CommitReport(ctx, commit)
	return err == nil, err
}

func (i *Importer) allCoverage(
	ctx context.Context, auditID, roundID string,
) ([]auditstore.CoverageRow, error) {
	result := make([]auditstore.CoverageRow, 0)
	after := -1
	for {
		page, err := i.store.ListCoverage(ctx, auditID, roundID, after, auditstore.MaxPageSize)
		if err != nil {
			return nil, err
		}
		result = append(result, page...)
		if len(page) < auditstore.MaxPageSize {
			return result, nil
		}
		after = page[len(page)-1].Ordinal
	}
}

func (i *Importer) reportFindings(
	ctx context.Context, audit auditstore.Audit,
) (reportFindingSections, error) {
	rows, err := i.store.ListReportFindings(ctx, audit.AuditID)
	if err != nil {
		return reportFindingSections{}, err
	}
	result := reportFindingSections{
		Confirmed: []reportFinding{}, Proposed: []reportFinding{}, Rejected: []reportFinding{},
		Duplicates: []reportFinding{}, NeedsEvidence: []reportFinding{},
	}
	for _, row := range rows {
		payload, err := i.artifacts.ReadProjectExact(ctx, audit.ProjectID, row.FirstProposal)
		if err != nil {
			return reportFindingSections{}, err
		}
		document, err := auditdomain.DecodeFindingProposal(payload)
		if err != nil {
			return reportFindingSections{}, fmt.Errorf("%w: report finding proposal is invalid", ErrPermanent)
		}
		value := reportFinding{
			FindingID: row.FindingID, State: row.State, Revision: row.Revision,
			FirstProposal: row.FirstProposal, Title: document.Title, Description: document.Description,
			Subject: document.Subject, Hypothesis: document.Hypothesis,
			SeveritySuggestion: document.SeveritySuggestion,
			StandardRefs:       append([]auditdomain.StandardReference{}, document.StandardRefs...),
			Limitations:        append([]string{}, document.Limitations...),
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
		switch row.State {
		case "confirmed":
			result.Confirmed = append(result.Confirmed, value)
		case "proposed":
			result.Proposed = append(result.Proposed, value)
		case "rejected":
			result.Rejected = append(result.Rejected, value)
		case "duplicate":
			result.Duplicates = append(result.Duplicates, value)
		case "needs-evidence":
			result.NeedsEvidence = append(result.NeedsEvidence, value)
		default:
			return reportFindingSections{}, fmt.Errorf("%w: report finding state is invalid", ErrPermanent)
		}
	}
	return result, nil
}

func summarizeCoverage(rows []auditstore.CoverageRow) reportCoverageSummary {
	result := reportCoverageSummary{SelectedItems: len(rows)}
	for _, row := range rows {
		switch row.Coverage.Status {
		case auditstore.CoverageNotTested:
			result.Counts.NotTested++
		case auditstore.CoverageInconclusive:
			result.Counts.Inconclusive++
		case auditstore.CoverageSatisfied:
			result.Counts.Satisfied++
		case auditstore.CoverageViolated:
			result.Counts.Violated++
		case auditstore.CoverageNotApplicable:
			result.Counts.NotApplicable++
		case auditstore.CoverageBlocked:
			result.Counts.Blocked++
		case auditstore.CoverageExcluded:
			result.Counts.Excluded++
		case auditstore.CoverageTracedComplete:
			result.Counts.TracedComplete++
		case auditstore.CoverageTracedPartial:
			result.Counts.TracedPartial++
		case auditstore.CoverageUnmapped:
			result.Counts.Unmapped++
		}
	}
	result.ApplicableDenominator = len(rows) - result.Counts.NotApplicable - result.Counts.Excluded
	result.AssessedApplicable = result.Counts.Satisfied + result.Counts.Violated + result.Counts.TracedComplete
	result.ZeroDenominator = result.ApplicableDenominator == 0
	if result.ApplicableDenominator != 0 {
		value := 100 * float64(result.AssessedApplicable) / float64(result.ApplicableDenominator)
		result.AssessedPercent = &value
	}
	return result
}

func hasIncompleteCoverage(counts reportCoverageCounts) bool {
	return counts.NotTested != 0 || counts.Inconclusive != 0 || counts.Blocked != 0 ||
		counts.Excluded != 0 || counts.TracedPartial != 0 || counts.Unmapped != 0
}

func humanSummary(report machineReport) string {
	var builder strings.Builder
	fmt.Fprintf(&builder, "Audit %s\n", report.AuditID)
	fmt.Fprintf(&builder, "Profile: %s@%s (%s)\n", report.Profile.Name, report.Profile.Version, report.Profile.Mode)
	fmt.Fprintf(&builder, "Conclusion: %s\n", report.Conclusion)
	fmt.Fprintf(&builder, "Selected items: %d\n", report.Coverage.SelectedItems)
	fmt.Fprintf(
		&builder,
		"Coverage: satisfied=%d violated=%d inconclusive=%d not-tested=%d blocked=%d not-applicable=%d excluded=%d traced-complete=%d traced-partial=%d unmapped=%d\n",
		report.Coverage.Counts.Satisfied, report.Coverage.Counts.Violated,
		report.Coverage.Counts.Inconclusive, report.Coverage.Counts.NotTested,
		report.Coverage.Counts.Blocked, report.Coverage.Counts.NotApplicable,
		report.Coverage.Counts.Excluded, report.Coverage.Counts.TracedComplete,
		report.Coverage.Counts.TracedPartial, report.Coverage.Counts.Unmapped,
	)
	fmt.Fprintf(
		&builder,
		"Findings: confirmed=%d proposed=%d rejected=%d duplicate=%d needs-evidence=%d\n",
		len(report.Findings.Confirmed), len(report.Findings.Proposed),
		len(report.Findings.Rejected), len(report.Findings.Duplicates),
		len(report.Findings.NeedsEvidence),
	)
	if report.Coverage.AssessedPercent == nil {
		builder.WriteString("Applicable coverage: N/A (zero denominator)\n")
	} else {
		fmt.Fprintf(&builder, "Applicable coverage: %.2f%%\n", *report.Coverage.AssessedPercent)
	}
	fmt.Fprintf(
		&builder,
		"Attempts: accepted=%d missing-output=%d invalid-result=%d failed=%d cancelled=%d collection-contract-invalid=%d\n",
		report.AttemptDispositions.AcceptedResult, report.AttemptDispositions.MissingOutput,
		report.AttemptDispositions.InvalidResult, report.AttemptDispositions.ExecutionFailed,
		report.AttemptDispositions.ExecutionCancelled,
		report.AttemptDispositions.ContractInvalid,
	)
	if report.StopReason != nil {
		fmt.Fprintf(&builder, "Stop reason: %s — %s\n", report.StopReason.Code, report.StopReason.Message)
	}
	if len(report.Baseline.InventoryGaps) != 0 {
		fmt.Fprintf(&builder, "Inventory gaps: %s\n", strings.Join(report.Baseline.InventoryGaps, ", "))
	}
	builder.WriteString("Completion describes the bounded Audit process; it is not a security or compliance certification.\n")
	return builder.String()
}

func cloneExactArtifactMap(values map[string]auditstore.ExactArtifact) map[string]auditstore.ExactArtifact {
	result := make(map[string]auditstore.ExactArtifact, len(values))
	for key, value := range values {
		result[key] = value
	}
	return result
}

func cloneStringMap(values map[string]string) map[string]string {
	result := make(map[string]string, len(values))
	for key, value := range values {
		result[key] = value
	}
	return result
}
