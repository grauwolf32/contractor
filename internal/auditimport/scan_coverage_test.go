package auditimport

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
)

func TestScanCoverageSeparatesExecutionFromSecurityVerdict(t *testing.T) {
	for _, scanner := range []string{"sqlmap", "nuclei"} {
		t.Run(scanner, func(t *testing.T) {
			task := auditdomain.ItemTask{Scan: &auditdomain.OpenAPIScanTask{Scanner: scanner, Runnable: true, Gaps: []string{}}}
			if scanner == "nuclei" {
				task.Scan.Gaps = []string{"http_method_not_replayed", "url_template_scan_only"}
			}
			result := auditdomain.CheckResult{Assessment: "inconclusive", EvidenceIDs: []string{"report"}, Coverage: auditdomain.ResultCoverage{
				Requested: expectedCoverage(task), Completed: expectedCoverage(task), Gaps: []string{},
			}}
			evidence := map[string]validatedEvidence{"report": {value: auditdomain.Evidence{Kind: "scanner-report"}}}
			coverage, err := semanticCoverage(config.AuditModeRiskAssessment, task, result, evidence)
			if err != nil || coverage.Status != auditstore.CoverageInconclusive || !equalStrings(coverage.Completed, expectedCoverage(task)) || !equalStrings(coverage.Gaps, task.Scan.Gaps) {
				t.Fatalf("wrong scanner coverage: %+v, %v", coverage, err)
			}
			for _, assessment := range []string{"satisfied", "violated", "supported", "refuted", "not-applicable", "not-tested", "blocked"} {
				result.Assessment = assessment
				if _, err := semanticCoverage(config.AuditModeRiskAssessment, task, result, evidence); err == nil {
					t.Fatalf("completed scanner promoted to %s", assessment)
				}
			}
			result.Assessment = "inconclusive"
			if _, err := semanticCoverage(config.AuditModeRiskAssessment, task, result, nil); err == nil {
				t.Fatal("completed without retained report")
			}
			evidence["report"] = validatedEvidence{value: auditdomain.Evidence{Kind: "source"}}
			if _, err := semanticCoverage(config.AuditModeRiskAssessment, task, result, evidence); err == nil {
				t.Fatal("non-scanner evidence satisfied scan")
			}
			result.Coverage.Requested = []string{"operation-resolution"}
			if _, err := semanticCoverage(config.AuditModeRiskAssessment, task, result, evidence); err == nil {
				t.Fatal("scan counted as operation tracing")
			}
		})
	}
}

func TestScanCoverageKeepsPreparationAndExecutionGaps(t *testing.T) {
	task := auditdomain.ItemTask{Scan: &auditdomain.OpenAPIScanTask{Scanner: "nuclei", Runnable: false, Gaps: []string{"missing_required_parameter", "url_template_scan_only"}}}
	baseline := baselineCoverage(task)
	if baseline.Status != auditstore.CoverageNotTested || !equalStrings(baseline.Requested, []string{"nuclei-url-template-scan"}) || !equalStrings(baseline.Gaps, task.Scan.Gaps) {
		t.Fatalf("lost baseline: %+v", baseline)
	}
	result := auditdomain.CheckResult{Assessment: "not-tested", EvidenceIDs: []string{}, Coverage: auditdomain.ResultCoverage{Requested: expectedCoverage(task), Completed: []string{}, Gaps: []string{}}}
	coverage, err := semanticCoverage(config.AuditModeRiskAssessment, task, result, nil)
	if err != nil || coverage.Status != auditstore.CoverageNotTested || !equalStrings(coverage.Gaps, task.Scan.Gaps) {
		t.Fatalf("preparation gap lost: %+v, %v", coverage, err)
	}
	result.Assessment = "blocked"
	result.Coverage.Gaps = []string{"scan_outcome_unknown"}
	coverage, err = semanticCoverage(config.AuditModeRiskAssessment, task, result, nil)
	if err != nil || coverage.Status != auditstore.CoverageBlocked || !equalStrings(coverage.Gaps, []string{"missing_required_parameter", "scan_outcome_unknown", "url_template_scan_only"}) {
		t.Fatalf("execution gap lost: %+v, %v", coverage, err)
	}
	result.Assessment = "inconclusive"
	result.EvidenceIDs = []string{"report"}
	result.Coverage.Completed = expectedCoverage(task)
	if _, err := semanticCoverage(config.AuditModeRiskAssessment, task, result, map[string]validatedEvidence{"report": {value: auditdomain.Evidence{Kind: "scanner-report"}}}); err == nil {
		t.Fatal("unprepared scan counted as completed")
	}
	origin, err := encodeTaskOrigin(task)
	if err != nil || !strings.Contains(string(origin), `"scanner":"nuclei"`) || !json.Valid(origin) {
		t.Fatalf("lost scan provenance: %s, %v", origin, err)
	}
}
