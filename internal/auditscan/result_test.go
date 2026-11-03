package auditscan

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/scanplan"
)

// retainedScan is one terminal scan attempt whose plan and report the reader
// serves from memory. A ref without an entry is missing.
type retainedScan struct {
	input    ResultInput
	plan     contracts.ArtifactRef
	report   contracts.ArtifactRef
	payloads map[string]artifacts.Payload
	fail     map[string]error
}

func newRetainedScan(t *testing.T) *retainedScan {
	t.Helper()
	ref := func(namespace, name string) contracts.ArtifactRef {
		revision := name + "-r1"
		return contracts.ArtifactRef{Namespace: namespace, Name: name, Revision: &revision}
	}
	scan := &retainedScan{
		plan: ref("scan-results", "plan"), report: ref("scan-results", "report"),
		payloads: map[string]artifacts.Payload{}, fail: map[string]error{},
	}
	plan := []byte(`{"retained":"plan"}`)
	scan.input = ResultInput{
		RunID: "run-1", CurrentStage: "stage-2", Scanner: "sqlmap", Runnable: true,
		Attempts: []planner.ScanAttempt{{
			StageExecutionID: "stage-1", StageName: "scan", Terminal: true,
			State: planner.ScanState{
				Plan: &scan.plan, PlanDigest: contentdigest.Bytes(plan),
				Jobs: []planner.ScanJobRecord{{
					ID: "job-1", Worker: "scanner", Status: planner.ScanJobCompleted,
					InputArtifacts: map[string]contracts.ArtifactRef{}, Report: &scan.report,
				}},
			},
		}},
	}
	scan.payloads[scan.report.Name] = artifacts.Payload{
		MediaType: auditdomain.JSONMediaType, Data: []byte(`{"tool":"scan_sqlmap","status":"completed"}`),
	}
	return scan
}

func (s *retainedScan) read(ctx context.Context, ref contracts.ArtifactRef, _ int) (artifacts.Payload, error) {
	if err := ctx.Err(); err != nil {
		return artifacts.Payload{}, err
	}
	if err := s.fail[ref.Name]; err != nil {
		return artifacts.Payload{}, err
	}
	payload, exists := s.payloads[ref.Name]
	if !exists {
		return artifacts.Payload{}, artifacts.ErrArtifactNotFound
	}
	return payload, nil
}

func (s *retainedScan) build(t *testing.T) (operationReport, error) {
	t.Helper()
	result, err := BuildResult(t.Context(), s.input, s.read)
	if err != nil {
		return operationReport{}, err
	}
	var report operationReport
	err = json.Unmarshal(result.Data, &report)
	if err != nil {
		t.Fatalf("assembled report is invalid: %v", err)
	}
	return report, nil
}

// Missing retained artifacts stay coverage gaps of a valid result: the scan
// may have run, but its evidence is unavailable.
func TestBuildResultKeepsMissingRetainedArtifactsAsCoverageGaps(t *testing.T) {
	scan := newRetainedScan(t)
	delete(scan.payloads, scan.report.Name)
	result, err := scan.build(t)
	if err != nil {
		t.Fatal(err)
	}
	for _, gap := range []string{"scan_plan_unavailable", "scan_report_unavailable"} {
		if !slices.Contains(result.Limitations, gap) {
			t.Fatalf("gaps = %v, want %s", result.Limitations, gap)
		}
	}
	if result.ExecutionCompleted {
		t.Fatalf("completed coverage without a report: %+v", result)
	}
}

// A read failure and cancellation leave the result buildable later, so
// BuildResult returns them unchanged and never as ErrInvalid.
func TestBuildResultReturnsReadFailuresUnchanged(t *testing.T) {
	transient := errors.New("artifact storage unavailable")
	for _, name := range []string{"plan", "report"} {
		t.Run(name, func(t *testing.T) {
			scan := newRetainedScan(t)
			scan.fail[name] = transient
			if _, err := scan.build(t); !errors.Is(err, transient) || errors.Is(err, ErrInvalid) {
				t.Fatalf("read failure = %v, want it unchanged", err)
			}
		})
	}
	t.Run("cancelled", func(t *testing.T) {
		scan := newRetainedScan(t)
		ctx, cancel := context.WithCancel(t.Context())
		cancel()
		if _, err := BuildResult(ctx, scan.input, scan.read); !errors.Is(err, context.Canceled) || errors.Is(err, ErrInvalid) {
			t.Fatalf("cancelled read = %v, want context.Canceled unchanged", err)
		}
	})
}

// Exact retained content and the pinned assignment never change, so content
// that can never build a result is ErrInvalid instead of an endless retry.
func TestBuildResultClassifiesInvalidRetainedContentAndAssignment(t *testing.T) {
	cases := map[string]func(*retainedScan){
		"plan that does not decode": func(s *retainedScan) {
			data := []byte(`{"retained":"plan"}`)
			s.payloads[s.plan.Name] = artifacts.Payload{MediaType: scanplan.PlanMediaType, Data: data}
		},
		"plan over its bound": func(s *retainedScan) {
			data := bytes.Repeat([]byte(" "), scanplan.MaxPlanBytes+1)
			s.input.Attempts[0].State.PlanDigest = contentdigest.Bytes(data)
			s.payloads[s.plan.Name] = artifacts.Payload{MediaType: scanplan.PlanMediaType, Data: data}
		},
		"report of another scanner": func(s *retainedScan) {
			s.payloads[s.report.Name] = artifacts.Payload{MediaType: auditdomain.JSONMediaType, Data: []byte(`{"tool":"scan_nuclei"}`)}
		},
		"report with another media type": func(s *retainedScan) {
			s.payloads[s.report.Name] = artifacts.Payload{MediaType: "text/plain", Data: []byte(`{"tool":"scan_sqlmap"}`)}
		},
		"report over its bound": func(s *retainedScan) {
			data := append([]byte(`{"tool":"scan_sqlmap","padding":"`), bytes.Repeat([]byte("x"), MaxScannerReportBytes)...)
			s.payloads[s.report.Name] = artifacts.Payload{MediaType: auditdomain.JSONMediaType, Data: append(data, '"', '}')}
		},
		"invalid Run identity": func(s *retainedScan) { s.input.RunID = "bad identity" },
		"unknown scanner":      func(s *retainedScan) { s.input.Scanner = "other" },
	}
	for name, change := range cases {
		t.Run(name, func(t *testing.T) {
			scan := newRetainedScan(t)
			change(scan)
			if _, err := scan.build(t); !errors.Is(err, ErrInvalid) {
				t.Fatalf("invalid input = %v, want ErrInvalid", err)
			}
		})
	}
}

func TestBuildResultRetainsReadableReportAsCompletedCoverage(t *testing.T) {
	scan := newRetainedScan(t)
	result, err := scan.build(t)
	if err != nil {
		t.Fatal(err)
	}
	if !result.ExecutionCompleted || len(result.Artifacts) != 1 ||
		!slices.Contains(result.Limitations, "scan_plan_unavailable") {
		t.Fatalf("readable report result = %+v", result)
	}
}

func TestBuildResultUnknownOutcomeDoesNotAuthorizeTrafficRetry(t *testing.T) {
	for _, status := range []string{planner.ScanJobStarted, planner.ScanJobUnknown, planner.ScanJobCompleted, planner.ScanJobFailed} {
		t.Run(string(status), func(t *testing.T) {
			scan := newRetainedScan(t)
			attempt := &scan.input.Attempts[0]
			scan.input.CurrentStage = attempt.StageExecutionID
			attempt.State.Jobs[0].Status = status
			if status != planner.ScanJobCompleted {
				attempt.State.Jobs[0].Report = nil
			}
			result, err := BuildResult(t.Context(), scan.input, scan.read)
			if err != nil {
				t.Fatal(err)
			}
			if result.Retryable != (status == planner.ScanJobFailed) {
				t.Fatalf("status %s retryable=%t", status, result.Retryable)
			}
			var report operationReport
			if err := json.Unmarshal(result.Data, &report); err != nil {
				t.Fatal(err)
			}
			if (status == planner.ScanJobStarted || status == planner.ScanJobUnknown) && !slices.Contains(report.Limitations, "scan_outcome_unknown") {
				t.Fatal("lost ambiguous outcome")
			}
		})
	}
}
func TestBuildResultBoundsEmbeddedEvidenceAndRemainsSelfContained(t *testing.T) {
	scan := newRetainedScan(t)
	scan.payloads[scan.report.Name] = artifacts.Payload{MediaType: "application/json", Data: append(append([]byte(`{"tool":"scan_sqlmap","observations":"`), bytes.Repeat([]byte("x"), MaxScannerReportBytes-100)...), []byte(`"}`)...)}
	for i := 0; i < 5; i++ {
		job := scan.input.Attempts[0].State.Jobs[0]
		job.ID = fmt.Sprintf("job-%d", i)
		scan.input.Attempts[0].State.Jobs = append(scan.input.Attempts[0].State.Jobs, job)
	}
	result, err := BuildResult(t.Context(), scan.input, scan.read)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Data) > MaxResultBytes {
		t.Fatal("unbounded report")
	}
	var report operationReport
	if err := json.Unmarshal(result.Data, &report); err != nil {
		t.Fatal(err)
	}
	if len(report.Artifacts) != 2 || !slices.Contains(report.Limitations, "scan_evidence_budget_exceeded") {
		t.Fatalf("evidence count=%d gaps=%v", len(report.Artifacts), report.Limitations)
	}
	for _, artifact := range report.Artifacts {
		if contentdigest.Bytes(artifact.Data) != artifact.Digest {
			t.Fatal("embedded exact bytes changed")
		}
	}
}
