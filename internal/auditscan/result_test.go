package auditscan

import (
	"bytes"
	"context"
	"errors"
	"os"
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
	read := func(name string) []byte {
		t.Helper()
		data, err := os.ReadFile("../../testdata/configs-scan/audit-openapi-scan/" + name)
		if err != nil {
			t.Fatal(err)
		}
		return data
	}
	ref := func(namespace, name string) contracts.ArtifactRef {
		revision := name + "-r1"
		return contracts.ArtifactRef{Namespace: namespace, Name: name, Revision: &revision}
	}
	source, settings := read("openapi.json"), read("sqlmap-settings.json")
	inventory, err := auditdomain.BuildOpenAPIScanInventory(source, auditdomain.JSONMediaType, settings,
		auditdomain.ExactInput{Name: "scan_options", Ref: ref("project", "scan-options"), Digest: contentdigest.Bytes(settings)},
		auditdomain.InventoryOptions{
			Round: 1, WorkflowRole: "check", SourceInputName: "api_document",
			SourceRef: ref("project", "api-document"), ApprovalRequirement: auditdomain.ApprovalActiveCheck,
		})
	if err != nil {
		t.Fatal(err)
	}
	manifest := inventory.ExecutionManifest
	taskRef := ref("project", "assigned-task")
	manifest.Items[0].TaskRef = &taskRef
	scan := &retainedScan{
		plan: ref("scan-results", "plan"), report: ref("scan-results", "report"),
		payloads: map[string]artifacts.Payload{}, fail: map[string]error{},
	}
	plan := []byte(`{"retained":"plan"}`)
	scan.input = ResultInput{
		RunID: "run-1", CurrentStage: "stage-2", Task: inventory.Tasks[0].Document, Manifest: manifest,
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

func (s *retainedScan) build(t *testing.T) (auditdomain.CheckResult, error) {
	t.Helper()
	result, err := BuildResult(t.Context(), s.input, s.read)
	if err != nil {
		return auditdomain.CheckResult{}, err
	}
	pkg, err := auditdomain.DecodeCheckResultPackage(result.Package)
	if err != nil {
		t.Fatalf("assembled package is invalid: %v", err)
	}
	return pkg.Results.Results[0], nil
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
		if !slices.Contains(result.Coverage.Gaps, gap) {
			t.Fatalf("gaps = %v, want %s", result.Coverage.Gaps, gap)
		}
	}
	if len(result.Coverage.Completed) != 0 {
		t.Fatalf("completed coverage without a report: %+v", result.Coverage)
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
		"assignment that differs from its manifest": func(s *retainedScan) {
			s.input.Manifest.Items[0].SubjectKey += "-changed"
		},
		"manifest without a task reference": func(s *retainedScan) {
			s.input.Manifest.Items[0].TaskRef = nil
		},
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
	if len(result.Coverage.Completed) != 1 || result.Assessment != "inconclusive" ||
		!slices.Contains(result.Coverage.Gaps, "scan_plan_unavailable") {
		t.Fatalf("readable report result = %+v", result)
	}
}
