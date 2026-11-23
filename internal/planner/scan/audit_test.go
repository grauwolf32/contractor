package scan

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	contractscan "github.com/grauwolf32/contractor/internal/contracts/scan"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/scanplan"
	"github.com/grauwolf32/contractor/internal/stageresult"
)

type auditFactoryHistory struct {
	*factorySessions
	stageID string
	prior   []planner.ScanAttempt
	err     error
}

func (s *auditFactoryHistory) ReadAuditScanHistory(context.Context, string) ([]planner.ScanAttempt, error) {
	if s.err != nil {
		return nil, s.err
	}
	history := cloneFactoryValue(s.prior)
	return append(history, planner.ScanAttempt{StageExecutionID: s.stageID, StageName: "execute", State: cloneFactoryValue(s.state)}), nil
}

func newAuditFactoryHarness(t *testing.T, scanner string) (*factoryHarness, *auditFactoryHistory) {
	t.Helper()
	h := newFactoryHarness(t, 1)
	snapshot, _, err := configload.LoadUnionReadOnly("../../../testdata/configs-scan", filepath.Join(t.TempDir(), "managed"))
	if err != nil {
		t.Fatal(err)
	}
	selector := "audit-openapi-" + scanner + "-scan@1"
	if scanner == "web" {
		selector = "audit-web-nuclei-scan@1"
	}
	workflow, err := snapshot.Workflow(selector)
	if err != nil {
		t.Fatal(err)
	}
	h.invocation.Stage = workflow.Stages[workflow.EntryStage]
	h.invocation.Context.Artifacts = map[string]*contracts.ArtifactRef{}
	h.invocation.Workers = map[string]contracts.WorkerHandle{}
	for name, binding := range h.invocation.Stage.Agents {
		h.invocation.Workers[name] = contracts.WorkerHandle{
			AllocationID: "allocation-" + name, AgentTemplateRef: binding.Template.Ref, WorkerRuntimeRef: binding.Template.Runtime,
			AgentCard: map[string]any{"name": name}, LeaseExpiresAt: h.invocation.Deadline,
		}
	}
	read := func(name string) []byte {
		t.Helper()
		data, err := os.ReadFile("../../../testdata/configs-scan/audit-openapi-scan/" + name)
		if err != nil {
			t.Fatal(err)
		}
		return data
	}
	var source, values []byte
	h.invocation.Context.Parameters = map[string]string{}
	sourceSlot := "openapi"
	if scanner == "web" {
		var readErr error
		source, readErr = os.ReadFile("../../../testdata/configs-scan/audit-web-scan/target.json")
		if readErr != nil {
			t.Fatal(readErr)
		}
		parameters, readErr := os.ReadFile("../../../testdata/configs-scan/audit-web-scan/parameters.json")
		if readErr != nil {
			t.Fatal(readErr)
		}
		if err := json.Unmarshal(parameters, &h.invocation.Context.Parameters); err != nil {
			t.Fatal(err)
		}
		sourceSlot = "target"
	} else {
		source, values = read("openapi.json"), read(scanner+"-request-values.json")
		if err := json.Unmarshal(read(scanner+"-parameters.json"), &h.invocation.Context.Parameters); err != nil {
			t.Fatal(err)
		}
	}

	material := func(slot string, data []byte) auditcontract.Material {
		ref := h.artifacts.put(contracts.ArtifactRef{Namespace: "inputs", Name: slot}, artifacts.Payload{MediaType: auditdomain.JSONMediaType, Data: data})
		h.invocation.Context.Artifacts[slot] = &ref
		return auditcontract.Material{ScopeKind: "run", ScopeID: h.invocation.RunID, Ref: ref, Digest: auditdomain.DigestBytes(data), MediaType: auditdomain.JSONMediaType, SizeBytes: int64(len(data))}
	}
	sourceMaterial := material(sourceSlot, source)
	if scanner != "web" {
		valuesMaterial := material("request_values", values)
		prepared, err := auditdomain.BuildPreparedOpenAPIWork(t.Context(), auditdomain.PreparedOpenAPIInput{Source: sourceMaterial, Data: source, Values: &valuesMaterial, ValuesData: values, Options: auditcontract.OpenAPIWorkArguments{Workflow: "scan", Selection: "scan-requests", MaxItems: 16, Scanner: scanner, Parameters: h.invocation.Context.Parameters}})
		if err != nil || len(prepared.Items) != 1 {
			t.Fatalf("ordinary preparation: %+v %v", prepared, err)
		}
		material("operation", prepared.Items[0].Data)
	}
	if scanner == "web" {
		h.invocation.Context.Artifacts["rulepack"] = nil
	}
	history := &auditFactoryHistory{factorySessions: h.sessions, stageID: h.invocation.StageExecutionID}
	h.factory.sessions = history
	h.invoker.tool = "scan_" + scanner
	if scanner == "web" {
		h.invoker.tool = "scan_nuclei"
	}
	return h, history
}

type scanFixtureReport struct {
	Schema             string   `json:"schema"`
	ExecutionCompleted bool     `json:"execution_completed"`
	Limitations        []string `json:"limitations"`
	Artifacts          []struct {
		Kind      string                `json:"kind"`
		Digest    string                `json:"digest"`
		Ref       contracts.ArtifactRef `json:"ref"`
		SizeBytes int                   `json:"size_bytes"`
	} `json:"artifacts"`
}

func auditFactoryResult(t *testing.T, h *factoryHarness, result stageresult.StageContentResult) scanFixtureReport {
	t.Helper()
	payload := h.artifacts.payload(result.Artifacts["report"])
	var report scanFixtureReport
	if err := json.Unmarshal(payload.Data, &report); err != nil || payload.MediaType != "application/json" || report.Schema != "contractor.scan.operation-report.v1" {
		t.Fatalf("ordinary report: %+v %v", payload, err)
	}
	for _, artifact := range report.Artifacts {
		if artifact.Ref.ValidateExact() != nil || !contentdigest.Valid(artifact.Digest) || artifact.SizeBytes < 1 {
			t.Fatal("invalid exact source descriptor")
		}
	}
	if bytes.Contains(payload.Data, []byte("data_base64")) {
		t.Fatal("report bytes embedded in aggregate")
	}

	return report
}

func TestAuditExecutorRetainsAssignedRequestAndURL(t *testing.T) {
	for _, scanner := range []string{"sqlmap", "nuclei"} {
		t.Run(scanner, func(t *testing.T) {
			h, _ := newAuditFactoryHarness(t, scanner)
			result := h.run(t)
			pkg := auditFactoryResult(t, h, result)
			check := pkg
			if result.Outcome != stageresult.StageSucceeded || len(h.invoker.calls) != 1 || !check.ExecutionCompleted {
				t.Fatalf("scan outcome: %+v, coverage: %+v, calls: %d", result, check, len(h.invoker.calls))
			}
			foundReport := false
			for _, evidence := range pkg.Artifacts {
				foundReport = foundReport || evidence.Kind == "scanner-report"
			}
			if !foundReport {
				t.Fatal("raw scanner report not retained")
			}
			call := h.invoker.calls[0].request
			if scanner == "sqlmap" {
				var request scanplan.SQLMapRequest
				if err := json.Unmarshal(h.artifacts.payload(call.Artifacts["request"]).Data, &request); err != nil {
					t.Fatal(err)
				}
				if request.Method != "POST" || request.Body != `{"name":"Milo"}` || !slices.Contains(request.TestParameters, "name") {
					t.Fatalf("request changed: %+v", request)
				}
				if slices.Contains(request.Headers, contractscan.HTTPRequestHeader{Name: "authorization", Value: "Bearer local-fixture-token"}) || call.Parameters["credential_id"] != "fixture-credential" {
					t.Fatal("credential leaked into request or reference lost")
				}
			} else {
				if call.Parameters["target"] != "http://127.0.0.1:8080/api/pets/7?search=Milo" || !slices.Contains(check.Limitations, "url_template_scan_only") {
					t.Fatalf("fixed URL or limits lost: %+v %+v", call, check)
				}
				plan, err := scanplan.DecodePlan(h.artifacts.payload(*h.sessions.state.Plan).Data)
				if err != nil || plan.Jobs[0].Parameters["template_ids"] != "http-missing-security-headers" {
					t.Fatalf("template selection changed: %v", err)
				}
			}
		})
	}
}

func TestAuditExecutorRetryUsesOutcomeInsteadOfStageIdentity(t *testing.T) {
	for _, outcome := range []string{"completed", "unknown", "failed", "lost-journal-write"} {
		t.Run(outcome, func(t *testing.T) {
			h, history := newAuditFactoryHarness(t, "sqlmap")
			switch outcome {
			case "unknown":
				h.invoker.invokeErr = errors.New("transport lost after dispatch")
			case "failed":
				h.invoker.observation = map[string]any{"status": "failed", "exitCode": 1}
			case "lost-journal-write":
				h.sessions.finishErr = errors.New("write acknowledgement lost")
			}
			first, err := h.instance(t).Run(t.Context())
			if (err != nil) != (outcome == "lost-journal-write") {
				t.Fatalf("first attempt: %v", err)
			}
			history.prior = []planner.ScanAttempt{{StageExecutionID: history.stageID, StageName: "execute", Terminal: true, State: cloneFactoryValue(h.sessions.state)}}
			h.invocation.StageExecutionID = "next-stage-attempt"
			history.stageID = h.invocation.StageExecutionID
			h.sessions.state, h.sessions.completion, h.sessions.finishErr = planner.ScanState{}, nil, nil
			h.invoker.invokeErr = nil
			h.invoker.observation = map[string]any{"status": "completed", "exitCode": 0}
			result := h.run(t)
			pkg := auditFactoryResult(t, h, result)
			wantCalls := 1
			if outcome == "failed" {
				wantCalls = 2
				if first.Artifacts["materials"].SameExact(result.Artifacts["materials"]) {
					t.Fatal("retry reused the previous attempt's immutable material set")
				}
			}
			if len(h.invoker.calls) != wantCalls {
				t.Fatalf("scanner calls = %d, want %d", len(h.invoker.calls), wantCalls)
			}
			if outcome == "unknown" || outcome == "lost-journal-write" {
				if pkg.ExecutionCompleted || !slices.Contains(pkg.Limitations, "scan_outcome_unknown") {
					t.Fatal("unknown action treated as a completed check")
				}
			}
		})
	}
}

func TestAuditExecutorRejectsChangedPinnedBytesAndUnavailableHistory(t *testing.T) {
	for _, fault := range []string{"openapi", "request_values", "operation", "history"} {
		t.Run(fault, func(t *testing.T) {
			h, history := newAuditFactoryHarness(t, "sqlmap")
			if fault == "history" {
				history.err = errors.New("storage unavailable")
			} else {
				key := factoryRefKey(*h.invocation.Context.Artifacts[fault])
				payload := h.artifacts.values[key]
				if fault == "operation" {
					payload.Data = []byte(strings.ReplaceAll(string(payload.Data), `"schema":"contractor.audit.openapi-work-item.v1"`, `"schema":"invalid"`))
				} else {
					payload.Data = append(payload.Data, ' ')
				}
				h.artifacts.values[key] = payload
			}
			if _, err := h.instance(t).Run(t.Context()); err == nil || len(h.invoker.calls) != 0 {
				t.Fatalf("fault caused dispatch: %v", err)
			}
		})
	}
}
