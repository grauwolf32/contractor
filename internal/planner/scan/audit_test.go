package scan

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/scanplan"
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
	snapshot, err := config.Load("../../../configs", config.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	workflow, err := snapshot.Workflow("audit-openapi-" + scanner + "-scan@1")
	if err != nil {
		t.Fatal(err)
	}
	h.invocation.Stage = workflow.Stages[workflow.AuditTask.Stage]
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
		data, err := os.ReadFile("../../../configs/scan/examples/audit-openapi-scan/" + name)
		if err != nil {
			t.Fatal(err)
		}
		return data
	}
	source, settings := read("openapi.json"), read(scanner+"-settings.json")
	sourceRef, settingsRef := factoryExactRef("project", "api-document"), factoryExactRef("project", "scan-options")
	inventory, err := auditdomain.BuildOpenAPIScanInventory(source, auditdomain.JSONMediaType, settings,
		auditdomain.ExactInput{Name: "scan_options", Ref: settingsRef, Digest: "sha256:" + stableHash(string(settings))},
		auditdomain.InventoryOptions{Round: 1, WorkflowRole: "check", SourceInputName: "api_document", SourceRef: sourceRef, ApprovalRequirement: auditdomain.ApprovalActiveCheck})
	if err != nil {
		t.Fatal(err)
	}
	manifest := inventory.ExecutionManifest
	taskRef := factoryExactRef("project", "assigned-task")
	manifest.Items[0].TaskRef = &taskRef
	// Real dispatch rewrites input names to Workflow aliases. A surrounding
	// Stage may consume another immutable input that the scan never receives.
	manifest.Items[0].Inputs = []auditdomain.ExactInput{
		{Name: "notes", Ref: factoryExactRef("project", "notes"), Digest: "sha256:" + stableHash("operator notes")},
		{Name: "openapi", Ref: sourceRef, Digest: inventory.Tasks[0].Document.SourceContentDigest},
		{Name: "settings", Ref: settingsRef, Digest: inventory.Tasks[0].Document.Scan.Settings.Digest},
	}
	manifestBytes, err := auditdomain.EncodeExecutionManifest(manifest)
	if err != nil {
		t.Fatal(err)
	}
	for slot, payload := range map[string]artifacts.Payload{
		"openapi":            {MediaType: auditdomain.JSONMediaType, Data: source},
		"settings":           {MediaType: auditdomain.JSONMediaType, Data: settings},
		"task":               {MediaType: auditdomain.PackageMediaType, Data: inventory.Tasks[0].Package},
		"execution_manifest": {MediaType: auditdomain.JSONMediaType, Data: manifestBytes},
	} {
		ref := h.artifacts.put(contracts.ArtifactRef{Namespace: "inputs", Name: slot}, payload)
		h.invocation.Context.Artifacts[slot] = &ref
	}
	history := &auditFactoryHistory{factorySessions: h.sessions, stageID: h.invocation.StageExecutionID}
	h.factory.sessions = history
	h.invoker.tool = "scan_" + scanner
	return h, history
}

func auditFactoryResult(t *testing.T, h *factoryHarness, result contracts.StageContentResult) auditdomain.CheckResultPackage {
	t.Helper()
	payload := h.artifacts.payload(result.Artifacts["report"])
	pkg, err := auditdomain.DecodeCheckResultPackage(payload.Data)
	if err != nil || payload.MediaType != auditdomain.PackageMediaType {
		t.Fatalf("result package: %v", err)
	}
	return pkg
}

func TestAuditExecutorRetainsAssignedRequestAndURL(t *testing.T) {
	for _, scanner := range []string{"sqlmap", "nuclei"} {
		t.Run(scanner, func(t *testing.T) {
			h, _ := newAuditFactoryHarness(t, scanner)
			result := h.run(t)
			pkg := auditFactoryResult(t, h, result)
			check := pkg.Results.Results[0]
			if result.Outcome != contracts.StageSucceeded || len(h.invoker.calls) != 1 || check.Assessment != "inconclusive" || len(check.Coverage.Completed) != 1 {
				t.Fatalf("scan outcome: %+v, coverage: %+v, calls: %d", result, check, len(h.invoker.calls))
			}
			foundReport := false
			for _, evidence := range pkg.Evidence.Evidence {
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
				if !slices.Contains(request.Headers, contracts.HTTPRequestHeader{Name: "authorization", Value: "Bearer local-fixture-token"}) {
					t.Fatal("authentication lost")
				}
			} else {
				if call.Parameters["target"] != "http://127.0.0.1:8080/api/pets/7?search=Milo" || !slices.Contains(check.Coverage.Gaps, "url_template_scan_only") {
					t.Fatalf("fixed URL or limits lost: %+v %+v", call, check)
				}
				plan, err := scanplan.DecodePlan(h.artifacts.payload(*h.sessions.state.Plan).Data)
				if err != nil || plan.Jobs[0].Execution.Arguments["template_ids"].Value != "http-missing-security-headers" {
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
			_, err := h.instance(t).Run(t.Context())
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
			}
			if len(h.invoker.calls) != wantCalls {
				t.Fatalf("scanner calls = %d, want %d", len(h.invoker.calls), wantCalls)
			}
			if outcome == "unknown" || outcome == "lost-journal-write" {
				if len(pkg.Results.Results[0].Coverage.Completed) != 0 || !slices.Contains(pkg.Results.Results[0].Coverage.Gaps, "scan_outcome_unknown") {
					t.Fatal("unknown action treated as a completed check")
				}
			}
		})
	}
}

func TestAuditExecutorRejectsChangedPinnedBytesAndUnavailableHistory(t *testing.T) {
	for _, fault := range []string{"openapi", "settings", "execution_manifest", "history"} {
		t.Run(fault, func(t *testing.T) {
			h, history := newAuditFactoryHarness(t, "sqlmap")
			if fault == "history" {
				history.err = errors.New("storage unavailable")
			} else {
				key := factoryRefKey(*h.invocation.Context.Artifacts[fault])
				payload := h.artifacts.values[key]
				if fault == "execution_manifest" {
					payload.Data = []byte(strings.ReplaceAll(string(payload.Data), `"name":"settings"`, `"name":"substituted"`))
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
