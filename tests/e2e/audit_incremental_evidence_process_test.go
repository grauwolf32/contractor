//go:build e2e

package e2e

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
)

// The Server commits while its reply is withheld. Runtime dies before it can
// submit a conclusion. Ordinary Audit reads recover exact selected bytes across
// Server restart; no target, model or synthetic conclusion repairs the loss.
func TestAuditIncrementalEvidenceAcrossProcesses(t *testing.T) {
	testAuditInterruptedRetention(t, false, false)
}

func TestAuditPartialJudgmentsAcrossProcesses(t *testing.T) {
	testAuditInterruptedRetention(t, true, false)
}

func testAuditInterruptedRetention(t *testing.T, withJudgment, browser bool) {
	var requests atomic.Int32
	target := startEngagementScanTarget(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		w.Header().Set("Content-Type", "text/plain")
		fmt.Fprint(w, "observed "+r.URL.Path)
	}))
	t.Cleanup(target.Close)
	accepted := make(chan auditcontract.EvidenceReceipt, 1)
	judgments := make(chan auditcontract.JudgmentReceipt, 1)
	release := make(chan struct{})
	var releaseOnce, savedOnce sync.Once
	t.Cleanup(func() { releaseOnce.Do(func() { close(release) }) })
	stage := basicConclusionStage(false)
	stage.tools = append(stage.tools, "http_request")
	stage.steps = stage.steps[:2]
	for _, path := range []string{"/control", "/test"} {
		stage.steps = append(stage.steps, toolGatewayStep("http_request", fixedArguments(map[string]any{
			"url": target.URL + path, "method": "GET", "body_type": "none",
		})))
	}
	recordSources := func(index int) domainGatewayStep {
		return toolGatewayStep("evidence", func(request map[string]any) (map[string]any, error) {
			keys, err := assignedAuditKeys(request)
			if err != nil {
				return nil, err
			}
			sources := []any{map[string]any{"kind": "artifact", "namespace": "inputs", "name": "material"}}
			exchanges := findingsToolResponses(request, "http_request")
			if len(exchanges) != 2 {
				return nil, fmt.Errorf("expected two actual exchanges, got %d", len(exchanges))
			}
			for _, exchange := range exchanges {
				if exchange["status"] != json.Number("200") {
					return nil, fmt.Errorf("fixture probe did not succeed: status=%v (%T), error=%v", exchange["status"], exchange["status"], exchange["error"])
				}
				sources = append(sources, map[string]any{"kind": "http", "request_id": exchange["request_id"]})
			}
			return map[string]any{"item_key": keys[index], "sources": sources, "interpretation": "The selected report and two local responses record observations; they do not establish the assigned security hypothesis.", "tested_conditions": []string{"selected source; actual control and test requests"}, "limitations": []string{"The HTTP status alone does not verify an access-control property"}}, nil
		})
	}
	stage.steps = append(stage.steps, recordSources(0))
	if withJudgment {
		stage.steps = append(stage.steps, recordSources(1))
		for _, text := range []string{"Preliminary fixture judgment A", "Latest accepted correction B"} {
			stage.steps = append(stage.steps, toolGatewayStep("submit_result", func(request map[string]any) (map[string]any, error) {
				keys, err := assignedAuditKeys(request)
				if err != nil || len(keys) != 2 {
					return nil, fmt.Errorf("expected two assigned hypotheses: %v", err)
				}
				return map[string]any{"item_key": keys[0], "verdict": "inconclusive", "rationale": text, "limitations": []string{"The captured responses do not establish the scoped security property"}}, nil
			}))
		}
	}
	h := startFindingsHarnessForBrowser(t, []domainGatewayStage{stage}, browser, func(root string) {
		installBasicAuditProgram(t, root)
		if err := os.WriteFile(filepath.Join(root, "agent-templates/basic_check.yaml"), []byte(strings.Replace(basicCheckTemplate, "  sandboxProfile:", "    - {ref: http-tools@1, tools: [http_request]}\n  sandboxProfile:", 1)), 0600); err != nil {
			t.Fatal(err)
		}
		program := replaceAuditFixture(t, basicSeededProcessProgram(t), "  inputs:\n", "  inputs:\n    engagement: {required: true, media_types: [application/vnd.contractor.engagement+json]}\n")
		program = replaceAuditFixture(t, program, "active_operations: prohibited", "active_operations: allowed")
		if err := os.WriteFile(filepath.Join(root, "audit-profiles/basic_process.yaml"), []byte(program), 0600); err != nil {
			t.Fatal(err)
		}
	}, func(h *findingsHarness) {
		h.runtimeExtraArguments = []string{"--allowed-target-network", "127.0.0.0/8"}
		h.proxyPrivateAPI(func(response *http.Response) error {
			evidence := strings.HasSuffix(response.Request.URL.Path, "/audit-evidence")
			judgment := withJudgment && strings.HasSuffix(response.Request.URL.Path, "/audit-results")
			if (!evidence && !judgment) || response.StatusCode != http.StatusCreated {
				return nil
			}
			raw, err := io.ReadAll(io.LimitReader(response.Body, 1<<20))
			if err != nil {
				return err
			}
			response.Body.Close()
			response.Body = io.NopCloser(bytes.NewReader(raw))
			if evidence {
				var receipt auditcontract.EvidenceReceipt
				if err := json.Unmarshal(raw, &receipt); err != nil {
					return err
				}
				savedOnce.Do(func() { accepted <- receipt })
				if withJudgment {
					return nil
				}
			} else {
				var receipt auditcontract.JudgmentReceipt
				if err := json.Unmarshal(raw, &receipt); err != nil {
					return err
				}
				if len(receipt.Generations) != 1 || receipt.Generations[0].Generation != 2 {
					return nil
				}
				judgments <- receipt
			}
			select {
			case <-release:
				return nil
			case <-response.Request.Context().Done():
				return response.Request.Context().Err()
			}
		})
	})
	project := createProjectResource(t, h.client, h.baseURL)
	engagement := uploadScanEngagement(t, h.client, h.baseURL, project.ProjectID, target.URL, false)
	source := uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "fixture", "source", "application/zip", auditProgramZip(t, map[string][]byte{"app.py": []byte(findingsSource)}))
	report := uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "fixture", "report", "text/plain", []byte(findingsEvidence))
	assertion := map[string]any{"schema": "contractor.audit.assertions.v1", "items": []any{map[string]any{"key": "boundary", "version": "1", "statement": "The selected access boundary can be bypassed", "scope": map[string]string{"deployment": "fixture"}, "grounds": "Source observation requiring a bounded check", "applicability": "selected fixture", "allowed_methods": []string{"http", "static"}, "required_evidence": []string{"Exact source"}}}}
	if withJudgment {
		items := assertion["items"].([]any)
		second := map[string]any{}
		for key, value := range items[0].(map[string]any) {
			second[key] = value
		}
		second["key"], second["statement"] = "unfinished", "A separate assigned boundary remains unresolved"
		assertion["items"] = append(items, second)
	}
	checklist := uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "fixture", "checklist", "application/json", mustBasicJSON(t, assertion))
	draft, _ := basicAuditRequest(t, h, http.MethodPost, "/v1/projects/"+project.ProjectID+"/audits", map[string]any{"profile": map[string]string{"name": "basic-process", "version": "1"}, "inputs": map[string]artifactRef{"source": source, "report": report, "checklist": checklist, "engagement": engagement}}, "evidence-draft", 0, http.StatusCreated)
	id := draft["auditId"].(string)
	basicAuditRequest(t, h, http.MethodPost, "/v1/audits/"+id+"/start", nil, "evidence-start", uint64(draft["revision"].(float64)), http.StatusOK)
	var receipt auditcontract.EvidenceReceipt
waitForAcceptance:
	for {
		select {
		case receipt = <-accepted:
			break waitForAcceptance
		default:
			if failures := h.gateway.Failures(); len(failures) > 0 {
				t.Fatalf("evidence model fixture failed before acceptance: %v", failures)
			}
			basicPoll(t, h)
		}
	}
	if receipt.Material.ScopeID != id || len(receipt.Sources) != 3 {
		t.Fatalf("wrong accepted selection: %+v", receipt)
	}
	var judgment auditcontract.JudgmentReceipt
	if withJudgment {
	waitForJudgment:
		for {
			select {
			case judgment = <-judgments:
				break waitForJudgment
			default:
				if failures := h.gateway.Failures(); len(failures) > 0 {
					t.Fatalf("judgment fixture failed before acceptance: %v", failures)
				}
				basicPoll(t, h)
			}
		}
	}
	h.runtimeProcess.stop(t)
	releaseOnce.Do(func() { close(release) })
	h.restartServer()
	cancelled := false
	for range 16 {
		current := basicAuditGet(t, h, id)
		if state := current["state"]; state == "completed" || state == "cancelled" || state == "failed" {
			// A model-inconclusive or system-stop result can settle the attempt;
			// normal drain may finish before an explicit cancellation is admitted.
			cancelled = true
			break
		}
		revision := uint64(current["revision"].(float64))
		request, err := http.NewRequestWithContext(h.ctx, http.MethodPost, h.baseURL+"/v1/audits/"+id+"/cancel", nil)
		if err != nil {
			t.Fatal(err)
		}
		request.Header.Set("Authorization", "Bearer "+publicToken)
		request.Header.Set("Idempotency-Key", fmt.Sprintf("evidence-cancel-%d", revision))
		request.Header.Set("If-Match", fmt.Sprintf(`"%d"`, revision))
		response, err := h.client.Do(request)
		if err != nil {
			t.Fatal(err)
		}
		data, err := io.ReadAll(io.LimitReader(response.Body, 1<<20))
		response.Body.Close()
		if err != nil {
			t.Fatal(err)
		}
		if response.StatusCode == http.StatusPreconditionFailed {
			// Restart recovery may advance the owner revision. A definite stale
			// rejection has no mutation; re-read before admitting a new command.
			basicPoll(t, h)
			continue
		}
		if response.StatusCode != http.StatusAccepted {
			t.Fatalf("cancel HTTP %d: %s", response.StatusCode, data)
		}
		cancelled = true
		break
	}
	if !cancelled {
		t.Fatal("bounded cancellation admission did not settle")
	}
	for {
		value, _ := basicAuditRequest(t, h, http.MethodGet, "/v1/audits/"+id+"/tasks?limit=64", nil, "", 0, http.StatusOK)
		var page struct {
			Items []basicTask `json:"items"`
		}
		if err := json.Unmarshal(mustBasicJSON(t, value), &page); err != nil {
			t.Fatal(err)
		}
		expected := 1
		if withJudgment {
			expected = 2
		}
		settled := len(page.Items) == expected
		for _, task := range page.Items {
			settled = settled && task.State == "settled"
		}
		if settled {
			modelCount, systemCount := 0, 0
			for _, task := range page.Items {
				if task.Result != nil {
					if !withJudgment || !task.Result.Ref.SameExact(judgment.Material.Ref) || task.Stop != nil {
						t.Fatal("wrong accepted partial judgment")
					}
					modelCount++
				} else {
					if task.Stop == nil || task.Stop.Author != "system" || task.Stop.Meaning != "no_accepted_conclusion" || task.Stop.Verdict == nil || *task.Stop.Verdict != "inconclusive" {
						t.Fatal("missing attributed system interruption")
					}
					systemCount++
				}
			}
			if systemCount != 1 || withJudgment && modelCount != 1 {
				t.Fatal("partial results were lost or manufactured")
			}
			break
		}
		basicPoll(t, h)
	}
	for {
		state := basicAuditGet(t, h, id)["state"]
		if state == "completed" || state == "cancelled" || state == "failed" {
			break
		}
		basicPoll(t, h)
	}
	var runID string
	// Read the accepted note before producer deletion to obtain its machine origin.
	materials, _ := basicAuditRequest(t, h, http.MethodGet, "/v1/audits/"+id+"/materials?limit=100", nil, "", 0, http.StatusOK)
	if materials["page"].(map[string]any)["hasMore"].(bool) {
		t.Fatal("small evidence fixture unexpectedly exceeded its bounded page")
	}
	foundNote, foundSource, foundJudgment := false, false, !withJudgment
	foundHTTP := map[string]bool{}
	readRetained := func(afterPurge bool) {
		materials, _ = basicAuditRequest(t, h, http.MethodGet, "/v1/audits/"+id+"/materials?limit=100", nil, "", 0, http.StatusOK)
		for _, raw := range materials["items"].([]any) {
			entry := raw.(map[string]any)
			material := entry["material"].(map[string]any)
			ref := material["ref"].(map[string]any)
			selected := ref["name"] == receipt.Material.Ref.Name
			for _, source := range receipt.Sources {
				selected = selected || ref["name"] == source.Ref.Name
			}
			selected = selected || withJudgment && ref["name"] == judgment.Material.Ref.Name
			if !selected {
				continue
			}
			path := "/v1/audits/" + id + "/artifacts?" + url.Values{"key": {entry["key"].(string)}, "revision": {ref["revision"].(string)}}.Encode()
			data := readInterruptedAuditMaterial(t, h, path)
			if withJudgment && ref["name"] == judgment.Material.Ref.Name {
				var document auditcontract.ResultDocument
				if json.Unmarshal(data, &document) != nil || len(document.Items) != 1 || document.Items[0].Author != "model" || document.Items[0].Rationale != "Latest accepted correction B" || document.Items[0].Verdict == nil || *document.Items[0].Verdict != "inconclusive" || document.Items[0].Origin.Hypothesis == nil || auditdomain.DigestBytes(data) != judgment.Material.Digest {
					t.Fatal("latest committed correction changed across interruption")
				}
				if runID != "" && runID != document.Items[0].Origin.RunID {
					t.Fatal("judgment origin changed")
				}
				foundJudgment = true
			} else if ref["name"] == receipt.Sources[0].Ref.Name {
				foundSource = true
				if string(data) != findingsEvidence || auditdomain.DigestBytes(data) != receipt.Sources[0].Digest {
					t.Fatal("retained selected source changed")
				}
			} else if material["media_type"] == "application/vnd.contractor.http-exchange+json" {
				exchange, err := auditdomain.DecodeHTTPExchangeArtifact(data)
				if err != nil || len(exchange.Capture.Attempts) != 1 || exchange.Capture.Attempts[0].Response == nil {
					t.Fatal("retained HTTP exchange changed")
				}
				attempt := exchange.Capture.Attempts[0]
				if attempt.Response.Status != http.StatusOK || attempt.Response.Message.BodyBase64 == nil {
					t.Fatal("retained HTTP response changed")
				}
				body, err := base64.StdEncoding.DecodeString(*attempt.Response.Message.BodyBase64)
				path := strings.TrimPrefix(attempt.Request.URL, target.URL)
				if err != nil || (path != "/control" && path != "/test") || string(body) != "observed "+path {
					t.Fatal("retained HTTP bytes changed")
				}
				foundHTTP[path] = true
			} else {
				foundNote = true
				var note auditcontract.EvidenceRecord
				if json.Unmarshal(data, &note) != nil || note.Validate() != nil || note.Origin.AuditID != id || note.Origin.Hypothesis == nil {
					t.Fatal("accepted evidence lost its machine-derived check origin")
				}
				if afterPurge && runID != note.Origin.RunID {
					t.Fatal("producer purge changed evidence origin")
				}
				runID = note.Origin.RunID
			}
		}
		if !foundNote || !foundSource || !foundJudgment || len(foundHTTP) != 2 {
			t.Fatal("Worker loss discarded accepted sources or annotation")
		}
	}
	readRetained(false)
	if withJudgment {
		var run struct {
			State string `json:"state"`
		}
		auditProgramGET(t, h.client, h.baseURL+"/v1/runs/"+runID, &run)
		if run.State != "failed" && run.State != "cancelled" {
			t.Fatalf("partial judgment changed technical Run outcome: %s", run.State)
		}
	}
	deleteASVSBacktraceRun(t, h.ctx, h.client, h.baseURL, runID)
	foundNote, foundSource = false, false
	foundJudgment = !withJudgment
	foundHTTP = map[string]bool{}
	readRetained(true)
	if browser {
		runResearchBrowser(t, h, project.ProjectID, id, "interrupted")
	}
	if requests.Load() != 2 || len(h.gateway.Failures()) != 0 {
		t.Fatalf("unexpected target/model replay: requests=%d failures=%v", requests.Load(), h.gateway.Failures())
	}
}

func readInterruptedAuditMaterial(t *testing.T, h *findingsHarness, path string) []byte {
	t.Helper()
	for range 32 {
		request, err := http.NewRequestWithContext(h.ctx, http.MethodGet, h.baseURL+path, nil)
		if err != nil {
			t.Fatal(err)
		}
		request.Header.Set("Authorization", "Bearer "+publicToken)
		response, err := h.client.Do(request)
		if err != nil {
			t.Fatal(err)
		}
		data, err := io.ReadAll(io.LimitReader(response.Body, 8<<20))
		response.Body.Close()
		if err != nil {
			t.Fatal(err)
		}
		if response.StatusCode == http.StatusConflict || response.StatusCode == http.StatusServiceUnavailable {
			basicPoll(t, h)
			continue
		}
		if response.StatusCode != http.StatusOK {
			t.Fatalf("exact retained read HTTP %d: %s", response.StatusCode, data)
		}
		return data
	}
	t.Fatal("exact retained read did not settle within its bounded attempts")
	return nil
}
