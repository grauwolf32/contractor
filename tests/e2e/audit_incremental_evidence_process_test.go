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
	var requests atomic.Int32
	target := startEngagementScanTarget(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		w.Header().Set("Content-Type", "text/plain")
		fmt.Fprint(w, "observed "+r.URL.Path)
	}))
	t.Cleanup(target.Close)
	accepted := make(chan auditcontract.EvidenceReceipt, 1)
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
	stage.steps = append(stage.steps, toolGatewayStep("evidence", func(request map[string]any) (map[string]any, error) {
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
		return map[string]any{"item_key": keys[0], "sources": sources, "interpretation": "The selected report and two local responses record observations; they do not establish the assigned security hypothesis.", "tested_conditions": []string{"selected source; actual control and test requests"}, "limitations": []string{"The HTTP status alone does not verify an access-control property"}}, nil
	}))
	h := startConfiguredFindingsHarness(t, []domainGatewayStage{stage}, func(root string) {
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
			if !strings.HasSuffix(response.Request.URL.Path, "/audit-evidence") || response.StatusCode != http.StatusCreated {
				return nil
			}
			raw, err := io.ReadAll(io.LimitReader(response.Body, 1<<20))
			if err != nil {
				return err
			}
			response.Body.Close()
			response.Body = io.NopCloser(bytes.NewReader(raw))
			var receipt auditcontract.EvidenceReceipt
			if err := json.Unmarshal(raw, &receipt); err != nil {
				return err
			}
			savedOnce.Do(func() { accepted <- receipt })
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
	h.runtimeProcess.stop(t)
	releaseOnce.Do(func() { close(release) })
	h.restartServer()
	current := basicAuditGet(t, h, id)
	basicAuditRequest(t, h, http.MethodPost, "/v1/audits/"+id+"/cancel", nil, "evidence-cancel", uint64(current["revision"].(float64)), http.StatusAccepted)
	for {
		value, _ := basicAuditRequest(t, h, http.MethodGet, "/v1/audits/"+id+"/tasks?limit=64", nil, "", 0, http.StatusOK)
		var page struct {
			Items []basicTask `json:"items"`
		}
		if err := json.Unmarshal(mustBasicJSON(t, value), &page); err != nil {
			t.Fatal(err)
		}
		if len(page.Items) == 1 && page.Items[0].State == "settled" {
			if page.Items[0].Result != nil {
				t.Fatal("Worker loss manufactured an unsubmitted conclusion")
			}
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
	foundNote, foundSource := false, false
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
			if !selected {
				continue
			}
			path := "/v1/audits/" + id + "/artifacts?" + url.Values{"key": {entry["key"].(string)}, "revision": {ref["revision"].(string)}}.Encode()
			data, _ := download(t, h.client, h.baseURL+path)
			if ref["name"] == receipt.Sources[0].Ref.Name {
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
		if !foundNote || !foundSource || len(foundHTTP) != 2 {
			t.Fatal("Worker loss discarded accepted sources or annotation")
		}
	}
	readRetained(false)
	deleteASVSBacktraceRun(t, h.ctx, h.client, h.baseURL, runID)
	foundNote, foundSource = false, false
	foundHTTP = map[string]bool{}
	readRetained(true)
	if requests.Load() != 2 || len(h.gateway.Failures()) != 0 {
		t.Fatalf("unexpected target/model replay: requests=%d failures=%v", requests.Load(), h.gateway.Failures())
	}
}
