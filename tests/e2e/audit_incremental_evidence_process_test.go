//go:build e2e

package e2e

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	pb "github.com/grauwolf32/contractor/api/v1alpha1/runtimepb"
	"github.com/grauwolf32/contractor/internal/runtimewire"
	"google.golang.org/protobuf/proto"
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
	"github.com/grauwolf32/contractor/internal/findingintake"
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
	if browser {
		stage.tools = append(stage.tools, "open_source_archive", "read_source")
	}
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
		if browser {
			stage.steps = append(stage.steps,
				toolGatewayStep("open_source_archive", fixedArguments(map[string]any{"namespace": "inputs", "name": "source"})),
				toolGatewayStep("read_source", fixedArguments(map[string]any{"path": "app.py", "start_line": 1, "max_lines": 100})),
				toolGatewayStep("finding", func(request map[string]any) (map[string]any, error) {
					exchanges := findingsToolResponses(request, "http_request")
					notes := findingsToolResponses(request, "evidence")
					if len(exchanges) != 2 || len(notes) != 2 || notes[0]["persistence"] != "accepted" {
						return nil, fmt.Errorf("mixed sources were not captured and retained")
					}
					selected, ok := notes[0]["sourceSelectors"].([]any)
					if !ok || len(selected) != 3 {
						return nil, fmt.Errorf("missing reusable exact source selectors")
					}
					return map[string]any{"title": "Incidental query concatenation observation", "description": findingsEvidence + " The local HTTP response supplies observation context only; it does not prove exploitation.", "file": "app.py", "range": map[string]int{"start_line": 4, "end_line": 10}, "cwe": "CWE-89", "incidental": true, "sources": []any{map[string]any{"kind": "code", "file": "app.py", "start_line": 4, "end_line": 10}, map[string]any{"kind": "http", "request_id": exchanges[0]["request_id"]}, selected[0]}}, nil
				}))
		}
		if !browser {
			stage.steps = append(stage.steps, toolGatewayStep("finding", func(request map[string]any) (map[string]any, error) {
				exchanges := findingsToolResponses(request, "http_request")
				notes := findingsToolResponses(request, "evidence")
				if len(exchanges) != 2 || len(notes) != 2 {
					return nil, fmt.Errorf("missing control/probe observations")
				}
				selected, ok := notes[0]["sourceSelectors"].([]any)
				if !ok || len(selected) != 3 {
					return nil, fmt.Errorf("missing exact material selector")
				}
				return map[string]any{"title": "Compare control and probe", "description": "Both observed responses are evidence of this incidental observation.", "file": "app.py", "incidental": true, "sources": []any{selected[0], map[string]any{"kind": "http", "request_id": exchanges[0]["request_id"]}, map[string]any{"kind": "http", "request_id": exchanges[1]["request_id"]}}}, nil
			}))
		}
		for _, text := range []string{"Preliminary fixture judgment A", "Latest accepted correction B"} {
			stage.steps = append(stage.steps, toolGatewayStep("submit_result", func(request map[string]any) (map[string]any, error) {
				if browser {
					published := findingsToolResponses(request, "finding")
					if len(published) != 1 || published[0]["finding_id"] == nil {
						return nil, fmt.Errorf("incidental Finding was not acknowledged: %v", published)
					}
				}
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
		template := strings.Replace(basicCheckTemplate, "  sandboxProfile:", "    - {ref: http-tools@1, tools: [http_request]}\n  sandboxProfile:", 1)
		if browser {
			template = strings.Replace(template, "  sandboxProfile:", "    - {ref: source-analysis@1, tools: [open_source_archive, read_source]}\n  sandboxProfile:", 1)
		}
		if err := os.WriteFile(filepath.Join(root, "agent-templates/basic_check.yaml"), []byte(template), 0600); err != nil {
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
			evidence := response.Request.URL.Path == pb.Allocation_SubmitAuditEvidence_FullMethodName
			judgment := withJudgment && response.Request.URL.Path == pb.Allocation_SubmitAuditJudgments_FullMethodName
			if (!evidence && !judgment) || response.StatusCode != http.StatusOK {
				return nil
			}
			raw, err := io.ReadAll(io.LimitReader(response.Body, 1<<20))
			if err != nil {
				return err
			}
			response.Body.Close()
			response.Body = io.NopCloser(bytes.NewReader(raw))
			if len(raw) < 5 || raw[0] != 0 || int(binary.BigEndian.Uint32(raw[1:5])) != len(raw)-5 {
				return fmt.Errorf("invalid Audit RPC frame")
			}

			if evidence {
				var receipt auditcontract.EvidenceReceipt
				var message pb.AuditEvidenceReceipt
				if err := proto.Unmarshal(raw[5:], &message); err != nil {
					return err
				}
				if err := runtimewire.DecodeDomain(&message, &receipt, 1<<20, (1<<20)+1024); err != nil {
					return err
				}
				savedOnce.Do(func() { accepted <- receipt })
				if withJudgment {
					return nil
				}
			} else {
				var receipt auditcontract.JudgmentReceipt
				var message pb.AuditJudgmentsResponse
				if err := proto.Unmarshal(raw[5:], &message); err != nil {
					return err
				}
				if message.GetReceipt() == nil {
					return nil
				}
				if err := runtimewire.DecodeDomain(message.GetReceipt(), &receipt, 1<<20, (1<<20)+1024); err != nil {
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
	var multiReceiptID string
	if withJudgment && !browser {
		var findings struct {
			Items []struct {
				FindingID string `json:"findingId"`
			} `json:"items"`
		}
		auditProgramGET(t, h.client, h.baseURL+"/v1/projects/"+project.ProjectID+"/findings?limit=100", &findings)
		if len(findings.Items) != 1 {
			t.Fatalf("missing multi-HTTP Finding: %+v", findings)
		}
		var detail struct {
			FirstReceiptID string `json:"firstReceiptId"`
		}
		auditProgramGET(t, h.client, h.baseURL+"/v1/projects/"+project.ProjectID+"/findings/"+findings.Items[0].FindingID, &detail)
		multiReceiptID = detail.FirstReceiptID
		var source findingintake.Receipt
		auditProgramGET(t, h.client, h.baseURL+"/v1/projects/"+project.ProjectID+"/finding-sources/"+detail.FirstReceiptID, &source)
		if len(source.Evidence) != 3 || len(source.Document.HTTPExchanges) != 2 {
			t.Fatalf("lost HTTP evidence after restart: %+v", source.Document)
		}
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
	var incidental findingintake.Receipt
	if browser {
		receipts := findingsReceipts(t, h, runID)
		if len(receipts) != 1 || receipts[0].FindingID == nil || receipts[0].Origin.Audit == nil || receipts[0].Origin.Audit.AuditID != id || receipts[0].Origin.Audit.TaskID != "" || receipts[0].Origin.Audit.Hypothesis != nil || len(receipts[0].ProjectEvidence) != 4 {
			t.Fatalf("incidental mixed-source Finding was lost or acquired a hypothesis: %+v", receipts)
		}
		incidental = receipts[0]
	}
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
	if multiReceiptID != "" {
		var saved findingintake.Receipt
		auditProgramGET(t, h.client, h.baseURL+"/v1/projects/"+project.ProjectID+"/finding-sources/"+multiReceiptID, &saved)
		if !saved.Origin.RunDeleted || len(saved.ProjectEvidence) != 3 || len(saved.Document.HTTPExchanges) != 2 {
			t.Fatal("producer deletion lost HTTP evidence", saved)
		}
	}
	if browser {
		var saved findingintake.Receipt
		auditProgramGET(t, h.client, h.baseURL+"/v1/projects/"+project.ProjectID+"/finding-sources/"+incidental.ReceiptID, &saved)
		if saved.FindingID == nil || *saved.FindingID != *incidental.FindingID || !saved.Origin.RunDeleted || saved.Origin.Audit == nil || saved.Origin.Audit.Hypothesis != nil || saved.Origin.Audit.TaskID != "" || len(saved.ProjectEvidence) != 4 {
			t.Fatalf("producer purge changed incidental provenance: %+v", saved)
		}
		for _, source := range saved.ProjectEvidence {
			if source.Ref.Revision == nil {
				t.Fatal("Finding source is not exact")
			}
			data, _ := download(t, h.client, h.baseURL+"/v1/projects/"+project.ProjectID+"/artifacts/"+source.Ref.Namespace+"/"+source.Ref.Name+"?revision="+url.QueryEscape(*source.Ref.Revision))
			if auditdomain.DigestBytes(data) != source.Digest || int64(len(data)) != source.SizeBytes {
				t.Fatal("retained mixed Finding source bytes changed")
			}
		}
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
