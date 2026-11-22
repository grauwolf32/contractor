//go:build e2e

package e2e

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
)

// The Server commits while its reply is withheld. Runtime dies before it can
// submit a conclusion. Ordinary Audit reads recover exact selected bytes across
// Server restart; no target, model or synthetic conclusion repairs the loss.
func TestAuditIncrementalEvidenceAcrossProcesses(t *testing.T) {
	accepted := make(chan auditcontract.EvidenceReceipt, 1)
	release := make(chan struct{})
	var releaseOnce, savedOnce sync.Once
	t.Cleanup(func() { releaseOnce.Do(func() { close(release) }) })
	stage := basicConclusionStage(false)
	stage.steps = stage.steps[:2]
	stage.steps = append(stage.steps, toolGatewayStep("evidence", func(request map[string]any) (map[string]any, error) {
		keys, err := assignedAuditKeys(request)
		if err != nil {
			return nil, err
		}
		return map[string]any{"item_key": keys[0], "sources": []any{map[string]any{"kind": "artifact", "namespace": "inputs", "name": "material"}}, "interpretation": "The exact selected report records source reasoning, not independently verified deployment behavior.", "tested_conditions": []string{"selected fixture source"}, "limitations": []string{"No live target request was executed"}}, nil
	}))
	h := startConfiguredFindingsHarness(t, []domainGatewayStage{stage}, func(root string) {
		installBasicAuditProgram(t, root)
		if err := os.WriteFile(filepath.Join(root, "audit-profiles/basic_process.yaml"), []byte(basicSeededProcessProgram(t)), 0600); err != nil {
			t.Fatal(err)
		}
	}, func(h *findingsHarness) {
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
	source := uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "fixture", "source", "application/zip", auditProgramZip(t, map[string][]byte{"app.py": []byte(findingsSource)}))
	report := uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "fixture", "report", "text/plain", []byte(findingsEvidence))
	assertion := map[string]any{"schema": "contractor.audit.assertions.v1", "items": []any{map[string]any{"key": "boundary", "version": "1", "statement": "The selected access boundary can be bypassed", "scope": map[string]string{"deployment": "fixture"}, "grounds": "Source observation requiring a bounded check", "applicability": "selected fixture", "allowed_methods": []string{"static"}, "required_evidence": []string{"Exact source"}}}}
	checklist := uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "fixture", "checklist", "application/json", mustBasicJSON(t, assertion))
	draft, _ := basicAuditRequest(t, h, http.MethodPost, "/v1/projects/"+project.ProjectID+"/audits", map[string]any{"profile": map[string]string{"name": "basic-process", "version": "1"}, "inputs": map[string]artifactRef{"source": source, "report": report, "checklist": checklist}}, "evidence-draft", 0, http.StatusCreated)
	id := draft["auditId"].(string)
	basicAuditRequest(t, h, http.MethodPost, "/v1/audits/"+id+"/start", nil, "evidence-start", uint64(draft["revision"].(float64)), http.StatusOK)
	var receipt auditcontract.EvidenceReceipt
	select {
	case receipt = <-accepted:
	case <-h.ctx.Done():
		t.Fatal("evidence did not reach Server acceptance")
	}
	if receipt.Material.ScopeID != id || len(receipt.Sources) != 1 {
		t.Fatalf("wrong accepted selection: %+v", receipt)
	}
	h.runtimeProcess.stop(t)
	releaseOnce.Do(func() { close(release) })
	h.restartServer()
	materials, _ := basicAuditRequest(t, h, http.MethodGet, "/v1/audits/"+id+"/materials?limit=100", nil, "", 0, http.StatusOK)
	if materials["page"].(map[string]any)["hasMore"].(bool) {
		t.Fatal("small evidence fixture unexpectedly exceeded its bounded page")
	}
	foundNote, foundSource := false, false
	for _, raw := range materials["items"].([]any) {
		entry := raw.(map[string]any)
		material := entry["material"].(map[string]any)
		ref := material["ref"].(map[string]any)
		if ref["name"] != receipt.Material.Ref.Name && ref["name"] != receipt.Sources[0].Ref.Name {
			continue
		}
		path := "/v1/audits/" + id + "/artifacts?" + url.Values{"key": {entry["key"].(string)}, "revision": {ref["revision"].(string)}}.Encode()
		data, _ := download(t, h.client, h.baseURL+path)
		if ref["name"] == receipt.Sources[0].Ref.Name {
			foundSource = true
			if string(data) != findingsEvidence || auditdomain.DigestBytes(data) != receipt.Sources[0].Digest {
				t.Fatal("retained selected source changed")
			}
		} else {
			foundNote = true
			var note auditcontract.EvidenceRecord
			if json.Unmarshal(data, &note) != nil || note.Validate() != nil || note.Origin.AuditID != id || note.Origin.Hypothesis == nil {
				t.Fatal("accepted evidence lost its machine-derived check origin")
			}
		}
	}
	if !foundNote || !foundSource {
		t.Fatal("Worker loss discarded accepted evidence")
	}
}
