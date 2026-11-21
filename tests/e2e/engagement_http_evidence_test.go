//go:build e2e

package e2e

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/engagementcontract"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

func TestEngagementHTTPSelectedEvidenceAcrossProcesses(t *testing.T) {
	selectedHTTPEvidenceAcrossProcesses(t, false)
}

func TestEngagementHTTPFindingReusesSelectedEvidenceAcrossProcesses(t *testing.T) {
	selectedHTTPEvidenceAcrossProcesses(t, true)
}

func selectedHTTPEvidenceAcrossProcesses(t *testing.T, withFinding bool) {
	t.Helper()
	secure := newEngagementTargetFixture(t)
	target := newEngagementHTTPObserver(t, engagementFixtureIP(t, secure))
	target.setHandler(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/selected" {
			w.Header().Set("Location", "/api/final")
			w.WriteHeader(302)
			return
		}
		w.WriteHeader(403)
		_, _ = w.Write([]byte("denied"))
	})
	h := startEngagementProcessHarnessOptions(t, secure.caFile, "pass", nil, false, func(root string) {
		if !withFinding {
			return
		}
		path := filepath.Join(root, "agent-templates", "engagement_fixture.yaml")
		body, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		content := strings.Replace(string(body), "  sandboxProfile:", "    - ref: security-findings@1\n      tools: [finding]\n  sandboxProfile:", 1)
		if err := os.WriteFile(path, []byte(content), 0600); err != nil {
			t.Fatal(err)
		}
	})
	pool := engagementPool(t, h)
	paths := []string{target.server.URL + "/api/noise", target.server.URL + "/api/selected"}
	audit := startEngagementMatrixAudit(t, h, []string{target.server.URL}, paths, []engagementcontract.Route{engagementRoute(target.server.URL, "/api")}, nil, 1024, func() {
		stage := engagementRequestStage([]map[string]any{{"url": paths[0]}, {"url": paths[1]}})
		stage.steps[len(stage.steps)-2] = toolGatewayStep("submit_check_result", auditResultArguments(map[string]any{
			"rationale": "Selected HTTP response denies access in these conditions.", "tested_conditions": []string{"anonymous HTTP request"}, "limitations": []string{}, "evidence": []string{}, "http_request_ids": []int{2},
		}))
		if withFinding {
			final := stage.steps[len(stage.steps)-1]
			stage.tools = append(stage.tools, "finding")
			stage.steps = append(stage.steps[:len(stage.steps)-1], toolGatewayStep("finding", fixedArguments(map[string]any{
				"title": "HTTP denial observation", "description": "A retained anonymous HTTP request received 403; fixture problem publication is explicit.",
				"request_id": 2,
			})), toolGatewayStep("submit_check_result", func(request map[string]any) (map[string]any, error) {
				replies := findingsToolResponses(request, "finding")
				if len(replies) != 1 || replies[0]["receipt_id"] == nil || replies[0]["finding_id"] == nil {
					return nil, fmt.Errorf("Finding client did not accept selected HTTP receipt: %+v", replies)
				}
				return map[string]any{
					"rationale": "Selected HTTP response denies access in these conditions.", "tested_conditions": []string{"anonymous HTTP request"}, "limitations": []string{}, "evidence": []string{}, "http_request_ids": []int{2},
				}, nil
			}), final)
		}
		h.gateway.mu.Lock()
		defer h.gateway.mu.Unlock()
		h.gateway.stages = []domainGatewayStage{stage}
	})
	waitEngagementAuditClosed(t, h, pool, audit)
	tasks := commonPreparedTaskPages(t, h.findingsHarness, audit)
	if len(tasks) != 1 || tasks[0].Result == nil || tasks[0].GapCode != nil {
		t.Fatalf("selected result not accepted: %+v; gateway %v", tasks, h.gateway.Failures())
	}
	result := basicTaskArtifact(t, h.findingsHarness, audit, tasks[0], "result")
	var document struct {
		Schema string `json:"schema"`
		Items  []struct {
			Evidence []contracts.MaterialDescriptor `json:"evidence"`
		} `json:"items"`
	}
	if err := json.Unmarshal(result, &document); err != nil || document.Schema != "contractor.audit.result.v1" || len(document.Items) != 1 || len(document.Items[0].Evidence) != 1 {
		t.Fatalf("invalid retained result: %s %v", result, err)
	}
	material := document.Items[0].Evidence[0]
	if material.ScopeKind != "audit" || material.ScopeID != audit || material.MediaType != auditdomain.HTTPExchangeMediaType {
		t.Fatalf("foreign evidence: %+v", material)
	}
	descriptor, _ := strictjson.Canonical(material)
	key := "evidence/" + strings.TrimPrefix(contentdigest.Bytes(descriptor), "sha256:")
	read := func() []byte {
		path := "/v1/audits/" + audit + "/artifacts?" + url.Values{"key": {key}, "revision": {*material.Ref.Revision}}.Encode()
		request, err := http.NewRequestWithContext(h.ctx, http.MethodGet, h.baseURL+path, nil)
		if err != nil {
			t.Fatal(err)
		}
		request.Header.Set("Authorization", "Bearer "+publicToken)
		response := do(t, h.client, request, http.StatusOK)
		defer response.Body.Close()
		data, err := io.ReadAll(io.LimitReader(response.Body, 8<<20))
		if err != nil {
			t.Fatal(err)
		}
		return data
	}
	payload := read()
	envelope, err := auditdomain.DecodeHTTPExchangeArtifact(payload)
	if err != nil || envelope.Capture.RequestID != "2" || len(envelope.Capture.Attempts) != 2 || envelope.Capture.Attempts[0].Response.Status != 302 || envelope.Capture.Attempts[1].Response.Status != 403 {
		t.Fatalf("capture does not match target: %+v %v", envelope, err)
	}
	if envelope.Capture.ObservedAt == "" || contentdigest.Bytes(payload) != material.Digest {
		t.Fatal("missing observation or changed retained bytes")
	}
	var captures, proposals int
	if err := pool.QueryRow(h.ctx, `SELECT count(*) FROM artifact_bindings WHERE scope_kind='run' AND scope_id=$1 AND namespace='http-exchanges'`, envelope.Origin.RunID).Scan(&captures); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(h.ctx, `SELECT count(*) FROM finding_proposal_receipts WHERE run_id=$1`, envelope.Origin.RunID).Scan(&proposals); err != nil {
		t.Fatal(err)
	}
	expectedProposals := 0
	if withFinding {
		expectedProposals = 1
	}
	if captures != 1 || proposals != expectedProposals {
		t.Fatalf("saved unselected traffic or invented proposal: captures=%d proposals=%d", captures, proposals)
	}
	deleteASVSBacktraceRun(t, h.ctx, h.client, h.baseURL, envelope.Origin.RunID)
	if !bytes.Equal(payload, read()) || len(target.requests()) != 3 {
		t.Fatal("source cleanup changed evidence or repeated target traffic")
	}
}
