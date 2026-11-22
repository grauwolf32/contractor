//go:build e2e

package e2e

import (
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/engagementcontract"
)

func TestEngagementOrdinaryRunAcrossProcesses(t *testing.T) {
	target := newEngagementTargetFixture(t)
	observer := newEngagementHTTPObserver(t, engagementFixtureIP(t, target))
	const token = "ordinary-run-private-bearer-canary"
	observer.setHandler(func(w http.ResponseWriter, r *http.Request) {
		if (r.URL.Path != "/api/ok" && r.URL.Path != "/api/next") || r.Header.Get("Authorization") != "Bearer "+token {
			t.Error("ordinary Run violated selected route or identity")
		}
		w.WriteHeader(http.StatusNoContent)
	})
	h := startEngagementProcessHarnessOptions(t, target.caFile, "pass", nil, true, func(root string) {
		for source, dest := range map[string]string{"run-agent.yaml": "agent-templates/engagement_run.yaml", "run-workflow.yaml": "workflows/engagement_run.yaml"} {
			body, err := os.ReadFile(filepath.Join(repoRoot(t), "tests/e2e/testdata/engagement", source))
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(root, dest), body, 0600); err != nil {
				t.Fatal(err)
			}
		}
		if err := os.WriteFile(filepath.Join(root, "instructions/engagement-run-fixture.md"), []byte("Make the explicit fixture HTTP requests and write the bounded result Artifact."), 0600); err != nil {
			t.Fatal(err)
		}
	})
	credential := engagementCredential(t, h, "ordinary-account", "target-bearer@1", map[string]string{"token": token}, token)
	limit := uint32(1)
	document := engagementcontract.Document{Schema: engagementcontract.Schema,
		Egress: engagementcontract.Egress{Routes: []engagementcontract.Route{engagementRoute(observer.server.URL, "/api")}, Exclusions: []engagementcontract.Route{}},
		Identities: []engagementcontract.Identity{
			{ID: "anonymous", Kind: "anonymous", Origins: []string{observer.server.URL}},
			{ID: "account", Kind: "bearer", Origins: []string{observer.server.URL}, Credential: credential},
		}, Limits: &engagementcontract.HTTPLimits{RequestsPerSecond: &limit}}
	body, err := json.Marshal(document)
	if err != nil {
		t.Fatal(err)
	}
	policy := uploadProjectArtifact(t, h.client, h.baseURL, "ordinary-policy", engagementcontract.MediaType, body)
	steps := []domainGatewayStep{toolGatewayStep("http_credentials", fixedArguments(map[string]any{}))}
	for _, path := range []string{"/forbidden", "/api/ok", "/api/next"} {
		steps = append(steps, toolGatewayStep("http_request", fixedArguments(map[string]any{"url": observer.server.URL + path, "credential_id": credential.CredentialID})))
	}
	steps = append(steps, toolGatewayStep("write_text_artifact", fixedArguments(map[string]any{
		"name": "result", "media_type": "text/plain",
		"text": "ordinary HTTP policy verified", "expected_revision": nil,
	})), finalGatewayStep("Ordinary Run complete", map[string]domainArtifactBinding{"result": {namespace: "engagement-run", name: "result"}}))
	h.gateway.mu.Lock()
	h.gateway.stages = []domainGatewayStage{{name: "ordinary-engagement", tools: []string{"http_request", "http_credentials", "write_text_artifact"}, steps: steps}}
	h.gateway.stageIndex, h.gateway.stepIndex = 0, 0
	h.gateway.mu.Unlock()
	var run runCreateResponse
	engagementAuditPOST(t, h, "/v1/runs", map[string]any{"workflow": "engagement-ordinary-run@1", "artifacts": map[string]artifactRef{"network_policy": policy}}, 0, http.StatusAccepted, &run)
	select {
	case <-h.gateway.blockedRequest():
	case <-h.ctx.Done():
		t.Fatal("ordinary Worker did not reach its first model call")
	}
	deleteCredential := func(status int) {
		t.Helper()
		request, err := http.NewRequestWithContext(h.ctx, http.MethodDelete, h.baseURL+"/v1/operations/runtime-credentials/"+credential.CredentialID, nil)
		if err != nil {
			t.Fatal(err)
		}
		request.Header.Set("Authorization", "Bearer "+publicToken)
		response := do(t, h.client, request, status)
		_ = response.Body.Close()
	}
	deleteCredential(http.StatusConflict)
	h.gateway.releaseBlockedRequest()
	waitForRun(t, h.ctx, h.server, h.runtimeProcess, h.gateway, h.client, h.baseURL, run.RunID)
	pool := engagementPool(t, h)
	waitEngagementCondition(t, h.ctx, func() bool {
		var released bool
		err := pool.QueryRow(h.ctx, `SELECT NOT EXISTS (SELECT 1 FROM stage_allocations a JOIN stage_executions s USING(stage_execution_id) WHERE s.run_id=$1 AND a.release_completed_at IS NULL)`, run.RunID).Scan(&released)
		if err != nil {
			t.Fatal(err)
		}
		return released
	})
	requests := observer.requests()
	if len(requests) != 2 || requests[0] != "/api/ok" || requests[1] != "/api/next" {
		t.Fatalf("policy admitted unexpected requests: %v", requests)
	}
	deleteCredential(http.StatusNoContent)
	// Keeping the policy Artifact after execution must not retain a credential.
	var audits int
	if err := pool.QueryRow(h.ctx, `SELECT count(*) FROM audits`).Scan(&audits); err != nil || audits != 0 {
		t.Fatal("ordinary Run acquired Audit ownership")
	}
}
