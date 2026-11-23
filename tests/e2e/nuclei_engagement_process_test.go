//go:build e2e

package e2e

import (
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"sync/atomic"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/engagementcontract"
)

// The actual engine establishes that its proxy, cookie and retry paths use
// the allocation owner; a deterministic scanner fixture cannot prove this.
func TestNucleiEngagementAcrossProductionProcesses(t *testing.T) {
	if _, err := exec.LookPath("nuclei"); err != nil {
		t.Fatal("real Nuclei is required on PATH")
	}
	secure := newEngagementTargetFixture(t)
	target := newEngagementHTTPObserver(t, engagementFixtureIP(t, secure))
	templates := filepath.Join(t.TempDir(), "templates")
	if err := os.Mkdir(templates, 0o700); err != nil {
		t.Fatal(err)
	}
	const rule = "id: scoped-http\ninfo:\n  name: Allocation HTTP fixture\n  author: contractor\n  severity: info\nhttp:\n  - method: GET\n    path: [\"{{RootURL}}/allowed/nuclei\"]\n    matchers:\n      - type: status\n        status: [200]\n"
	if err := os.WriteFile(filepath.Join(templates, "fixture.yaml"), []byte(rule), 0o600); err != nil {
		t.Fatal(err)
	}
	h := startEngagementProcessHarnessOptions(
		t, secure.caFile, "pass", map[string]string{"NUCLEI_TEMPLATES_DIR": templates}, true,
		func(root string) {
			replaceScanConfig(t, filepath.Join(root, "agent-templates/engagement_fixture.yaml"),
				"  sandboxProfile: local-workdir@1",
				"    - ref: scan@1\n      tools: [scan_nuclei]\n  sandboxProfile: local-workdir@1")
		},
	)
	pool := engagementPool(t, h)
	const secret = "private-real-nuclei-bearer-canary"
	credential := engagementCredential(t, h, "nuclei-account", "target-api-keys@1",
		map[string]any{"fields": map[string]string{
			"header:Authorization": "Bearer " + secret, "query:api_key": secret, "cookie:client": secret,
		}}, secret)
	var invalidHeaders atomic.Int64
	target.setHandler(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer "+secret || r.Header.Get("Proxy-Authorization") != "" || r.URL.Query().Get("api_key") != secret {
			invalidHeaders.Add(1)
		}
		if cookie, err := r.Cookie("client"); err != nil || cookie.Value != secret {
			invalidHeaders.Add(1)
		}
		w.WriteHeader(http.StatusOK)
	})
	f := newEngagementIdentityAudit(t, h, []string{target.server.URL}, nil)
	f.document.Egress.Routes = []engagementcontract.Route{engagementRoute(target.server.URL, "/allowed")}
	requests := uint32(2)
	f.document.Limits.RequestsPerSecond = &requests
	f.addIdentity(t, "scanner", "api-keys", credential)
	draft := f.draft(t)
	stage := engagementRequestStage(nil)
	stage.name = "engagement/real-nuclei"
	stage.tools = append(stage.tools, "scan_nuclei")
	stage.steps = append(stage.steps[:2],
		toolGatewayStep("http_request", fixedArguments(map[string]any{
			"url": target.server.URL + "/allowed/http", "credential_id": credential.CredentialID,
		})),
		toolGatewayStep("scan_nuclei", fixedArguments(map[string]any{
			"url": target.server.URL + "/allowed", "template_ids": "scoped-http",
			"credential_id": credential.CredentialID, "timeout_seconds": 30,
		})),
		toolGatewayStep("http_request", fixedArguments(map[string]any{
			"url": target.server.URL + "/allowed/next", "credential_id": credential.CredentialID,
		})),
		stage.steps[len(stage.steps)-2], stage.steps[len(stage.steps)-1],
	)
	h.gateway.mu.Lock()
	h.gateway.stages = []domainGatewayStage{stage}
	h.gateway.stageIndex, h.gateway.stepIndex = 0, 0
	h.gateway.mu.Unlock()
	engagementAuditPOST(t, h, "/v1/audits/"+draft.AuditID+"/start", nil,
		draft.Revision, http.StatusOK, nil)
	select {
	case <-h.gateway.blockedRequest():
	case <-h.ctx.Done():
		t.Fatal("Audit Worker did not reach Gateway")
	}
	if len(target.requests()) != 0 {
		t.Fatal("preparation issued target traffic")
	}
	h.gateway.releaseBlockedRequest()
	waitForAuditProgram(t, h.ctx, h.server, h.runtimeProcess, h.gateway, h.client,
		h.baseURL, draft.AuditID, 1)
	if got := target.requests(); !slices.Equal(got, []string{"/allowed/http?api_key=" + secret, "/allowed/nuclei?api_key=" + secret, "/allowed/next?api_key=" + secret}) {
		t.Fatalf("HTTP tool and Nuclei did not share the authorized transport: %v", got)
	}
	if invalidHeaders.Load() != 0 {
		t.Fatal("assigned credential or private proxy header isolation failed")
	}
	assertEngagementSecretsAbsent(t, h, pool)
	h.proxy.verify(t)
}
