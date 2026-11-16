//go:build e2e

package e2e

import (
	"encoding/json"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"slices"
	"sync/atomic"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/engagementcontract"
	"github.com/grauwolf32/contractor/internal/projectsitemap"
)

func TestFFUFEngagementAcrossProductionProcesses(t *testing.T) {
	version, err := exec.Command("ffuf", "-V").Output()
	if err != nil || string(version) != "ffuf version: 2.3.0\n" {
		t.Fatal("pinned ffuf 2.3.0 is required")
	}
	secure := newEngagementTargetFixture(t)
	target := newEngagementHTTPObserver(t, engagementFixtureIP(t, secure))
	h := startEngagementProcessHarnessOptions(t, secure.caFile, "pass", nil, false, func(root string) {
		for source, destination := range map[string]string{
			"ffuf-agent.yaml":    "agent-templates/engagement_ffuf.yaml",
			"ffuf-workflow.yaml": "workflows/engagement_ffuf.yaml",
		} {
			body, err := os.ReadFile(filepath.Join(repoRoot(t), "tests/e2e/testdata/engagement", source))
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(root, destination), body, 0o600); err != nil {
				t.Fatal(err)
			}
		}
	})
	pool := engagementPool(t, h)
	const secret = "private-ffuf-process-bearer-canary"
	credential := engagementCredential(t, h, "ffuf-account", "target-bearer@1", map[string]string{"token": secret}, secret)
	var invalid atomic.Bool
	target.setHandler(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer "+secret || r.Header.Get("Proxy-Authorization") != "" {
			invalid.Store(true)
		}
		w.WriteHeader(http.StatusOK)
	})
	f := newEngagementIdentityAudit(t, h, []string{target.server.URL}, nil)
	f.document.Egress.Routes = []engagementcontract.Route{engagementRoute(target.server.URL, "/allowed")}
	f.addIdentity(t, "scanner", "bearer", credential)
	body, err := json.Marshal(f.document)
	if err != nil {
		t.Fatal(err)
	}
	policy := uploadProjectScopeArtifact(t, h.client, h.baseURL, f.project.ProjectID, "inputs", "policy", engagementcontract.MediaType, body)
	wordlist := uploadProjectScopeArtifact(t, h.client, h.baseURL, f.project.ProjectID, "inputs", "words", "text/vnd.contractor.wordlist", []byte("one\ntwo\n"))
	for _, scenario := range []struct{ path, state string }{{"allowed", "succeeded"}, {"forbidden", "failed"}} {
		t.Run(scenario.path, func(t *testing.T) {
			var run runCreateResponse
			engagementAuditPOST(t, h, "/v1/projects/"+f.project.ProjectID+"/runs", map[string]any{
				"workflow":   "engagement-ffuf@1",
				"parameters": map[string]string{"target": target.server.URL + "/" + scenario.path + "/FUZZ", "credential_id": credential.CredentialID},
				"artifacts":  map[string]artifactRef{"network_policy": policy, "wordlist": wordlist},
			}, 0, http.StatusAccepted, &run)
			status := waitForRunState(t, h.ctx, h.server, h.runtimeProcess, h.gateway, h.client, h.baseURL, run.RunID, scenario.state)
			if scenario.state == "succeeded" {
				data, _ := download(t, h.client, h.baseURL+"/v1/runs/"+run.RunID+"/outputs/report")
				var report scanProcessReport
				if err := json.Unmarshal(data, &report); err != nil {
					t.Fatal(err)
				}
				if report.Tool != "scan_ffuf" || report.Observation["scanComplete"] != true || report.Observation["wordlistEntries"] != float64(2) {
					t.Fatal("actual ffuf report lost its normalized contract")
				}
				if !reflect.DeepEqual(status.Inputs["wordlist"], report.InputArtifacts["wordlist"]) {
					t.Fatal("ffuf report changed the exact selected wordlist")
				}
				assertEngagementSecretsAbsent(t, h, pool, data)
			}
		})
	}
	if invalid.Load() || !slices.Equal(target.requests(), []string{"/allowed/one", "/allowed/two"}) {
		t.Fatal("ffuf did not preserve scope and assigned credential isolation")
	}
	// Automatic report collection uses the same normalized ffuf adapter, real
	// Server and PostgreSQL. The denied Run cannot create a response observation.
	data, _ := download(t, h.client, h.baseURL+"/v1/projects/"+f.project.ProjectID+"/artifacts/sitemap/map")
	document, err := projectsitemap.Decode(data)
	if err != nil {
		t.Fatal(err)
	}
	var paths []string
	for _, group := range document.Groups {
		for _, member := range group.Members {
			paths = append(paths, member.Path)
			if len(member.Summaries) != 1 || member.Summaries[0].Status != 200 || member.Summaries[0].Provenance.Kind != "ffuf" {
				t.Fatal("actual ffuf report lost normalized response provenance")
			}
		}
	}
	slices.Sort(paths)
	if !slices.Equal(paths, []string{"/allowed/one", "/allowed/two"}) {
		t.Fatal("actual ffuf report did not preserve the exact permitted sitemap members")
	}
	assertEngagementSecretsAbsent(t, h, pool, data)
	h.proxy.verify(t)
}
