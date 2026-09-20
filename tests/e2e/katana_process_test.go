//go:build e2e

package e2e

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/scanplan"
)

func TestKatanaDiscoveryAcrossProductionProcesses(t *testing.T) {
	if testing.Short() {
		t.Skip("end-to-end process test")
	}
	h := startScanStackForTools(t, "katana-discovery@1", "katana")
	var foreignRequests atomic.Int32
	foreign := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		foreignRequests.Add(1)
		_, _ = io.WriteString(w, "outside the seed origin")
	}))
	t.Cleanup(foreign.Close)
	var mutex sync.Mutex
	var requests []string
	seed := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mutex.Lock()
		requests = append(requests, r.URL.RequestURI())
		mutex.Unlock()
		w.Header().Set("Content-Type", "text/html")
		if r.URL.Path == "/" {
			_, _ = fmt.Fprintf(w, `<html><body><a href="/alpha?x=2">alpha</a><a href="/beta">beta</a><a href="/beta">duplicate</a><a href="/redirect">redirect</a><a href="%s/outside">other origin</a></body></html>`, foreign.URL)
		} else if r.URL.Path == "/redirect" {
			http.Redirect(w, r, foreign.URL+"/redirected", http.StatusFound)
		} else {
			_, _ = io.WriteString(w, "<html><body>local leaf</body></html>")
		}
	}))
	t.Cleanup(seed.Close)
	var beforeRuns int
	if err := h.pool.QueryRow(h.ctx, "SELECT count(*) FROM workflow_runs").Scan(&beforeRuns); err != nil {
		t.Fatal(err)
	}
	runID := h.createRun(t, "katana-discovery@1", map[string]string{"target": seed.URL + "/"}, nil)
	status, report := h.completedReport(t, runID, "scan_katana")
	h.assertExecution(t, status, "scan_katana", "")
	if len(status.Outputs) != 2 || status.Outputs["targets"].Revision == nil || report.Observation["discoveryComplete"] != false || report.Observation["diagnosticsRedacted"] != true || report.Observation["stdout"] != "" || report.Observation["stderr"] != "" {
		t.Fatalf("discovery must publish both outputs and explicit coverage limits: %+v %+v", status.Outputs, report.Observation)
	}
	data, media := download(t, h.client, h.baseURL+"/v1/runs/"+url.PathEscape(runID)+"/outputs/targets")
	want := seed.URL + "/\n" + seed.URL + "/alpha?x=2\n" + seed.URL + "/beta\n" + seed.URL + "/redirect\n"
	if media != scanplan.TargetListMediaType || string(data) != want {
		t.Fatalf("unexpected discovered input media=%q data=%q want=%q", media, data, want)
	}
	var source artifactRef
	encodedSource, err := json.Marshal(report.Observation["targetsArtifact"])
	if err != nil || json.Unmarshal(encodedSource, &source) != nil || source.Namespace != "scanner" || source.Name != "targets" || source.Revision == nil {
		t.Fatalf("report does not retain exact Worker target-list provenance: %s", encodedSource)
	}
	digest := fmt.Sprintf("sha256:%x", sha256.Sum256(data))
	if report.Observation["targetsDigest"] != digest {
		t.Fatalf("report target digest does not identify exported bytes: %+v", report.Observation)
	}
	var stageResult struct {
		Artifacts map[string]artifactRef `json:"artifacts"`
	}
	if err := json.Unmarshal(status.Attempts[0].Result, &stageResult); err != nil || !reflect.DeepEqual(stageResult.Artifacts["targets"], source) {
		t.Fatalf("targets Stage result lost its exact Worker source: %s", status.Attempts[0].Result)
	}
	pinned, pinnedMedia := download(t, h.client, h.baseURL+"/v1/runs/"+url.PathEscape(runID)+"/artifacts/scanner/targets?revision="+url.QueryEscape(*source.Revision))
	if !bytes.Equal(pinned, data) || pinnedMedia != media {
		t.Fatal("published targets differ from their exact Worker revision")
	}
	results, ok := report.Observation["results"].([]any)
	if !ok || len(results) != 4 {
		t.Fatalf("discovery lost per-URL provenance: %+v", report.Observation)
	}
	for _, raw := range results {
		result, ok := raw.(map[string]any)
		if !ok || result["method"] != "GET" || result["source"] != seed.URL+"/" || !strings.HasPrefix(result["url"].(string), seed.URL+"/") {
			t.Fatalf("invalid discovered URL provenance: %+v", raw)
		}
	}
	if foreignRequests.Load() != 0 {
		t.Fatalf("discovery escaped its exact origin: %d requests", foreignRequests.Load())
	}
	mutex.Lock()
	observed := append([]string(nil), requests...)
	mutex.Unlock()
	sort.Strings(observed)
	if !reflect.DeepEqual(observed, []string{"/", "/alpha?x=2", "/beta", "/redirect"}) {
		t.Fatalf("discovery issued unexpected local requests: %v", observed)
	}
	// Consume the actual exported bytes in the pure planner. This verifies a
	// reusable input without dispatching any follow-up network scans.
	snapshot, err := config.Load(filepath.Join(h.repositoryRoot, "configs", "scan"), config.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	workflow, err := snapshot.Workflow("target-scan-plan@1")
	if err != nil {
		t.Fatal(err)
	}
	stage := workflow.Stages[workflow.EntryStage]
	bindings := make(map[string]scanplan.ToolBinding, len(stage.Agents))
	for name, binding := range stage.Agents {
		bindings[name] = scanplan.ToolBinding{Namespace: binding.Namespace, Template: binding.Template}
	}
	input := scanplan.PlanInput{Artifact: contracts.ArtifactRef{Namespace: source.Namespace, Name: source.Name, Revision: source.Revision}, MediaType: media, Data: data}
	plan, err := scanplan.BuildPlan(input, *stage.ScanPlan, bindings, nil)
	if err != nil {
		t.Fatalf("discovered TargetList is not accepted by scan-plan: %v", err)
	}
	if len(plan.Candidates) != 5 || len(plan.Jobs) != 4 || !reflect.DeepEqual(plan.Source.Artifact, input.Artifact) || plan.Source.ContentDigest != digest {
		t.Fatalf("unexpected pure follow-up plan or lost source identity: %+v", plan)
	}
	planBytes, err := scanplan.MarshalPlan(plan)
	if err != nil {
		t.Fatal(err)
	}
	secondPlan, err := scanplan.BuildPlan(input, *stage.ScanPlan, bindings, nil)
	if err != nil {
		t.Fatal(err)
	}
	secondBytes, err := scanplan.MarshalPlan(secondPlan)
	if err != nil || !bytes.Equal(planBytes, secondBytes) {
		t.Fatal("same discovered input and policy produced a different plan")
	}
	var afterRuns int
	if err := h.pool.QueryRow(h.ctx, "SELECT count(*) FROM workflow_runs").Scan(&afterRuns); err != nil || afterRuns != beforeRuns+1 {
		t.Fatalf("discovery unexpectedly created further Runs: before=%d after=%d error=%v", beforeRuns, afterRuns, err)
	}
	if evidenceRoot := os.Getenv("CONTRACTOR_KATANA_EVIDENCE_DIR"); evidenceRoot != "" {
		if err := os.MkdirAll(evidenceRoot, 0o700); err != nil {
			t.Fatal(err)
		}
		reportBytes, err := json.MarshalIndent(report, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		summary, err := json.MarshalIndent(map[string]any{
			"status": "passed", "runId": runID, "outputs": status.Outputs,
			"sourceTargets": source, "targetsDigest": digest, "requests": observed,
			"outOfScopeRequests": foreignRequests.Load(), "discoveryComplete": false,
			"modelCalls":     status.Attempts[0].Metrics.ModelCalls,
			"planCandidates": len(plan.Candidates), "planJobs": len(plan.Jobs),
			"followupRuns": afterRuns - beforeRuns - 1,
		}, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		for name, value := range map[string][]byte{"report.json": reportBytes, "targets.txt": data, "plan.json": planBytes, "evidence.json": summary} {
			if err := os.WriteFile(filepath.Join(evidenceRoot, name), value, 0o600); err != nil {
				t.Fatal(err)
			}
		}
	}
	t.Logf("Katana Run %s: 4 exact targets, 4 local requests, 0 outside requests, 0 model calls; pure planner accepted 5 candidates/4 jobs without execution", runID)
}
