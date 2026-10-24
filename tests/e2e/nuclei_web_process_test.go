//go:build e2e

package e2e

import (
	"bytes"
	"encoding/json"
	"fmt"
	"github.com/grauwolf32/contractor/internal/cli"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/scanstore"
	"github.com/grauwolf32/contractor/internal/strictjson"
)

// Use the shipped Workflow, Server, Scheduler and Runtime with an installed
// Nuclei engine. Only the operator package pin changes to the local fixture.
func TestNucleiWebAcrossProductionProcesses(t *testing.T) {
	runNucleiWebProcessAcceptance(t, false)
}

func TestNucleiWebWithDeterministicScannerAcrossProcesses(t *testing.T) {
	runNucleiWebProcessAcceptance(t, true)
}

func runNucleiWebProcessAcceptance(t *testing.T, deterministic bool) {
	if testing.Short() {
		t.Fatal("Nuclei web process acceptance cannot run in short mode")
	}
	binaries := []string{}
	if !deterministic {
		binaries = append(binaries, "nuclei")
	}
	h := startConfiguredScanStack(t, "", func(h *scanProcessHarness, root string) {
		if deterministic {
			binaryRoot := filepath.Join(h.temporaryRoot, "fixture-bin")
			if err := os.Mkdir(binaryRoot, 0700); err != nil {
				t.Fatal(err)
			}
			python := os.Getenv("CONTRACTOR_SCAN_PYTHON")
			if python == "" {
				python = filepath.Join(h.repositoryRoot, "runtime/.venv/bin/python")
			}
			source := "#!" + python + "\n" + `import json, pathlib, sys, urllib.request
args = sys.argv[1:]
if '-version' in args:
    print('Nuclei Engine Version: v3.11.1')
elif '-validate' in args:
    root = pathlib.Path(args[args.index('-t') + 1])
    if any(b'contractor-invalid-matcher' in p.read_bytes() for p in root.glob('*.yaml')):
        sys.exit(2)
elif '-tl' in args:
    root = pathlib.Path(args[args.index('-t') + 1])
    print('\n'.join(str(path.relative_to(root)) for path in sorted(root.rglob('*.yaml'))))
else:
    target = pathlib.Path(args[args.index('-l') + 1]).read_text().strip()
    with urllib.request.urlopen(target.rstrip('/') + '/nuclei', timeout=3) as response:
        body = response.read(1024)
    if b'contractor-controlled-scan-fixture' not in body:
        raise RuntimeError('unexpected loopback fixture')
    print(json.dumps({'template-id': 'contractor-local-fixture'}))
`
			if err := os.WriteFile(filepath.Join(binaryRoot, "nuclei"), []byte(source), 0700); err != nil {
				t.Fatal(err)
			}
			h.runtimePathPrefix = binaryRoot
		}
		for _, relative := range []string{"workflows/nuclei_web.yaml", "agent-templates/nuclei_web_scan.yaml", "instructions/nuclei-web.md"} {
			data, err := os.ReadFile(filepath.Join(h.repositoryRoot, "configs", relative))
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(root, relative), data, 0600); err != nil {
				t.Fatal(err)
			}
		}
		path := filepath.Join(h.temporaryRoot, "nuclei-templates", "fixture.yaml")
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		data = bytes.Replace(data, []byte("severity: info"), []byte("severity: info\n  tags: contractor,fixture"), 1)
		if err := os.WriteFile(path, data, 0600); err != nil {
			t.Fatal(err)
		}
		manifest := map[string]any{"schemaVersion": 1, "files": []map[string]any{{"path": "fixture.yaml", "digest": contentdigest.Bytes(data), "size": len(data)}}}
		canonical, err := strictjson.Canonical(manifest)
		if err != nil {
			t.Fatal(err)
		}
		replaceScanConfig(t, filepath.Join(root, "agent-templates/nuclei_web_scan.yaml"), "sha256:51a3bd4eef14e95bdeb0bad14ca4bd9bd455479a7044b244b76dee8f6dba5014", contentdigest.Bytes(canonical))
	}, binaries...)
	knobs := []string{"omitted", "template_ids", "tags", "severity", "rate_limit"}
	if deterministic {
		knobs = []string{"omitted"}
	}
	for _, key := range knobs {
		t.Run(key, func(t *testing.T) {
			parameters := map[string]string{"target": h.targetURL}
			switch key {
			case "template_ids":
				parameters[key] = "contractor-local-fixture"
			case "tags":
				parameters[key] = "contractor"
			case "severity":
				parameters[key] = "info"
			case "rate_limit":
				parameters[key] = "1000"
			}
			id := h.createRun(t, "nuclei-web@1", parameters, nil)
			status, report := h.completedReport(t, id, "scan_nuclei")
			h.assertExecution(t, status, "scan_nuclei", "")
			manifest, media := download(t, h.client, h.baseURL+"/v1/runs/"+id+"/outputs/template_manifest")
			var provenance struct {
				SchemaVersion         int    `json:"schemaVersion"`
				PackageDigest         string `json:"packageDigest"`
				EngineVersion         string `json:"engineVersion"`
				SelectedTemplateCount int    `json:"selectedTemplateCount"`
				Source                struct {
					Kind string `json:"kind"`
				} `json:"source"`
			}
			if err := json.Unmarshal(manifest, &provenance); err != nil {
				t.Fatal(err)
			}
			var fields map[string]json.RawMessage
			if err := json.Unmarshal(manifest, &fields); err != nil {
				t.Fatal(err)
			}
			if media != "application/vnd.contractor.nuclei-manifest+json" || provenance.SchemaVersion != 2 || provenance.Source.Kind != "installed" || provenance.SelectedTemplateCount != 1 || !strings.HasPrefix(provenance.EngineVersion, "v") || fields["files"] != nil || fields["selectedTemplates"] != nil {
				t.Fatalf("incomplete provenance: %+v", provenance)
			}
			if report.Observation["expectedPackageDigest"] != provenance.PackageDigest || report.Observation["actualPackageDigest"] != provenance.PackageDigest {
				t.Fatal("report does not match package manifest")
			}
			results, ok := report.Observation["results"].([]any)
			if !ok || len(results) != 1 {
				t.Fatalf("scan did not execute local HTTP template: %+v", report.Observation)
			}
		})
	}
	t.Run("invalid_admission", func(t *testing.T) {
		for _, params := range []map[string]string{{"target": h.targetURL, "rate_limit": "0"}, {"target": h.targetURL, "rate_limit": "1001"}, {"target": h.targetURL, "rate_limit": "010"}, {"target": h.targetURL, "severity": "bogus"}, {"target": "file:///tmp/target"}} {
			body, err := json.Marshal(map[string]any{"workflow": "nuclei-web@1", "parameters": params, "artifacts": map[string]any{}})
			if err != nil {
				t.Fatal(err)
			}
			request, err := http.NewRequest(http.MethodPost, h.baseURL+"/v1/runs", bytes.NewReader(body))
			if err != nil {
				t.Fatal(err)
			}
			request.Header.Set("Authorization", "Bearer "+publicToken)
			request.Header.Set("Content-Type", "application/json")
			request.Header.Set("Idempotency-Key", "invalid-"+randomHex(t, 8))
			response := do(t, h.client, request, http.StatusBadRequest)
			response.Body.Close()
		}
	})
	if deterministic {
		t.Run("resumable_project_sweep", func(t *testing.T) {
			runNucleiProjectSweep(t, h)
		})
	}
	t.Run("custom_user_rulepack_without_installation", func(t *testing.T) {
		path := filepath.Join(h.temporaryRoot, "nuclei-templates")
		payload, err := os.ReadFile(filepath.Join(path, "fixture.yaml"))
		if err != nil {
			t.Fatal(err)
		}
		payload = bytes.ReplaceAll(payload, []byte("contractor-local-fixture"), []byte("contractor-custom-fixture"))
		ref := h.upload(t, "custom-rulepack", "application/yaml", payload, nil)
		if err := os.Rename(path, path+".saved"); err != nil {
			t.Fatal(err)
		}
		defer func() {
			if err := os.Rename(path+".saved", path); err != nil {
				t.Error(err)
			}
		}()
		id := h.createRun(t, "nuclei-web@1", map[string]string{"target": h.targetURL}, map[string]artifactRef{"rulepack": ref})
		// The admitted Run reads its fork even after the owner advances the source.
		h.upload(t, "custom-rulepack", "application/yaml", []byte("invalid replacement"), ref.Revision)
		_, report := h.completedReport(t, id, "scan_nuclei")
		if report.Observation["source"] != "rulepack" {
			t.Fatalf("lost custom mode: %+v", report.Observation)
		}
		manifest, _ := download(t, h.client, h.baseURL+"/v1/runs/"+id+"/outputs/template_manifest")
		var provenance struct {
			SchemaVersion int `json:"schemaVersion"`
			Files         []struct {
				Digest string `json:"digest"`
			} `json:"files"`
			Selected []struct {
				ID     string `json:"id"`
				Digest string `json:"digest"`
			} `json:"selectedTemplates"`
			Source struct {
				Kind          string `json:"kind"`
				ContentDigest string `json:"contentDigest"`
			} `json:"source"`
		}
		if err := json.Unmarshal(manifest, &provenance); err != nil {
			t.Fatal(err)
		}
		if provenance.SchemaVersion != 2 || provenance.Source.Kind != "rulepack" || len(provenance.Files) != 1 || len(provenance.Selected) != 1 || provenance.Selected[0].ID != "contractor-custom-fixture" || provenance.Selected[0].Digest != provenance.Files[0].Digest || provenance.Files[0].Digest != contentdigest.Bytes(payload) || provenance.Source.ContentDigest != contentdigest.Bytes(payload) {
			t.Fatalf("wrong custom provenance: %+v", provenance)
		}
		before := len(scanFixtureRequests(h))
		bad := h.upload(t, "invalid-rulepack", "application/yaml", []byte("id: bad\nhttp: []\n"), nil)
		id = h.createRun(t, "nuclei-web@1", map[string]string{"target": h.targetURL}, map[string]artifactRef{"rulepack": bad})
		h.terminal(t, id, "failed")
		if len(scanFixtureRequests(h)) != before {
			t.Fatal("invalid rulepack reached target")
		}
		// Structurally valid HTTP rules still require complete native syntax validation.
		nativeBad := bytes.ReplaceAll(payload, []byte("type: word"), []byte("type: contractor-invalid-matcher"))
		if bytes.Equal(payload, nativeBad) {
			t.Fatal("native validation fixture did not change")
		}
		bad = h.upload(t, "native-invalid-rulepack", "application/yaml", nativeBad, nil)
		id = h.createRun(t, "nuclei-web@1", map[string]string{"target": h.targetURL}, map[string]artifactRef{"rulepack": bad})
		h.terminal(t, id, "failed")
		if len(scanFixtureRequests(h)) != before {
			t.Fatal("native-invalid pack reached target")
		}

	})
	t.Run("changed_package_never_scans", func(t *testing.T) {
		before := len(scanFixtureRequests(h))
		path := filepath.Join(h.temporaryRoot, "nuclei-templates", "fixture.yaml")
		stream, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0600)
		if err != nil {
			t.Fatal(err)
		}
		_, err = stream.WriteString("\n# content changed under the same ID\n")
		if err != nil {
			t.Fatal(err)
		}
		stream.Close()
		id := h.createRun(t, "nuclei-web@1", map[string]string{"target": h.targetURL}, nil)
		h.terminal(t, id, "failed")
		if len(scanFixtureRequests(h)) != before {
			t.Fatal("mismatched package reached the target")
		}
	})
}

func runNucleiProjectSweep(t *testing.T, h *scanProcessHarness) {
	t.Helper()
	projects := make([]projectResourceResponse, 0, 7)
	for index := 0; index < 7; index++ {
		request, err := http.NewRequest(http.MethodPost, h.baseURL+"/v1/projects", strings.NewReader(`{"kind":"project","name":"Nuclei sweep fixture","description":"controlled application"}`))
		if err != nil {
			t.Fatal(err)
		}
		request.Header.Set("Authorization", "Bearer "+publicToken)
		request.Header.Set("Content-Type", "application/json")
		request.Header.Set("Idempotency-Key", fmt.Sprintf("sweep-project-%d", index))
		response := do(t, h.client, request, http.StatusCreated)
		var project projectResourceResponse
		decodeResponse(t, response, &project)
		response.Body.Close()
		if index < 3 {
			project = updateProjectTarget(t, h.client, h.baseURL, project.ProjectID, project.Revision, map[string]any{"url": h.targetURL}, http.StatusOK)
		}
		projects = append(projects, project)
	}
	var lost atomic.Bool
	var creates atomic.Int32
	proxy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(io.LimitReader(r.Body, 16*1024+1))
		if err != nil {
			t.Error(err)
			return
		}
		request, err := http.NewRequestWithContext(r.Context(), r.Method, h.baseURL+r.URL.RequestURI(), bytes.NewReader(body))
		if err != nil {
			t.Error(err)
			return
		}
		request.Header = r.Header.Clone()
		response, err := h.client.Do(request)
		if err != nil {
			t.Error(err)
			return
		}
		defer response.Body.Close()
		if r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/runs") {
			creates.Add(1)
			if lost.CompareAndSwap(false, true) {
				_, _ = io.Copy(io.Discard, response.Body)
				connection, _, err := w.(http.Hijacker).Hijack()
				if err != nil {
					t.Error(err)
				} else {
					connection.Close()
				}
				return
			}
		}
		for key, values := range response.Header {
			w.Header()[key] = values
		}
		w.WriteHeader(response.StatusCode)
		_, _ = io.Copy(w, response.Body)
	}))
	defer proxy.Close()
	manifest := filepath.Join(h.temporaryRoot, "sweep.json")
	invoke := func(action string) []struct {
		ProjectID string `json:"projectId"`
		State     string `json:"state"`
		RunID     string `json:"runId"`
	} {
		var stdout, stderr bytes.Buffer
		command := cli.New(strings.NewReader(""), &stdout, &stderr, func(name string) string {
			if name == "CONTRACTOR_API_TOKEN" {
				return publicToken
			}
			return ""
		})
		args := []string{"--server", proxy.URL, "--output", "json", "scan-sweep", action, manifest}
		if action == "create" {
			args = append(args, "--all", "--param", "template_ids=contractor-local-fixture")
		}
		if err := command.Run(h.ctx, args); err != nil {
			t.Fatalf("sweep %s: %v (%s)", action, err, stderr.String())
		}
		var outcomes []struct {
			ProjectID string `json:"projectId"`
			State     string `json:"state"`
			RunID     string `json:"runId"`
		}
		if err := json.Unmarshal(stdout.Bytes(), &outcomes); err != nil {
			t.Fatal(err)
		}
		return outcomes
	}
	outcomes := invoke("create")
	if len(outcomes) != 7 || !lost.Load() || creates.Load() != 4 {
		t.Fatalf("sweep did not replay lost acknowledgement: members=%d requests=%d", len(outcomes), creates.Load())
	}
	created, skipped := 0, 0
	for _, member := range outcomes {
		if member.State == "skipped" {
			skipped++
			continue
		}
		if member.State != "created" {
			t.Fatalf("unexpected member outcome %+v", member)
		}
		status, _ := h.completedReport(t, member.RunID, "scan_nuclei")
		if status.ProjectID == nil || *status.ProjectID != member.ProjectID {
			t.Fatal("Run lost Project membership")
		}
		created++
	}
	if created != 3 || skipped != 4 {
		t.Fatalf("created/skipped = %d/%d", created, skipped)
	}
	updateProjectTarget(t, h.client, h.baseURL, projects[0].ProjectID, projects[0].Revision, map[string]any{"url": h.targetURL + "/changed"}, http.StatusOK)
	before := len(scanFixtureRequests(h))
	_ = invoke("resume")
	if creates.Load() != 4 || len(scanFixtureRequests(h)) != before {
		t.Fatal("resume repeated a settled Run or scan")
	}
	runDurableNucleiSweep(t, h, []string{projects[1].ProjectID, projects[4].ProjectID})
}

func runDurableNucleiSweep(t *testing.T, h *scanProcessHarness, projectIDs []string) {
	t.Helper()
	payload, err := os.ReadFile(filepath.Join(h.temporaryRoot, "nuclei-templates", "fixture.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	source := h.upload(t, "durable-sweep-pack", "application/yaml", payload, nil)
	var lost, replayed atomic.Bool
	proxy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		request, err := http.NewRequestWithContext(r.Context(), r.Method, h.baseURL+r.URL.RequestURI(), r.Body)
		if err != nil {
			t.Error(err)
			return
		}
		request.Header = r.Header.Clone()
		response, err := h.client.Do(request)
		if err != nil {
			t.Error(err)
			return
		}
		defer response.Body.Close()
		if r.Method == http.MethodPost && r.URL.Path == "/v1/scan-sweeps" {
			if lost.CompareAndSwap(false, true) {
				if response.StatusCode != http.StatusCreated {
					t.Errorf("lost acknowledgement status=%d", response.StatusCode)
				}
				_, _ = io.Copy(io.Discard, response.Body)
				connection, _, err := w.(http.Hijacker).Hijack()
				if err != nil {
					t.Error(err)
				} else {
					connection.Close()
				}
				return
			}
			if response.Header.Get("Idempotency-Replayed") == "true" {
				replayed.Store(true)
			}
		}
		for key, values := range response.Header {
			w.Header()[key] = values
		}
		w.WriteHeader(response.StatusCode)
		_, _ = io.Copy(w, response.Body)
	}))
	defer proxy.Close()
	invoke := func(destination any, args ...string) {
		t.Helper()
		var stdout, stderr bytes.Buffer
		command := cli.New(strings.NewReader(""), &stdout, &stderr, func(name string) string {
			if name == "CONTRACTOR_API_TOKEN" {
				return publicToken
			}
			return ""
		})
		if err := command.Run(h.ctx, append([]string{"--server", proxy.URL, "--output", "json", "scan-sweep", "remote"}, args...)); err != nil {
			t.Fatalf("remote sweep %v: %v (%s)", args, err, stderr.String())
		}
		if destination != nil {
			if err := json.Unmarshal(stdout.Bytes(), destination); err != nil {
				t.Fatalf("remote sweep output: %v (%s)", err, stdout.String())
			}
		}
	}
	createArgs := []string{"create", "--workflow", "nuclei-web@1", "--param", "template_ids=contractor-local-fixture", "--rulepack", source.Namespace + "/" + source.Name + "@" + *source.Revision, "--idempotency-key", "durable-sweep-create"}
	for _, id := range projectIDs {
		createArgs = append(createArgs, "--project", id)
	}
	var sweep scanstore.Sweep
	invoke(&sweep, createArgs...)
	if !lost.Load() || !replayed.Load() {
		t.Fatal("durable CLI did not replay committed creation after lost acknowledgement")
	}
	h.upload(t, "durable-sweep-pack", "application/yaml", []byte("invalid replacement"), source.Revision)
	var members []scanstore.Member
	runID := ""
	waitScanCondition(t, h, "durable sweep ordinary child admission", func() bool {
		invoke(&members, "members", sweep.SweepID, "--limit", "1", "--all-pages")
		for _, m := range members {
			if m.RunID != nil {
				runID = *m.RunID
				return true
			}
		}
		return false
	})
	if len(members) != 2 {
		t.Fatalf("durable members=%+v", members)
	}
	_, report := h.completedReport(t, runID, "scan_nuclei")
	if report.Observation["source"] != "rulepack" || report.Observation["sourceContentDigest"] != contentdigest.Bytes(payload) {
		t.Fatalf("durable sweep lost retained source: %+v", report.Observation)
	}
	waitScanCondition(t, h, "durable sweep terminal settlement", func() bool {
		invoke(&sweep, "show", sweep.SweepID)
		return sweep.State == "completed"
	})
	invoke(&members, "members", sweep.SweepID, "--all-pages")
	for _, member := range members {
		if member.RunID != nil && (member.RunState == nil || *member.RunState != "succeeded" || !member.Settled) {
			t.Fatalf("CLI lost settled child state: %+v", member)
		}
	}
	var page struct {
		Items []scanstore.Sweep `json:"items"`
	}
	invoke(&page, "list")
	if len(page.Items) != 1 || page.Items[0].SweepID != sweep.SweepID {
		t.Fatalf("durable list=%+v", page)
	}
	before := len(scanFixtureRequests(h))
	var replay scanstore.Sweep
	invoke(&replay, createArgs...)
	if replay.SweepID != sweep.SweepID {
		t.Fatal("durable replay created another sweep")
	}
	invoke(&sweep, "cancel", sweep.SweepID)
	if sweep.State != "completed" {
		t.Fatal("CLI changed a completed sweep's cancellation state")
	}
	invoke(nil, "purge", sweep.SweepID, "--yes")
	if len(scanFixtureRequests(h)) != before {
		t.Fatal("durable replay or cleanup rescanned target")
	}
	// Child report and input lifecycle are independent of the sweep's purge.
	h.completedReport(t, runID, "scan_nuclei")
}
