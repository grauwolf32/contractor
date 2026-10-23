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
				PackageDigest     string `json:"packageDigest"`
				EngineVersion     string `json:"engineVersion"`
				SelectedTemplates []struct {
					ID     string `json:"id"`
					Digest string `json:"digest"`
				} `json:"selectedTemplates"`
			}
			if err := json.Unmarshal(manifest, &provenance); err != nil {
				t.Fatal(err)
			}
			if media != "application/vnd.contractor.nuclei-manifest+json" || len(provenance.SelectedTemplates) != 1 || provenance.SelectedTemplates[0].ID != "contractor-local-fixture" || !strings.HasPrefix(provenance.EngineVersion, "v") {
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
}
