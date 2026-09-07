//go:build e2e

package e2e

import (
	"archive/zip"
	"bytes"
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/gitimport"
	"github.com/grauwolf32/contractor/internal/localpki"
	"golang.org/x/crypto/ssh/knownhosts"
)

func TestGitArtifactsProductionContainers(t *testing.T) {
	if testing.Short() {
		t.Skip("production container gate")
	}
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Fatal("CONTRACTOR_TEST_DATABASE_URL is required")
	}
	root, build := repoRoot(t), t.TempDir()
	binary := filepath.Join(build, "contractor-server")
	runChecked(t, root, map[string]string{"CGO_ENABLED": "0"}, "go", "build", "-o", binary, "./cmd/contractor-server")
	if err := os.WriteFile(filepath.Join(build, "Containerfile"), []byte("FROM scratch\nCOPY contractor-server /contractor-server\nENTRYPOINT [\"/contractor-server\"]\n"), 0600); err != nil {
		t.Fatal(err)
	}
	image := "localhost/contractor-git-gate:" + randomHex(t, 8)
	runChecked(t, build, nil, "podman", "build", "-q", "-t", image, build)
	t.Cleanup(func() { _ = exec.Command("podman", "rmi", "-f", image).Run() })
	host := gitGateHost(t)
	for _, backend := range []string{"postgresql", "filesystem"} {
		t.Run(backend, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
			defer cancel()
			work := t.TempDir()
			database := isolatedDatabase(t, ctx, databaseURL)
			runChecked(t, root, map[string]string{"CONTRACTOR_DATABASE_URL": database}, binary, "migrate")
			generator := localpki.Generator{}
			ca, err := generator.InitCA(filepath.Join(work, "pki"), false)
			if err != nil {
				t.Fatal(err)
			}
			leaf := localpki.LeafOptions{IPAddresses: []net.IP{net.ParseIP("127.0.0.1"), net.ParseIP(host)}}
			control, err := generator.IssueControlPlane(filepath.Join(work, "pki"), localpki.ControlPlaneOptions{LeafOptions: leaf, URI: "urn:contractor:control-plane:git-gate"})
			if err != nil {
				t.Fatal(err)
			}
			agent, err := generator.IssueAgent(filepath.Join(work, "pki"), "git-gate", leaf)
			if err != nil {
				t.Fatal(err)
			}
			repo, commit := gitGateRepository(t)
			https := gitGateServeHTTPS(t, repo, host, control.Certificate, control.PrivateKey)
			owner, privateKey := gitGateSigner(t)
			sshAddress, sshHost := gitGateServeSSH(t, repo, host, owner)
			known := filepath.Join(work, "known_hosts")
			trusted := []byte(knownhosts.Line([]string{sshAddress}, sshHost.PublicKey()) + "\n")
			if err := os.WriteFile(known, trusted, 0600); err != nil {
				t.Fatal(err)
			}
			master := filepath.Join(work, "master-key")
			key := make([]byte, 32)
			if _, err := rand.Read(key); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(master, []byte(base64.StdEncoding.EncodeToString(key)), 0600); err != nil {
				t.Fatal(err)
			}
			gateway := newDomainGateway(llmGatewayToken)
			gateway.stages = []domainGatewayStage{{name: "git-workspace", tools: []string{"changed_paths", "diff", "edit", "read_file"}, steps: []domainGatewayStep{
				toolGatewayStep("read_file", fixedArguments(map[string]any{"path": "source.txt", "start_line": 1, "max_lines": 10})),
				toolGatewayStep("edit", fixedArguments(map[string]any{"path": "source.txt", "old": "before", "new": "after", "replace_all": false})),
				finalGatewayStep("Edited the imported Git source", nil),
			}}}
			t.Cleanup(gateway.close)
			configs := stageE2EConfiguration(t, filepath.Join(root, "configs", "e2e"), filepath.Join(work, "configs"), gateway.URL())
			publicAddress, privateAddress := freeAddress(t), freeAddress(t)
			publicURL, privateURL := "http://"+publicAddress, "https://"+privateAddress
			httpsURL := https.server.URL + "/repo.git"
			parsed, _ := url.Parse(httpsURL)
			environment := map[string]string{
				"CONTRACTOR_DATABASE_URL": database, "CONTRACTOR_OPERATOR_CONFIG_ROOT": configs, "CONTRACTOR_MANAGED_CONFIG_ROOT": "/managed",
				"CONTRACTOR_PUBLIC_LISTEN": publicAddress, "CONTRACTOR_PRIVATE_LISTEN": privateAddress, "CONTRACTOR_PRIVATE_URL": privateURL,
				"CONTRACTOR_CA_FILE": ca.Certificate, "CONTRACTOR_CONTROL_PLANE_CERT_FILE": control.Certificate, "CONTRACTOR_CONTROL_PLANE_KEY_FILE": control.PrivateKey,
				"CONTRACTOR_LLM_GATEWAY_TOKEN": llmGatewayToken, "CONTRACTOR_PUBLIC_BEARER_TOKEN": publicToken, "CONTRACTOR_LOCAL_AUTH_FILE": writeE2ELocalAuth(t, work, "git-owner"),
				"CONTRACTOR_BROWSER_ORIGINS": "https://ui.contractor.invalid", "CONTRACTOR_ARTIFACT_BLOB_BACKEND": backend,
				"CONTRACTOR_GIT_ALLOWED_REMOTES": parsed.Host + "," + sshAddress, "CONTRACTOR_GIT_KNOWN_HOSTS_FILE": known, "CONTRACTOR_CREDENTIAL_MASTER_KEY_FILE": master, "SSL_CERT_FILE": ca.Certificate,
			}
			name := "contractor-git-gate-" + randomHex(t, 8)
			args := []string{"run", "--name", name, "--network=host", "--read-only", "--read-only-tmpfs=false", "--cap-drop=all", "--security-opt=no-new-privileges", "--tmpfs=/managed:rw,nosuid,nodev,noexec,size=64m", "-v", work + ":" + work + ":ro"}
			if backend == "filesystem" {
				environment["CONTRACTOR_ARTIFACT_BLOB_PATH"] = "/blobs"
				args = append(args, "--tmpfs=/blobs:rw,nosuid,nodev,noexec,size=256m")
			}
			for key, value := range environment {
				args = append(args, "--env", key+"="+value)
			}
			args = append(args, image, "serve")
			server := startProcess(t, "Git Server "+backend, root, nil, "podman", args...)
			t.Cleanup(func() { _ = exec.Command("podman", "rm", "-f", name).Run() })
			client := &http.Client{Timeout: 130 * time.Second}
			waitForHTTP(t, ctx, server, client, publicURL+"/readyz", 200)
			call := func(method, path string, body any, status int) []byte {
				return gitGateJSON(t, client, publicURL, method, path, body, status)
			}
			keyState := call("PUT", "/v1/settings/git-key", map[string]string{"privateKey": string(privateKey)}, 200)
			if bytes.Contains(keyState, privateKey) || bytes.Contains(keyState, []byte("PRIVATE KEY")) {
				t.Fatal("key readback")
			}
			first := gitGateImport(t, client, publicURL, "/v1/artifacts/sources/https/git-import", httpsURL, "initial", 201)
			sshURL := "ssh://git@" + sshAddress + "/repo.git"
			second := gitGateImport(t, client, publicURL, "/v1/artifacts/sources/ssh/git-import", sshURL, "initial", 201)
			if first.GitSource.ResolvedCommit != commit || second.GitSource.ResolvedCommit != commit {
				t.Fatal("wrong Git commit")
			}
			zipURL := publicURL + "/v1/artifacts/sources/https?revision=" + url.QueryEscape(*first.Artifact.Revision)
			zipBytes, _ := download(t, client, zipURL)
			archive, err := zip.NewReader(bytes.NewReader(zipBytes), int64(len(zipBytes)))
			if err != nil || len(archive.File) != 1 {
				t.Fatalf("source ZIP: %v", err)
			}
			projectJSON := call("POST", "/v1/projects", map[string]any{"kind": "project", "name": "Git project", "description": ""}, 201)
			var project projectResourceResponse
			if err := json.Unmarshal(projectJSON, &project); err != nil {
				t.Fatal(err)
			}
			projectImport := gitGateImport(t, client, publicURL, "/v1/projects/"+project.ProjectID+"/artifacts/sources/source/git-import", httpsURL, "initial", 201)
			call("DELETE", "/v1/settings/git-key", nil, 204)
			gitGateImport(t, client, publicURL, "/v1/artifacts/sources/missing-key/git-import", sshURL, "initial", 422)
			call("PUT", "/v1/settings/git-key", map[string]string{"privateKey": string(privateKey)}, 200)
			wrongHost, _ := gitGateSigner(t)
			if err := os.WriteFile(known, []byte(knownhosts.Line([]string{sshAddress}, wrongHost.PublicKey())+"\n"), 0600); err != nil {
				t.Fatal(err)
			}
			gitGateImport(t, client, publicURL, "/v1/artifacts/sources/untrusted/git-import", sshURL, "initial", 422)
			if err := os.WriteFile(known, trusted, 0600); err != nil {
				t.Fatal(err)
			}
			gitGateImport(t, client, publicURL, "/v1/artifacts/sources/forbidden/git-import", "https://127.0.0.1:1/repo.git", "", 422)
			gitGateImport(t, client, publicURL, "/v1/artifacts/sources/missing-ref/git-import", httpsURL, "not-a-ref", 422)
			gitGateImport(t, client, publicURL, "/v1/artifacts/sources/https/git-import", httpsURL, "initial", 409)
			// Move the branch and remove the key; the Run consumes the retained exact
			// Project snapshot through normal allocation hydration and automatic export.
			if err := os.WriteFile(filepath.Join(repo, "source.txt"), []byte("branch moved\n"), 0600); err != nil {
				t.Fatal(err)
			}
			gitGateCommand(t, repo, "add", ".")
			gitGateCommand(t, repo, "commit", "-m", "advance")
			call("DELETE", "/v1/settings/git-key", nil, 204)
			python := filepath.Join(root, "runtime", ".venv", "bin", "python")
			runtimeAddress := freeAddress(t)
			runtimeProcess := startCodeAnalysisRuntime(t, "Git Python Runtime", root, python, privateURL, "https://"+runtimeAddress, runtimeAddress, filepath.Join(work, "runtime-work"), "memory", "", ca.Certificate, agent)
			waitForProcessLog(t, ctx, runtimeProcess, "runtime agent registered")
			runJSON := call("POST", "/v1/projects/"+project.ProjectID+"/runs", map[string]any{"workflow": "workspace-roundtrip@1", "parameters": map[string]string{}, "artifacts": map[string]any{"source": projectImport.Artifact}}, 202)
			var run runCreateResponse
			if err := json.Unmarshal(runJSON, &run); err != nil {
				t.Fatal(err)
			}
			status := waitForRunState(t, ctx, server, runtimeProcess, gateway, client, publicURL, run.RunID, "succeeded")
			if status.Outputs["workspace_diff"].Revision == nil {
				t.Fatal("Runtime did not export workspace diff")
			}
			diff, _ := download(t, client, publicURL+"/v1/runs/"+run.RunID+"/outputs/workspace_diff")
			if !bytes.Contains(diff, []byte("-before")) || !bytes.Contains(diff, []byte("+after")) {
				t.Fatal("Runtime did not edit the exact Git snapshot")
			}
			metadata := call("GET", "/v1/runs/"+run.RunID+"/artifacts/inputs/source/metadata", nil, 200)
			var input artifacts.Metadata
			if err := json.Unmarshal(metadata, &input); err != nil {
				t.Fatal(err)
			}
			if input.GitSource == nil || input.GitSource.ResolvedCommit != commit {
				t.Fatal("Run backtrace lost Git commit")
			}
			runtimeProcess.stop(t)
			// A 63 MiB expanded snapshot with incompressible files stays below both
			// consumer and ZIP caps. Hold three ordinary 64 MiB uploads concurrently.
			gitGateCommand(t, repo, "checkout", "-b", "large")
			for index := 0; index < 16; index++ {
				data := make([]byte, (4<<20)-(64<<10))
				if _, err := rand.Read(data); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(filepath.Join(repo, fmt.Sprintf("file-%02d.bin", index)), data, 0600); err != nil {
					t.Fatal(err)
				}
			}
			gitGateCommand(t, repo, "add", ".")
			gitGateCommand(t, repo, "commit", "-m", "large snapshot")
			entered, release := https.blockNext()
			defer release()
			done := make(chan gitimport.ImportResult, 1)
			go func() {
				defer close(done)
				done <- gitGateImport(t, client, publicURL, "/v1/artifacts/sources/large/git-import", httpsURL, "large", 201)
			}()
			select {
			case <-entered:
			case <-ctx.Done():
				t.Fatal(ctx.Err())
			}
			gitGateImport(t, client, publicURL, "/v1/artifacts/sources/saturated/git-import", httpsURL, "initial", 503)
			closeTransfers := gitGateHoldTransfers(t, publicAddress)
			defer closeTransfers()
			release()
			var large gitimport.ImportResult
			select {
			case large = <-done:
			case <-ctx.Done():
				t.Fatal(ctx.Err())
			}
			if large.Artifact.Revision == nil {
				t.Fatal("large Git import did not publish a revision")
			}
			largeBytes, _ := download(t, client, publicURL+"/v1/artifacts/sources/large?revision="+url.QueryEscape(*large.Artifact.Revision))
			if len(largeBytes) < 62<<20 || len(largeBytes) > 64<<20 {
				t.Fatalf("near-cap ZIP=%d", len(largeBytes))
			}
			validation := exec.CommandContext(ctx, python, "-c", "import io,sys,zipfile; from contractor_runtime.toolsets.source_analysis.tools import _validate_entries; z=zipfile.ZipFile(io.BytesIO(sys.stdin.buffer.read())); _validate_entries(z.infolist()); assert len(z.infolist()) == 17")
			validation.Dir = filepath.Join(root, "runtime")
			validation.Stdin = bytes.NewReader(largeBytes)
			if output, err := validation.CombinedOutput(); err != nil {
				t.Fatalf("Runtime source archive validation: %v: %s", err, output)
			}
			blobGatePeak(t, name, "Git "+backend)
			closeTransfers()
			if bytes.Contains(largeBytes, privateKey) || strings.Contains(server.logs.redacted(), string(privateKey)) || strings.Contains(runtimeProcess.logs.redacted(), string(privateKey)) {
				t.Fatal("Git key leaked into bytes or process logs")
			}
			t.Logf("%s: HTTPS/SSH, Project Run, exact backtrace, trust/CAS/capacity and %d-byte Git ZIP passed", backend, len(largeBytes))
		})
	}
}

func gitGateJSON(t *testing.T, client *http.Client, base, method, path string, body any, status int) []byte {
	t.Helper()
	var data []byte
	if body != nil {
		var err error
		data, err = json.Marshal(body)
		if err != nil {
			t.Fatal(err)
		}
	}
	request, err := http.NewRequest(method, base+path, bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Authorization", "Bearer "+publicToken)
	request.Header.Set("Content-Type", "application/json")
	if method == "POST" {
		request.Header.Set("Idempotency-Key", "git-gate-"+randomHex(t, 8))
		request.Header.Set("If-None-Match", "*")
	}
	response := do(t, client, request, status)
	defer response.Body.Close()
	result, err := io.ReadAll(io.LimitReader(response.Body, 64<<10))
	if err != nil {
		t.Fatal(err)
	}
	return result
}
func gitGateImport(t *testing.T, client *http.Client, base, path, repository, ref string, status int) gitimport.ImportResult {
	t.Helper()
	body := map[string]string{"repositoryUrl": repository}
	if ref != "" {
		body["ref"] = ref
	}
	data := gitGateJSON(t, client, base, "POST", path, body, status)
	var result gitimport.ImportResult
	if status == 200 || status == 201 {
		if err := json.Unmarshal(data, &result); err != nil || result.Artifact.Revision == nil {
			t.Fatalf("Git result: %v", err)
		}
	}
	return result
}
func gitGateHoldTransfers(t *testing.T, address string) func() {
	t.Helper()
	connections := make([]net.Conn, 0, 3)
	payload := bytes.Repeat([]byte{0x5a}, (64<<20)-1)
	for index := 0; index < 3; index++ {
		conn, err := net.DialTimeout("tcp", address, 5*time.Second)
		if err != nil {
			t.Fatal(err)
		}
		connections = append(connections, conn)
		_ = conn.SetDeadline(time.Now().Add(120 * time.Second))
		_, err = fmt.Fprintf(conn, "PUT /v1/artifacts/sources/held_%d HTTP/1.1\r\nHost: %s\r\nAuthorization: Bearer %s\r\nContent-Type: application/octet-stream\r\nIf-None-Match: *\r\nContent-Length: %d\r\nConnection: close\r\n\r\n", index, address, publicToken, 64<<20)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := io.Copy(conn, bytes.NewReader(payload)); err != nil {
			t.Fatal(err)
		}
	}
	return func() {
		for _, conn := range connections {
			_ = conn.Close()
		}
	}
}
