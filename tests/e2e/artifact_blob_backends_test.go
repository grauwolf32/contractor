//go:build e2e

package e2e

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/localpki"
)

// This gate deliberately runs the production binary in a scratch container:
// no implicit writable /tmp, host blob directory or PVC can hide local writes.
func TestArtifactBlobBackendsContainers(t *testing.T) {
	if testing.Short() {
		t.Skip("container release gate")
	}
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Fatal("CONTRACTOR_TEST_DATABASE_URL is required")
	}
	if _, err := exec.LookPath("podman"); err != nil {
		t.Fatal("podman is required")
	}
	repositoryRoot, buildRoot := repoRoot(t), t.TempDir()
	binary := filepath.Join(buildRoot, "contractor-server")
	runChecked(t, repositoryRoot, map[string]string{"CGO_ENABLED": "0"}, "go", "build", "-o", binary, "./cmd/contractor-server")
	if err := os.WriteFile(filepath.Join(buildRoot, "Containerfile"), []byte("FROM scratch\nCOPY contractor-server /contractor-server\nENTRYPOINT [\"/contractor-server\"]\n"), 0600); err != nil {
		t.Fatal(err)
	}
	image := "localhost/contractor-blob-gate:" + randomHex(t, 8)
	runChecked(t, buildRoot, nil, "podman", "build", "-q", "-t", image, buildRoot)
	t.Cleanup(func() { _ = exec.Command("podman", "rmi", "-f", image).Run() })
	for _, backend := range []string{"postgresql", "filesystem"} {
		t.Run(backend, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
			defer cancel()
			root := t.TempDir()
			database := isolatedDatabase(t, ctx, databaseURL)
			runChecked(t, repositoryRoot, map[string]string{"CONTRACTOR_DATABASE_URL": database}, binary, "migrate")
			generator := localpki.Generator{}
			ca, err := generator.InitCA(filepath.Join(root, "pki"), false)
			if err != nil {
				t.Fatal(err)
			}
			leaf := localpki.LeafOptions{IPAddresses: []net.IP{net.ParseIP("127.0.0.1")}}
			control, err := generator.IssueControlPlane(filepath.Join(root, "pki"), localpki.ControlPlaneOptions{LeafOptions: leaf, URI: "urn:contractor:control-plane:blob-gate"})
			if err != nil {
				t.Fatal(err)
			}
			agent, err := generator.IssueAgent(filepath.Join(root, "pki"), "blob-gate", leaf)
			if err != nil {
				t.Fatal(err)
			}
			gateway := newFakeGateway(llmGatewayToken)
			t.Cleanup(gateway.close)
			configs := stageE2EConfiguration(t, filepath.Join(repositoryRoot, "configs", "e2e"), filepath.Join(root, "configs"), gateway.URL())
			publicAddress, privateAddress := freeAddress(t), freeAddress(t)
			publicURL, privateURL := "http://"+publicAddress, "https://"+privateAddress
			userID := "blob-gate-" + randomHex(t, 8)
			environment := map[string]string{
				"CONTRACTOR_DATABASE_URL": database, "CONTRACTOR_OPERATOR_CONFIG_ROOT": configs,
				"CONTRACTOR_MANAGED_CONFIG_ROOT": "/managed", "CONTRACTOR_PUBLIC_LISTEN": publicAddress,
				"CONTRACTOR_PRIVATE_LISTEN": privateAddress, "CONTRACTOR_PRIVATE_URL": privateURL,
				"CONTRACTOR_CA_FILE": ca.Certificate, "CONTRACTOR_CONTROL_PLANE_CERT_FILE": control.Certificate,
				"CONTRACTOR_CONTROL_PLANE_KEY_FILE": control.PrivateKey, "CONTRACTOR_LLM_GATEWAY_TOKEN": llmGatewayToken,
				"CONTRACTOR_PUBLIC_BEARER_TOKEN": publicToken, "CONTRACTOR_LOCAL_AUTH_FILE": writeE2ELocalAuth(t, root, userID),
				"CONTRACTOR_BROWSER_ORIGINS": "https://ui.contractor.invalid", "CONTRACTOR_ARTIFACT_BLOB_BACKEND": backend,
			}
			if backend == "filesystem" {
				environment["CONTRACTOR_ARTIFACT_BLOB_PATH"] = "/blobs"
			}
			client := &http.Client{Timeout: 45 * time.Second}
			var containerName string
			var server *childProcess
			start := func() {
				containerName = "contractor-blob-gate-" + randomHex(t, 8)
				args := []string{"run", "--name", containerName, "--network=host", "--read-only", "--read-only-tmpfs=false", "--cap-drop=all", "--security-opt=no-new-privileges", "--tmpfs=/managed:rw,nosuid,nodev,noexec,size=64m", "-v", root + ":" + root + ":ro"}
				if backend == "filesystem" {
					args = append(args, "--tmpfs=/blobs:rw,nosuid,nodev,noexec,size=1g")
				}
				for key, value := range environment {
					args = append(args, "--env", key+"="+value)
				}
				args = append(args, image, "serve")
				name := containerName
				// Register after startProcess so container shutdown precedes process cleanup.
				server = startProcess(t, "blob Server "+backend, repositoryRoot, nil, "podman", args...)
				t.Cleanup(func() { _ = exec.Command("podman", "rm", "-f", name).Run() })
				waitForHTTP(t, ctx, server, client, publicURL+"/readyz", http.StatusOK)
			}
			start()
			// Same authenticated public and allocation-scoped mTLS APIs as normal runs.
			python := filepath.Join(repositoryRoot, "runtime", ".venv", "bin", "python")
			runtimeAddress := freeAddress(t)
			runtimeProcess := startRuntimeWithAdapters(t, "blob Python Runtime", repositoryRoot, python, privateURL, runtimeAddress, filepath.Join(root, "runtime-work"), ca.Certificate, agent, nil)
			waitForProcessLog(t, ctx, runtimeProcess, "runtime agent registered")
			input := uploadInput(t, client, publicURL)
			runID := createRun(t, client, publicURL, input)
			completed := waitForRun(t, ctx, server, runtimeProcess, gateway, client, publicURL, runID)
			if completed.Outputs["result"].Revision == nil {
				t.Fatal("private Artifact copy did not publish exact output")
			}
			output, _ := download(t, client, publicURL+"/v1/runs/"+url.PathEscape(runID)+"/outputs/result")
			if string(output) != e2eInput {
				t.Fatal("public/private Artifact bytes disagree")
			}
			runtimeProcess.stop(t)

			payload := bytes.Repeat([]byte{0x5a}, 64<<20)
			ref := blobGateUpload(t, client, publicURL, "maximum", payload, http.StatusCreated)
			exact := publicURL + "/v1/artifacts/projects/maximum?revision=" + url.QueryEscape(*ref.Revision)
			blobGateVerify(t, client, exact, payload)
			// Content-Length alone must reject oversize before allocating/reading bytes.
			oversized, _ := http.NewRequest(http.MethodPut, publicURL+"/v1/artifacts/projects/oversized", bytes.NewReader([]byte{1}))
			oversized.ContentLength = (64 << 20) + 1
			oversized.Header.Set("Expect", "100-continue")
			oversized.Header.Set("Authorization", "Bearer "+publicToken)
			oversized.Header.Set("Content-Type", "application/octet-stream")
			oversized.Header.Set("If-None-Match", "*")
			response := do(t, client, oversized, http.StatusRequestEntityTooLarge)
			response.Body.Close()
			blobGateSaturation(t, client, publicAddress, publicURL, runID, payload)
			blobGatePeak(t, containerName, backend)
			runChecked(t, repositoryRoot, nil, "podman", "stop", "--time=10", containerName)
			start() // New container and fresh tmpfs, retained PostgreSQL registry.
			if backend == "postgresql" {
				blobGateVerify(t, client, exact, payload)
			} else {
				request, _ := http.NewRequest(http.MethodGet, exact, nil)
				request.Header.Set("Authorization", "Bearer "+publicToken)
				response := do(t, client, request, http.StatusServiceUnavailable)
				body := copyBounded(response.Body)
				response.Body.Close()
				if !strings.Contains(body, "artifact_content_missing") {
					t.Fatalf("lost emptyDir response: %s", body)
				}
				request, _ = http.NewRequest(http.MethodGet, publicURL+"/v1/artifacts/projects/maximum/metadata", nil)
				request.Header.Set("Authorization", "Bearer "+publicToken)
				response = do(t, client, request, http.StatusOK)
				response.Body.Close()
				blobGateRunMetadata(t, client, publicURL, runID)
			}
		})
	}
}

func blobGateUpload(t *testing.T, client *http.Client, base, name string, data []byte, status int) artifactRef {
	t.Helper()
	request, _ := http.NewRequest(http.MethodPut, base+"/v1/artifacts/projects/"+name, bytes.NewReader(data))
	request.Header.Set("Authorization", "Bearer "+publicToken)
	request.Header.Set("Content-Type", "application/octet-stream")
	request.Header.Set("If-None-Match", "*")
	response := do(t, client, request, status)
	defer response.Body.Close()
	if status != http.StatusCreated {
		return artifactRef{}
	}
	var result struct {
		Artifact artifactRef `json:"artifact"`
	}
	if err := json.NewDecoder(response.Body).Decode(&result); err != nil {
		t.Fatal(err)
	}
	if result.Artifact.Revision == nil {
		t.Fatal("upload returned no exact revision")
	}
	return result.Artifact
}

func blobGateVerify(t *testing.T, client *http.Client, target string, want []byte) {
	t.Helper()
	request, _ := http.NewRequest(http.MethodGet, target, nil)
	request.Header.Set("Authorization", "Bearer "+publicToken)
	response := do(t, client, request, http.StatusOK)
	defer response.Body.Close()
	hash := sha256.New()
	size, err := io.Copy(hash, response.Body)
	digest := sha256.Sum256(want)
	if err != nil || size != int64(len(want)) || !bytes.Equal(hash.Sum(nil), digest[:]) {
		t.Fatalf("download integrity: size=%d err=%v", size, err)
	}
}

func blobGateSaturation(t *testing.T, client *http.Client, address, base, runID string, payload []byte) {
	t.Helper()
	connections := make([]net.Conn, 4)
	for index := range connections {
		conn, err := net.DialTimeout("tcp", address, 5*time.Second)
		if err != nil {
			t.Fatal(err)
		}
		connections[index] = conn
		defer conn.Close()
		if err := conn.SetDeadline(time.Now().Add(60 * time.Second)); err != nil {
			t.Fatal(err)
		}
		_, err = fmt.Fprintf(conn, "PUT /v1/artifacts/projects/concurrent_%d HTTP/1.1\r\nHost: %s\r\nAuthorization: Bearer %s\r\nContent-Type: application/octet-stream\r\nIf-None-Match: *\r\nContent-Length: %d\r\nConnection: close\r\n\r\n", index, address, publicToken, len(payload))
		if err != nil {
			t.Fatal(err)
		}
		if _, err = io.Copy(conn, bytes.NewReader(payload[:len(payload)-1])); err != nil {
			t.Fatal(err)
		}
	}
	blobGateUpload(t, client, base, "over_capacity", []byte("small"), http.StatusServiceUnavailable)
	// Metadata remains usable under saturation; output downloads share the budget.
	blobGateRunMetadata(t, client, base, runID)
	request, _ := http.NewRequest(http.MethodGet, base+"/v1/runs/"+url.PathEscape(runID)+"/outputs/result", nil)
	request.Header.Set("Authorization", "Bearer "+publicToken)
	response := do(t, client, request, http.StatusServiceUnavailable)
	body := copyBounded(response.Body)
	response.Body.Close()
	if !strings.Contains(body, "artifact_transfer_capacity") {
		t.Fatalf("output saturation: %s", body)
	}
	// Cancellation releases capacity even while the other three full bodies wait.
	connections[0].Close()
	deadline := time.Now().Add(10 * time.Second)
	for {
		request, _ := http.NewRequest(http.MethodPut, base+"/v1/artifacts/projects/after_cancel", bytes.NewReader([]byte("small")))
		request.Header.Set("Authorization", "Bearer "+publicToken)
		request.Header.Set("Content-Type", "text/plain")
		request.Header.Set("If-None-Match", "*")
		response, err := client.Do(request)
		if err != nil {
			t.Fatal(err)
		}
		status := response.StatusCode
		response.Body.Close()
		if status == http.StatusCreated {
			break
		}
		if status != http.StatusServiceUnavailable || time.Now().After(deadline) {
			t.Fatalf("capacity not released after cancellation: %d", status)
		}
		time.Sleep(20 * time.Millisecond)
	}
	for _, conn := range connections[1:] {
		if _, err := conn.Write(payload[len(payload)-1:]); err != nil {
			t.Fatal(err)
		}
	}
	for index, conn := range connections[1:] {
		response, err := http.ReadResponse(bufio.NewReader(conn), nil)
		if err != nil {
			t.Fatal(err)
		}
		body := copyBounded(response.Body)
		response.Body.Close()
		if response.StatusCode != http.StatusCreated {
			t.Fatalf("concurrent upload %d: %d %s", index, response.StatusCode, body)
		}
	}
}

func blobGatePeak(t *testing.T, name, backend string) {
	t.Helper()
	output, err := exec.Command("podman", "inspect", "--format", "{{.State.Pid}}", name).Output()
	if err != nil {
		t.Fatal(err)
	}
	pid, err := strconv.Atoi(strings.TrimSpace(string(output)))
	if err != nil {
		t.Fatal(err)
	}
	status, err := os.ReadFile(fmt.Sprintf("/proc/%d/status", pid))
	if err != nil {
		t.Fatal(err)
	}
	for _, line := range strings.Split(string(status), "\n") {
		if strings.HasPrefix(line, "VmHWM:") {
			t.Logf("%s Server process peak RSS: %s", backend, line)
			return
		}
	}
	t.Fatal("kernel did not report process peak RSS")
}

func blobGateRunMetadata(t *testing.T, client *http.Client, base, runID string) {
	t.Helper()
	request, _ := http.NewRequest(http.MethodGet, base+"/v1/runs/"+url.PathEscape(runID), nil)
	request.Header.Set("Authorization", "Bearer "+publicToken)
	response := do(t, client, request, http.StatusOK)
	defer response.Body.Close()
	var run runStatus
	decodeResponse(t, response, &run)
	if run.Outputs["result"].Revision == nil {
		t.Fatal("Run metadata lost exact output reference")
	}
}
