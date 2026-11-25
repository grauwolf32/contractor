//go:build e2e

package e2e

import (
	"context"
	"encoding/json"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/localpki"
	"github.com/jackc/pgx/v5/pgxpool"
)

// TestArtifactHistoryAcrossProductionProcesses runs the production Server
// with a retained-history limit of two revisions per binding, taken from its
// environment, and a Python Runtime. A Run's input fork holds the exact
// revision it copies, so ordinary CAS writes of the owner's binding remove
// the oldest unheld revision instead; a removed revision reads as not found;
// once only held revisions remain the write is the permanent
// artifact_history_protected; and after a restart with a larger limit the
// stored holds and counters admit the same write.
func TestArtifactHistoryAcrossProductionProcesses(t *testing.T) {
	if testing.Short() {
		t.Skip("end-to-end process test")
	}
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Fatal("CONTRACTOR_TEST_DATABASE_URL is required")
	}
	repositoryRoot := repoRoot(t)
	temporaryRoot := t.TempDir()
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()

	isolateURL := isolatedDatabase(t, ctx, databaseURL)
	serverBinary := sharedServerBinary(t)
	runChecked(t, repositoryRoot, map[string]string{"CONTRACTOR_DATABASE_URL": isolateURL}, serverBinary, "migrate")
	pkiRoot := filepath.Join(temporaryRoot, "pki")
	generator := localpki.Generator{}
	caPaths, err := generator.InitCA(pkiRoot, false)
	if err != nil {
		t.Fatal(err)
	}
	leaf := localpki.LeafOptions{IPAddresses: []net.IP{net.ParseIP("127.0.0.1")}}
	controlPlanePaths, err := generator.IssueControlPlane(pkiRoot, localpki.ControlPlaneOptions{
		LeafOptions: leaf, URI: "urn:contractor:control-plane:history",
	})
	if err != nil {
		t.Fatal(err)
	}
	agentPaths, err := generator.IssueAgent(pkiRoot, "history-agent", leaf)
	if err != nil {
		t.Fatal(err)
	}
	gateway := newFakeGateway(llmGatewayToken)
	t.Cleanup(gateway.close)
	configRoot := stageE2EConfiguration(t, filepath.Join(repositoryRoot, "configs", "e2e"),
		filepath.Join(temporaryRoot, "configs"), gateway.URL())
	publicAddress, privateAddress, runtimeAddress := freeAddress(t), freeAddress(t), freeAddress(t)
	publicBaseURL, privateBaseURL, runtimeBaseURL := "http://"+publicAddress, "https://"+privateAddress, "https://"+runtimeAddress
	localAuthFile := writeE2ELocalAuth(t, temporaryRoot, "history-user-"+randomHex(t, 8))
	startServer := func(maxRevisions string) *childProcess {
		return startProcess(t, "Go Server", repositoryRoot, map[string]string{
			"CONTRACTOR_DATABASE_URL":                   isolateURL,
			"CONTRACTOR_OPERATOR_CONFIG_ROOT":           configRoot,
			"CONTRACTOR_PUBLIC_LISTEN":                  publicAddress,
			"CONTRACTOR_PRIVATE_LISTEN":                 privateAddress,
			"CONTRACTOR_PRIVATE_URL":                    privateBaseURL,
			"CONTRACTOR_CA_FILE":                        caPaths.Certificate,
			"CONTRACTOR_CONTROL_PLANE_CERT_FILE":        controlPlanePaths.Certificate,
			"CONTRACTOR_CONTROL_PLANE_KEY_FILE":         controlPlanePaths.PrivateKey,
			"CONTRACTOR_LLM_GATEWAY_TOKEN":              llmGatewayToken,
			"CONTRACTOR_PUBLIC_BEARER_TOKEN":            publicToken,
			"CONTRACTOR_LOCAL_AUTH_FILE":                localAuthFile,
			"CONTRACTOR_BROWSER_ORIGINS":                "https://ui.contractor.invalid",
			"CONTRACTOR_ARTIFACT_HISTORY_MAX_REVISIONS": maxRevisions,
		}, serverBinary, "serve")
	}
	server := startServer("2")
	client := &http.Client{Timeout: 5 * time.Second}
	waitForHTTP(t, ctx, server, client, publicBaseURL+"/readyz", http.StatusOK)

	python := filepath.Join(repositoryRoot, "runtime", ".venv", "bin", "python")
	if info, statErr := os.Stat(python); statErr != nil || info.IsDir() {
		t.Fatalf("Python Runtime environment is missing at %s; run 'cd runtime && uv sync --locked'", python)
	}
	runtimeProcess := startProcess(t, "Python Runtime Agent", filepath.Join(repositoryRoot, "runtime"),
		map[string]string{"PYTHONUNBUFFERED": "1"},
		python, "-m", "contractor_runtime", "--allocation-slots", "1",
		"--control-plane-url", privateBaseURL,
		"--advertised-control-url", runtimeBaseURL,
		"--advertised-a2a-url", runtimeBaseURL,
		"--ca-file", caPaths.Certificate,
		"--certificate-file", agentPaths.Certificate,
		"--private-key-file", agentPaths.PrivateKey,
		"--listen", runtimeAddress,
		"--work-root", filepath.Join(temporaryRoot, "runtime-work"),
		"--request-timeout-seconds", "10",
		"--shutdown-grace-seconds", "5",
	)
	controlClient := newMTLSClient(t, caPaths.Certificate, controlPlanePaths)
	waitForHTTP(t, ctx, runtimeProcess, controlClient, runtimeBaseURL+"/healthz", http.StatusOK)
	waitForProcessLog(t, ctx, runtimeProcess, "runtime agent registered")

	source := publicBaseURL + "/v1/artifacts/projects/source"
	put := func(expected string, body string, status int) (string, historyErrorEnvelope, string) {
		t.Helper()
		request, err := http.NewRequest(http.MethodPut, source, strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		request.Header.Set("Authorization", "Bearer "+publicToken)
		request.Header.Set("Content-Type", e2eMediaType)
		request.Header.Set("If-Match", `"`+expected+`"`)
		response := do(t, client, request, status)
		defer response.Body.Close()
		if status != http.StatusOK {
			var envelope historyErrorEnvelope
			decodeResponse(t, response, &envelope)
			return "", envelope, response.Header.Get("Retry-After")
		}
		var written struct {
			Artifact  artifactRef `json:"artifact"`
			MediaType string      `json:"mediaType"`
			Size      int64       `json:"size"`
		}
		decodeResponse(t, response, &written)
		if written.Artifact.Revision == nil {
			t.Fatal("write returned no revision")
		}
		return *written.Artifact.Revision, historyErrorEnvelope{}, ""
	}
	readExact := func(revision string, status int) {
		t.Helper()
		request, err := http.NewRequest(http.MethodGet, source+"?revision="+url.QueryEscape(revision), nil)
		if err != nil {
			t.Fatal(err)
		}
		request.Header.Set("Authorization", "Bearer "+publicToken)
		do(t, client, request, status).Body.Close()
	}

	// The Runtime copies r1 through a Run whose input fork holds r1.
	uploaded := uploadInput(t, client, publicBaseURL)
	r1 := *uploaded.Revision
	firstRun := createRun(t, client, publicBaseURL, uploaded)
	if output, ok := waitForRun(t, ctx, server, runtimeProcess, gateway, client, publicBaseURL, firstRun).Outputs["result"]; !ok || output.Revision == nil {
		t.Fatalf("the first Run has no exact result output")
	}
	r2, _, _ := put(r1, "second", http.StatusOK)
	r3, _, _ := put(r2, "third", http.StatusOK) // removes r2: r1 is held
	readExact(r2, http.StatusNotFound)
	if data, _ := download(t, client, source+"?revision="+url.QueryEscape(r1)); string(data) != e2eInput {
		t.Fatalf("held r1 reads %q", data)
	}

	// A second Run holds r3, the current revision; no unheld revision is left
	// to remove, so the next write is refused permanently.
	gateway.ResetScenario()
	held := uploaded
	held.Revision = &r3
	secondRun := createWorkflowRun(t, client, publicBaseURL, "artifact-report-fixture@1", "e2e-history-second", held)
	waitForRun(t, ctx, server, runtimeProcess, gateway, client, publicBaseURL, secondRun)
	_, refused, retryAfter := put(r3, "fourth", http.StatusConflict)
	if refused.Code != "artifact_history_protected" || refused.Retryable || retryAfter != "" {
		t.Fatalf("write over held history = %+v, Retry-After %q", refused, retryAfter)
	}
	pool, err := pgxpool.New(ctx, isolateURL)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(pool.Close)
	assertSourceHistory(t, ctx, pool, []string{r1, r3}, 2)

	// The holds and counters are stored: a restarted Server with a larger
	// limit admits the same write and removes nothing.
	server.stop(t)
	server = startServer("3")
	waitForHTTP(t, ctx, server, client, publicBaseURL+"/readyz", http.StatusOK)
	r4, _, _ := put(r3, "fourth", http.StatusOK)
	assertSourceHistory(t, ctx, pool, []string{r1, r3, r4}, 3)
	if len(gateway.Failures()) != 0 {
		t.Fatalf("fake gateway failures: %v", gateway.Failures())
	}
}

type historyErrorEnvelope struct {
	Code      string          `json:"code"`
	Message   string          `json:"message"`
	Retryable bool            `json:"retryable"`
	RequestID string          `json:"requestId"`
	Details   json.RawMessage `json:"details"`
}

// assertSourceHistory compares the retained revisions of the owner's
// projects/source binding, oldest first, and its revision counter.
func assertSourceHistory(t *testing.T, ctx context.Context, pool *pgxpool.Pool, want []string, counted int64) {
	t.Helper()
	var listed string
	var retained int64
	if err := pool.QueryRow(ctx, `SELECT
    (SELECT json_agg(revision ORDER BY created_at, revision) FROM artifact_binding_revisions AS revision
      WHERE revision.scope_kind = binding.scope_kind AND revision.scope_id = binding.scope_id
        AND revision.namespace = binding.namespace AND revision.name = binding.name)::text,
    binding.retained_revisions
FROM artifact_bindings AS binding
WHERE binding.scope_kind = 'user' AND binding.namespace = 'projects' AND binding.name = 'source'`).
		Scan(&listed, &retained); err != nil {
		t.Fatal(err)
	}
	var revisions []string
	if err := json.Unmarshal([]byte(listed), &revisions); err != nil {
		t.Fatal(err)
	}
	if strings.Join(revisions, ",") != strings.Join(want, ",") || retained != counted {
		t.Fatalf("source retains %v counted as %d, want %v and %d", revisions, retained, want, counted)
	}
}
