//go:build e2e

package e2e

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"io/fs"
	"net"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/localpki"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestToolsetRuntimeConfigurationAcrossProcesses(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Fatal("CONTRACTOR_TEST_DATABASE_URL is required")
	}
	root, temp := repoRoot(t), t.TempDir()
	ctx, cancel := context.WithTimeout(t.Context(), 12*time.Minute)
	defer cancel()
	database := isolatedDatabase(t, ctx, databaseURL)
	binary := filepath.Join(temp, "contractor-server")
	runChecked(t, root, nil, "go", "build", "-o", binary, "./cmd/contractor-server")
	runChecked(t, root, map[string]string{"CONTRACTOR_DATABASE_URL": database}, binary, "migrate")
	gateway := newMCPScriptedGateway(t)
	first, second := newMCPReleaseFixture(t, true), newMCPReleaseFixture(t, false)
	configRoot := stageE2EConfiguration(t, filepath.Join(root, "configs/e2e"), filepath.Join(temp, "configs"), gateway.server.URL+"/v1")
	fixtureRoot := filepath.Join(root, "tests/e2e/testdata/toolsets")
	err := filepath.WalkDir(fixtureRoot, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		relative, _ := filepath.Rel(fixtureRoot, path)
		target := filepath.Join(configRoot, relative)
		if entry.IsDir() {
			return os.MkdirAll(target, 0700)
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		return os.WriteFile(target, data, 0600)
	})
	if err != nil {
		t.Fatal(err)
	}
	pki := filepath.Join(temp, "pki")
	gen := localpki.Generator{}
	ca, err := gen.InitCA(pki, false)
	if err != nil {
		t.Fatal(err)
	}
	leaf := localpki.LeafOptions{IPAddresses: []net.IP{net.ParseIP("127.0.0.1")}}
	control, err := gen.IssueControlPlane(pki, localpki.ControlPlaneOptions{LeafOptions: leaf, URI: "urn:contractor:control-plane:toolsets"})
	if err != nil {
		t.Fatal(err)
	}
	identity, err := gen.IssueAgent(pki, "toolset-release-runtime", leaf)
	if err != nil {
		t.Fatal(err)
	}
	publicAddr, privateAddr, uiAddr := freeAddress(t), freeAddress(t), freeAddress(t)
	publicURL, privateURL := "http://"+publicAddr, "https://"+privateAddr
	target, _ := url.Parse(publicURL)
	apiProxy := httputil.NewSingleHostReverseProxy(target)
	uiTarget, _ := url.Parse("http://" + uiAddr)
	uiProxy := httputil.NewSingleHostReverseProxy(uiTarget)
	browserProxy := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/v1/") {
			apiProxy.ServeHTTP(w, r)
		} else {
			uiProxy.ServeHTTP(w, r)
		}
	}))
	t.Cleanup(browserProxy.Close)
	master := filepath.Join(temp, "master-key")
	if err := os.WriteFile(master, []byte(base64.StdEncoding.EncodeToString(bytes.Repeat([]byte{0x6b}, 32))), 0600); err != nil {
		t.Fatal(err)
	}
	serverEnv := map[string]string{"CONTRACTOR_DATABASE_URL": database, "CONTRACTOR_CONFIG_ROOT": configRoot, "CONTRACTOR_PUBLIC_LISTEN": publicAddr, "CONTRACTOR_PRIVATE_LISTEN": privateAddr, "CONTRACTOR_PRIVATE_URL": privateURL, "CONTRACTOR_CA_FILE": ca.Certificate, "CONTRACTOR_CONTROL_PLANE_CERT_FILE": control.Certificate, "CONTRACTOR_CONTROL_PLANE_KEY_FILE": control.PrivateKey, "CONTRACTOR_LLM_GATEWAY_TOKEN": llmGatewayToken, "CONTRACTOR_PUBLIC_USER_ID": "toolsets-owner", "CONTRACTOR_PUBLIC_BEARER_TOKEN": publicToken, "CONTRACTOR_LOCAL_AUTH_FILE": writeE2ELocalAuth(t, temp, "toolsets-owner"), "CONTRACTOR_BROWSER_ORIGINS": browserProxy.URL}
	startServer := func() *childProcess {
		return startProcess(t, "toolset Server", root, serverEnv, binary, "serve", "--credential-master-key-file", master)
	}
	server := startServer()
	client := &http.Client{Timeout: 10 * time.Second}
	waitForHTTP(t, ctx, server, client, publicURL+"/readyz", 200)
	runtimeAddr, work := freeAddress(t), filepath.Join(temp, "runtime-work")
	runtimeProcess := startRuntimeWithAdapters(t, "toolset Runtime", root, filepath.Join(root, "runtime/.venv/bin/python"), privateURL, runtimeAddr, work, ca.Certificate, identity, nil)
	controlClient := newMTLSClient(t, ca.Certificate, control)
	waitForHTTP(t, ctx, runtimeProcess, controlClient, "https://"+runtimeAddr+"/healthz", 200)
	waitForProcessLog(t, ctx, runtimeProcess, "runtime agent registered")
	pool, err := pgxpool.New(ctx, database)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(pool.Close)
	store := runstore.NewPostgresStore(pool)
	ops := &runtimeOperations{t: t, client: client, baseURL: publicURL}
	createBody := map[string]any{"workflow": "toolset-shared@1", "runtimeLabels": []string{"knowledge"}}
	create := func(key string, body any) createdLabeledRun {
		raw := ops.request("POST", "/v1/runs", body, 202, map[string]string{"Idempotency-Key": key})
		var run createdLabeledRun
		if json.Unmarshal(raw, &run) != nil || run.RunID == "" {
			t.Fatal("invalid Run create")
		}
		return run
	}
	var original createdLabeledRun
	var oldRef, newRef runtimeconfig.Ref
	t.Run("shared_sessions_late_binding_replay_native_and_restart", func(t *testing.T) {
		ops.t = t
		ops.request("POST", "/v1/runs", map[string]any{"workflow": "toolset-shared@1"}, 400, map[string]string{"Idempotency-Key": "missing-toolset"})
		ops.createRuntimeCredential("mcp-release", "mcp-headers@1", map[string]any{"headers": map[string]string{"Authorization": "Bearer " + mcpReleaseSecret}})
		spec := func(endpoint string) any {
			return map[string]any{"worker": map[string]any{"llmGateway": map[string]any{"gateway": "local-litellm@1"}, "toolsets": map[string]any{"mcp@1": map[string]any{"endpoint": endpoint, "credential": "mcp-release", "requestTimeoutSeconds": 30}}}}
		}
		oldRef = ops.publishRuntimeConfig("mcp-release", "1", spec(first.endpoint()))
		newRef = ops.publishRuntimeConfig("mcp-release", "2", spec(second.endpoint()))
		ops.createRuntimeLabel("knowledge", oldRef)
		ops.createRuntimeLabel("same-knowledge", oldRef)
		ops.createRuntimeLabel("different-knowledge", newRef)
		ops.request("POST", "/v1/runs", map[string]any{"workflow": "toolset-shared@1", "runtimeLabels": []string{"knowledge", "different-knowledge"}}, 409, map[string]string{"Idempotency-Key": "conflicting-toolset"})
		original = create("shared-toolsets", createBody)
		select {
		case <-first.firstCall:
		case <-time.After(30 * time.Second):
			t.Fatalf("MCP call did not start\n%s\n%s", server.logs.redacted(mcpReleaseSecret, llmGatewayToken), runtimeProcess.logs.redacted(mcpReleaseSecret, llmGatewayToken))
		}
		ops.rebindRuntimeLabel("knowledge", "1", newRef)
		ops.rebindRuntimeLabel("default", "1", newRef)
		replay := ops.requestResponse("POST", "/v1/runs", createBody, 202, map[string]string{"Idempotency-Key": "shared-toolsets"})
		ops.requireReplay(replay, "pinned MCP Run")
		var replayed createdLabeledRun
		_ = json.Unmarshal(replay.body, &replayed)
		if replayed.RunID != original.RunID || !samePinnedRuntimeConfiguration(original, replayed) {
			t.Fatal("Run replay changed pins")
		}
		first.release.Do(func() { close(first.unblock) })
		status := waitForRunTerminalAcross(t, ctx, server, []*childProcess{runtimeProcess}, client, publicURL, original.RunID)
		if status.State != "succeeded" {
			t.Fatalf("shared Run failed: %+v\n%s", status, runtimeProcess.logs.redacted(mcpReleaseSecret, llmGatewayToken))
		}
		assertToolsetReleaseAllocations(t, ctx, store, original.RunID, oldRef)
		waitForRuntimeIdle(t, ctx, runtimeProcess, controlClient, "https://"+runtimeAddr, work)
		sessions, calls, deleted, _ := first.counts()
		if sessions != 2 || calls != 2 || deleted != 2 {
			t.Fatalf("MCP sessions/calls/deletes: %d/%d/%d", sessions, calls, deleted)
		}
		if sessions, _, _, _ := second.counts(); sessions != 0 {
			t.Fatal("late stage used changed binding")
		}
		first.mu.Lock()
		for _, names := range first.calls {
			if len(names) != 1 || (names[0] != "search" && names[0] != "fetch") {
				t.Error("per-allocation call selection differs")
			}
		}
		first.mu.Unlock()
		server.stop(t)
		server = startServer()
		waitForHTTP(t, ctx, server, client, publicURL+"/readyz", 200)
		replay = ops.requestResponse("POST", "/v1/runs", createBody, 202, map[string]string{"Idempotency-Key": "shared-toolsets"})
		ops.requireReplay(replay, "restart replay")
		var restored createdLabeledRun
		_ = json.Unmarshal(replay.body, &restored)
		if !samePinnedRuntimeConfiguration(original, restored) {
			t.Fatal("restart lost original pins")
		}
		repeated := create("new-current-toolsets", createBody)
		if repeated.RuntimeConfiguration.Labels[0].Config != newRef {
			t.Fatal("new Run did not pin current binding")
		}
		if status := waitForRunTerminalAcross(t, ctx, server, []*childProcess{runtimeProcess}, client, publicURL, repeated.RunID); status.State != "succeeded" {
			t.Fatalf("new Run failed: %+v", status)
		}
		assertToolsetReleaseAllocations(t, ctx, store, repeated.RunID, newRef)
	})
	t.Run("protocol_faults_are_bounded_and_calls_are_not_replayed", func(t *testing.T) {
		for _, mode := range []string{"missing", "schema", "unauthorized", "drop", "content"} {
			t.Run(mode, func(t *testing.T) {
				ops.t = t
				ref := ops.publishRuntimeConfig("mcp-fault-"+mode, "1", map[string]any{"worker": map[string]any{"toolsets": map[string]any{"mcp@1": map[string]any{"endpoint": second.endpoint() + "/" + mode, "credential": "mcp-release", "connectTimeoutSeconds": 2, "requestTimeoutSeconds": 3}}}})
				ops.createRuntimeLabel("fault-"+mode, ref)
				_, before, _, _ := second.counts()
				run := create("fault-run-"+mode, map[string]any{"workflow": "toolset-shared@1", "runtimeLabels": []string{"fault-" + mode}})
				status := waitForRunTerminalAcross(t, ctx, server, []*childProcess{runtimeProcess}, client, publicURL, run.RunID)
				expected := "failed"
				expectedCalls := 0
				if mode == "drop" || mode == "content" {
					expected = "succeeded"
					expectedCalls = 2
				}
				if status.State != expected {
					t.Fatalf("fault %s state=%s want %s\n%s", mode, status.State, expected, runtimeProcess.logs.redacted(mcpReleaseSecret, llmGatewayToken))
				}
				_, after, _, _ := second.counts()
				if after-before != expectedCalls {
					t.Fatalf("remote calls=%d want %d", after-before, expectedCalls)
				}
				waitForRuntimeIdle(t, ctx, runtimeProcess, controlClient, "https://"+runtimeAddr, work)
				if mode == "drop" || mode == "content" {
					stages, err := store.ListStageExecutions(ctx, run.RunID)
					if err != nil {
						t.Fatal(err)
					}
					failures := int64(0)
					for _, stage := range stages {
						reports, err := store.ListStageExecutionReports(ctx, stage.StageExecutionID)
						if err != nil || len(reports) != 1 {
							t.Fatal("fault report missing")
						}
						for name, metric := range reports[0].Report.Worker.Metrics.Tools {
							if (name == "search" || name == "fetch") && metric.Failed != nil {
								failures += *metric.Failed
							}
						}
					}
					if failures != 2 {
						t.Fatalf("tool failure accounting=%d, want 2", failures)
					}
				}
				if mode == "missing" {
					if status.ResumeStageExecutionID == nil {
						t.Fatal("failed preparation has no safe resume boundary")
					}
					ops.rebindRuntimeLabel("fault-missing", "1", newRef)
					resumed := ops.request("POST", "/v1/runs/"+run.RunID+"/resume", map[string]string{"stageExecutionId": *status.ResumeStageExecutionID}, 202, nil)
					if !bytes.Equal(resumed, ops.request("POST", "/v1/runs/"+run.RunID+"/resume", map[string]string{"stageExecutionId": *status.ResumeStageExecutionID}, 202, nil)) {
						t.Fatal("resume replay duplicated replacement")
					}
					again := waitForRunTerminalAcross(t, ctx, server, []*childProcess{runtimeProcess}, client, publicURL, run.RunID)
					if again.State != "failed" {
						t.Fatal("replacement unexpectedly used current repaired binding")
					}
					stages, err := store.ListStageExecutions(ctx, run.RunID)
					if err != nil || len(stages) != 2 {
						t.Fatal("replacement attempt not persisted")
					}
					ids := map[string]bool{}
					for _, stage := range stages {
						allocations, err := store.ListStageAllocations(ctx, stage.StageExecutionID)
						if err != nil || len(allocations) != 1 {
							t.Fatal("replacement allocation missing")
						}
						value := allocations[0]
						ids[value.AllocationID] = true
						if value.RuntimeConfiguration.Origins.Toolsets["mcp@1"].Configs[0] != ref {
							t.Fatal("replacement changed pinned configuration")
						}
					}
					if len(ids) != 2 {
						t.Fatal("replacement reused allocation identity")
					}
				}
			})
		}
	})
	t.Run("browser_production_connection_journey", func(t *testing.T) {
		runToolsetBrowser(t, root, temp, uiAddr, browserProxy.URL, second.endpoint(), original.RunID)
	})
	t.Run("redaction_and_confirmed_cleanup", func(t *testing.T) {
		waitForRuntimeIdle(t, ctx, runtimeProcess, controlClient, "https://"+runtimeAddr, work)
		assertRuntimeLabelSecretsAbsent(t, ctx, pool, ops.evidence, []*childProcess{server, runtimeProcess}, []string{mcpReleaseSecret, "private-mcp-session-"})
		var leaked bool
		if err := pool.QueryRow(ctx, `SELECT EXISTS (SELECT 1 FROM allocation_execution_reports WHERE position($1 in report::text)>0 OR position('private-mcp-session-' in report::text)>0)`, mcpReleaseSecret).Scan(&leaked); err != nil || leaked {
			t.Fatal("private material reached durable allocation reports")
		}
		cleanupDeadline := time.Now().Add(5 * time.Second)
		for {
			_, _, _, left := first.counts()
			_, _, _, right := second.counts()
			if left+right == 0 {
				break
			}
			if time.Now().After(cleanupDeadline) {
				t.Fatalf("MCP sockets retained after confirmed release: %d", left+right)
			}
			time.Sleep(20 * time.Millisecond)
		}
		gateway.mu.Lock()
		defer gateway.mu.Unlock()
		if len(gateway.errors) > 0 {
			t.Fatal(gateway.errors)
		}
		for _, body := range gateway.bodies {
			if bytes.Contains(body, []byte(mcpReleaseSecret)) || bytes.Contains(body, []byte("private-mcp-session-")) {
				t.Fatal("model gateway received private material")
			}
		}
	})
}

func assertToolsetReleaseAllocations(t *testing.T, ctx context.Context, store *runstore.PostgresStore, runID string, ref runtimeconfig.Ref) {
	t.Helper()
	stages, err := store.ListStageExecutions(ctx, runID)
	if err != nil || len(stages) != 3 {
		t.Fatalf("stages: %d, %v", len(stages), err)
	}
	for _, stage := range stages {
		var allocations []runstore.StageAllocation
		for {
			allocations, err = store.ListStageAllocations(ctx, stage.StageExecutionID)
			if err != nil {
				t.Fatal(err)
			}
			if len(allocations) == 1 && allocations[0].ReleaseCompletedAt != nil {
				break
			}
			select {
			case <-ctx.Done():
				t.Fatal("allocation release not confirmed")
			case <-time.After(50 * time.Millisecond):
			}
		}
		value := allocations[0].RuntimeConfiguration
		if allocations[0].LogicalAgentName == "native" {
			if len(value.Origins.Toolsets) != 0 || len(value.Provenance.RuntimeCredentialRefs) != 0 {
				t.Fatal("native allocation retained MCP configuration or credentials")
			}
		} else {
			origin := value.Origins.Toolsets["mcp@1"]
			if len(origin.Configs) != 1 || origin.Configs[0] != ref || len(value.Provenance.RuntimeCredentialRefs) != 1 {
				t.Fatalf("MCP allocation origin mismatch: %+v", value)
			}
		}
		reports, err := store.ListStageExecutionReports(ctx, stage.StageExecutionID)
		if err != nil || len(reports) != 1 {
			t.Fatalf("durable allocation reports=%d: %v", len(reports), err)
		}
	}
}
