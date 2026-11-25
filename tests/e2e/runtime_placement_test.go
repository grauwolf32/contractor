//go:build e2e

package e2e

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"maps"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/localpki"
	"github.com/grauwolf32/contractor/internal/mtls"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Real Server, Python Runtime, mTLS, Scheduler and PostgreSQL paths. Only the
// external scanner is a deterministic fixture; scanner fidelity has its own gate.
func TestRuntimePlacementAcrossProductionProcesses(t *testing.T) {
	if testing.Short() {
		t.Fatal("the Runtime placement gate cannot run in short mode")
	}
	if os.Getenv("CONTRACTOR_TEST_DATABASE_URL") == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is unset")
	}
	h := startPlacementStack(t)
	h.startRuntime("placement-local", "local/lab", true)
	h.startRuntime("placement-cloud-prefix", "cloud/aws-test", true)
	h.startRuntime("placement-cloud-no-tool", "cloud/aws", false)
	step := func(name string, test func(*testing.T)) {
		if !t.Run(name, test) {
			t.FailNow()
		}
	}

	step("parent_and_local_matching", func(t *testing.T) {
		h.complete(t, h.create(t, map[string]string{"cloud": ""}, ""), "cloud/aws-test")
		h.complete(t, h.create(t, map[string]string{"local": "lab"}, ""), "local/lab")
	})

	var pinned string
	var oldAllocation runstore.StageAllocation
	step("capacity_wait_cancellation_and_late_connection", func(t *testing.T) {
		pinned = h.create(t, map[string]string{"cloud": "aws", "team": "appsec"}, "")
		waiting := waitForUnallocatedPreparingExecution(t, h.ctx, h.store, pinned)
		cancelled := h.create(t, map[string]string{"cloud": "aws"}, "")
		waitForUnallocatedPreparingExecution(t, h.ctx, h.store, cancelled)
		cancelWorkflowRun(t, h.operations.client, h.operations.baseURL, cancelled)
		h.terminal(t, cancelled, "cancelled")
		assertSameUnallocatedPreparingExecution(t, h.ctx, h.store, waiting)
		h.startRuntime("placement-cloud", "cloud/aws/eu-central", true)
		oldAllocation = h.complete(t, pinned, "cloud/aws/eu-central")
		if oldAllocation.RuntimeAgentID != h.principalID(t, "placement-cloud") {
			t.Fatalf("wrong principal received provider-specific allocation: %+v", oldAllocation)
		}
	})

	step("restart_changes_live_location_but_retains_history", func(t *testing.T) {
		h.stopRuntime("placement-cloud")
		h.startRuntime("placement-cloud", "cloud/azure", true)
		h.complete(t, h.create(t, map[string]string{"cloud": "azure"}, ""), "cloud/azure")
		historical := onlyRunAllocation(t, h.ctx, h.store, pinned)
		if historical.RuntimeAgentInstanceID != oldAllocation.RuntimeAgentInstanceID ||
			!slices.Equal(historical.RuntimePlacementLabels, oldAllocation.RuntimePlacementLabels) {
			t.Fatal("restarted incarnation reinterpreted historical allocation")
		}
		blocked := h.create(t, map[string]string{"cloud": "aws"}, "")
		waitForUnallocatedPreparingExecution(t, h.ctx, h.store, blocked)
		cancelWorkflowRun(t, h.operations.client, h.operations.baseURL, blocked)
		h.terminal(t, blocked, "cancelled")
		var repeat struct {
			Draft struct {
				Labels map[string]string `json:"labels"`
			} `json:"draft"`
		}
		h.decode(t, h.operations.request(http.MethodGet, "/v1/runs/"+pinned+"/repeat-draft", nil, http.StatusOK, nil), &repeat)
		if !maps.Equal(repeat.Draft.Labels, map[string]string{"cloud": "aws", "team": "appsec"}) {
			t.Fatalf("repeat changed labels after fleet restart: %+v", repeat.Draft.Labels)
		}
	})

	step("failed_run_resume_preserves_labels", func(t *testing.T) {
		marker := filepath.Join(h.temporary, "fail-enabled")
		if err := os.WriteFile(marker, []byte("enabled"), 0600); err != nil {
			t.Fatal(err)
		}
		failed := h.create(t, map[string]string{"cloud": "azure", "team": "appsec"}, "/fail-once")
		h.terminal(t, failed, "failed")
		executions, err := h.store.ListStageExecutions(h.ctx, failed)
		if err != nil || len(executions) != 1 {
			t.Fatalf("failed executions=%v err=%v", executions, err)
		}
		h.operations.request(http.MethodPost, "/v1/runs/"+failed+"/resume", map[string]string{"stageExecutionId": executions[0].StageExecutionID}, http.StatusAccepted, nil)
		// Keep the deterministic failure enabled for the new StageExecution.
		// Its outcome must not broaden the resumed Run's pinned placement.
		status := h.terminal(t, failed, "failed")
		if err := os.Remove(marker); err != nil {
			t.Fatal(err)
		}
		if !maps.Equal(status.Labels, map[string]string{"cloud": "azure", "team": "appsec"}) {
			t.Fatalf("resume changed labels: %+v", status.Labels)
		}
		executions, err = h.store.ListStageExecutions(h.ctx, failed)
		if err != nil || len(executions) != 2 {
			t.Fatalf("resumed executions=%v err=%v", executions, err)
		}
		for _, execution := range executions {
			allocations, err := h.store.ListStageAllocations(h.ctx, execution.StageExecutionID)
			if err != nil || len(allocations) != 1 || !slices.Equal(allocations[0].RuntimePlacementLabels, []string{"cloud/azure"}) {
				t.Fatalf("resumed allocation changed placement: %v err=%v", allocations, err)
			}
		}
	})

	step("empty_requirements_allow_undeclared_runtime", func(t *testing.T) {
		h.stopRuntime("placement-local")
		h.stopRuntime("placement-cloud-prefix")
		h.stopRuntime("placement-cloud-no-tool")
		h.stopRuntime("placement-cloud")
		undeclared := h.startRuntime("placement-undeclared", "", true)
		// Stopping a process does not revoke its Server-confirmed lease. Wait
		// for the ordinary retirement boundary before testing an unrestricted
		// launch against the sole remaining live process.
		waitForObservedRuntimeAgents(t, h.ctx, h.server, []*childProcess{undeclared}, h.operations.client, h.operations.baseURL, func(agents []observedRuntimeAgent) bool {
			return len(agents) == 1 && agents[0].ConfirmedLeaseUntil != nil && agents[0].State == "idle"
		})
		h.complete(t, h.create(t, map[string]string{"team": "appsec"}, ""), "")
	})
}

type placementProcessHarness struct {
	ctx                                                                          context.Context
	root, temporary, privateURL, python, scannerBin, emptyBin, templates, caFile string
	generator                                                                    localpki.Generator
	leaf                                                                         localpki.LeafOptions
	controlClient                                                                *http.Client
	server                                                                       *childProcess
	runtimes                                                                     map[string]*childProcess
	identities                                                                   map[string]localpki.Paths
	operations                                                                   *runtimeOperations
	store                                                                        *runstore.PostgresStore
}

func startPlacementStack(t *testing.T) *placementProcessHarness {
	return startPlacementStackWithConfiguration(t, nil)
}

func startPlacementStackWithConfiguration(t *testing.T, configure func(string)) *placementProcessHarness {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	t.Cleanup(cancel)
	h := &placementProcessHarness{ctx: ctx, root: repoRoot(t), temporary: t.TempDir(), runtimes: map[string]*childProcess{}, identities: map[string]localpki.Paths{}}
	t.Cleanup(func() {
		if t.Failed() {
			if h.server != nil {
				t.Logf("Server:\n%s", h.server.logs.redacted(publicToken))
			}
			for name, runtime := range h.runtimes {
				t.Logf("%s:\n%s", name, runtime.logs.redacted(publicToken))
			}
		}
	})
	databaseURL := isolatedDatabase(t, ctx, os.Getenv("CONTRACTOR_TEST_DATABASE_URL"))
	serverBinary := sharedServerBinary(t)
	runChecked(t, h.root, map[string]string{"CONTRACTOR_DATABASE_URL": databaseURL}, serverBinary, "migrate")
	pool, err := pgxpool.New(ctx, databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(pool.Close)
	h.store = runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
	pkiRoot := filepath.Join(h.temporary, "pki")
	ca, err := h.generator.InitCA(pkiRoot, false)
	if err != nil {
		t.Fatal(err)
	}
	h.caFile = ca.Certificate
	h.leaf = localpki.LeafOptions{IPAddresses: []net.IP{net.ParseIP("127.0.0.1")}}
	control, err := h.generator.IssueControlPlane(pkiRoot, localpki.ControlPlaneOptions{LeafOptions: h.leaf, URI: "urn:contractor:control-plane:placement-e2e"})
	if err != nil {
		t.Fatal(err)
	}
	h.controlClient = newMTLSClient(t, ca.Certificate, control)
	publicAddress, privateAddress := freeAddress(t), freeAddress(t)
	h.privateURL = "https://" + privateAddress
	h.operations = &runtimeOperations{t: t, client: &http.Client{Timeout: 5 * time.Second}, baseURL: "http://" + publicAddress}
	configRoot := filepath.Join(h.temporary, "configs")
	if err := os.CopyFS(configRoot, os.DirFS(filepath.Join(h.root, "tests", "e2e", "testdata", "runtime-placement"))); err != nil {
		t.Fatal(err)
	}
	for _, directory := range []string{"model-policies", "llm-gateways", "audit-profiles", "audit-standards", "skills"} {
		if err := os.MkdirAll(filepath.Join(configRoot, directory), 0700); err != nil {
			t.Fatal(err)
		}
	}
	if configure != nil {
		configure(configRoot)
	}
	h.server = startProcess(t, "Placement Server", h.root, map[string]string{
		"CONTRACTOR_DATABASE_URL":         databaseURL,
		"CONTRACTOR_OPERATOR_CONFIG_ROOT": configRoot,
		"CONTRACTOR_PUBLIC_LISTEN":        publicAddress, "CONTRACTOR_PRIVATE_LISTEN": privateAddress, "CONTRACTOR_PRIVATE_URL": h.privateURL,
		"CONTRACTOR_CA_FILE": ca.Certificate, "CONTRACTOR_CONTROL_PLANE_CERT_FILE": control.Certificate, "CONTRACTOR_CONTROL_PLANE_KEY_FILE": control.PrivateKey,
		"CONTRACTOR_PUBLIC_BEARER_TOKEN": publicToken,
		"CONTRACTOR_LOCAL_AUTH_FILE":     writeE2ELocalAuth(t, h.temporary, "placement-user"),
		"CONTRACTOR_BROWSER_ORIGINS":     "https://ui.contractor.invalid",
	}, serverBinary, "serve")
	waitForHTTP(t, ctx, h.server, h.operations.client, h.operations.baseURL+"/readyz", http.StatusOK)
	h.python = filepath.Join(h.root, "runtime", ".venv", "bin", "python")
	h.scannerBin, h.emptyBin, h.templates = filepath.Join(h.temporary, "bin"), filepath.Join(h.temporary, "empty-bin"), filepath.Join(h.temporary, "templates")
	for _, path := range []string{h.scannerBin, h.emptyBin, h.templates} {
		if err := os.Mkdir(path, 0700); err != nil {
			t.Fatal(err)
		}
	}
	// A real tool Worker runs this deterministic external process. No Gateway or
	// scanner installation is required, and the failing invocation is repeatable.
	writePlacementScannerFixture(t, h.scannerBin, h.python, filepath.Join(h.temporary, "fail-enabled"))
	return h
}

func (h *placementProcessHarness) startRuntime(name, placement string, capable bool) *childProcess {
	return h.startRuntimeSlots(name, placement, capable, 1)
}

func (h *placementProcessHarness) startRuntimeSlots(name, placement string, capable bool, slots int) *childProcess {
	t := h.operations.t
	t.Helper()
	identity, exists := h.identities[name]
	if !exists {
		var err error
		identity, err = h.generator.IssueAgent(filepath.Join(h.temporary, "pki"), name, h.leaf)
		if err != nil {
			t.Fatal(err)
		}
		h.identities[name] = identity
	}
	address := freeAddress(t)
	baseURL := "https://" + address
	args := []string{"-m", "contractor_runtime", "--allocation-slots", strconv.Itoa(slots), "--control-plane-url", h.privateURL,
		"--advertised-control-url", baseURL, "--advertised-a2a-url", baseURL,
		"--ca-file", h.caFile, "--certificate-file", identity.Certificate, "--private-key-file", identity.PrivateKey,
		"--listen", address, "--work-root", filepath.Join(h.temporary, name), "--request-timeout-seconds", "10", "--shutdown-grace-seconds", "5", "--allowed-target-network", "127.0.0.0/8"}
	if placement != "" {
		args = append(args, "--placement-label", placement)
	}
	path := h.emptyBin
	if capable {
		path = h.scannerBin
	}
	process := startProcess(t, name, filepath.Join(h.root, "runtime"), map[string]string{"PATH": path, "PYTHONUNBUFFERED": "1", "NUCLEI_TEMPLATES_DIR": h.templates, "CONTRACTOR_PLACEMENT_LABELS": ""}, h.python, args...)
	h.runtimes[name] = process
	waitForHTTP(t, h.ctx, process, h.controlClient, baseURL+"/healthz", http.StatusOK)
	waitForProcessLog(t, h.ctx, process, "runtime agent registered")
	// Read the real Operations projection, including the actual startup label.
	var agents struct {
		Items []struct {
			RuntimeAgentID string `json:"runtimeAgentId"`
			Live           *struct {
				PlacementLabels []string `json:"placementLabels"`
			} `json:"live"`
		} `json:"items"`
	}
	h.decode(t, h.operations.request(http.MethodGet, "/v1/operations/runtime-agent-principals?limit=20", nil, http.StatusOK, nil), &agents)
	found := false
	for _, agent := range agents.Items {
		if agent.RuntimeAgentID == h.principalID(t, name) && agent.Live != nil {
			found = slices.Equal(agent.Live.PlacementLabels, func() []string {
				if placement == "" {
					return []string{}
				}
				return []string{placement}
			}())
		}
	}
	if !found {
		t.Fatalf("Operations did not retain %s startup declaration %q: %+v", name, placement, agents)
	}
	return process
}

func (h *placementProcessHarness) stopRuntime(name string) {
	h.runtimes[name].stop(h.operations.t)
	delete(h.runtimes, name)
}

func (h *placementProcessHarness) principalID(t *testing.T, name string) string {
	t.Helper()
	identity := h.identities[name]
	pair, err := tls.LoadX509KeyPair(identity.Certificate, identity.PrivateKey)
	if err != nil {
		t.Fatal(err)
	}
	certificate, err := x509.ParseCertificate(pair.Certificate[0])
	if err != nil {
		t.Fatal(err)
	}
	principal, err := mtls.RuntimeAgentID(certificate)
	if err != nil {
		t.Fatal(err)
	}
	return principal
}

func (h *placementProcessHarness) create(t *testing.T, labels map[string]string, path string) string {
	t.Helper()
	// Keep target policy separate from the private Runtime service destinations.
	fixture := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	t.Cleanup(fixture.Close)
	data := h.operations.request(http.MethodPost, "/v1/runs", map[string]any{"workflow": "placement-probe@1", "parameters": map[string]string{"target": fixture.URL + path}, "artifacts": map[string]any{}, "labels": labels}, http.StatusAccepted, map[string]string{"Idempotency-Key": "placement-" + randomHex(t, 8)})
	var run runCreateResponse
	h.decode(t, data, &run)
	if run.RunID == "" {
		t.Fatal("missing created Run")
	}
	return run.RunID
}

func (h *placementProcessHarness) terminal(t *testing.T, runID, state string) runStatus {
	t.Helper()
	runtimes := make([]*childProcess, 0, len(h.runtimes))
	for _, process := range h.runtimes {
		runtimes = append(runtimes, process)
	}
	status := waitForRunTerminalAcross(t, h.ctx, h.server, runtimes, h.operations.client, h.operations.baseURL, runID)
	if status.State != state {
		attempts, _ := json.Marshal(status.Attempts)
		t.Fatalf("Run %s reached %s, want %s; attempts=%s", runID, status.State, state, attempts)
	}
	return status
}

func (h *placementProcessHarness) complete(t *testing.T, runID, placement string) runstore.StageAllocation {
	t.Helper()
	status := h.terminal(t, runID, "succeeded")
	if len(status.Attempts) != 1 || status.Attempts[0].Metrics == nil || status.Attempts[0].Metrics.ModelCalls != 0 || status.Attempts[0].Metrics.ToolCalls != 2 || !status.Attempts[0].Metrics.ReportsComplete || len(status.Outputs) != 1 {
		t.Fatalf("Run did not complete as one model-free tool invocation: %+v", status)
	}
	allocation := onlyRunAllocation(t, h.ctx, h.store, runID)
	want := []string{}
	if placement != "" {
		want = []string{placement}
	}
	if !slices.Equal(allocation.RuntimePlacementLabels, want) {
		t.Fatalf("allocation at wrong location: %+v, want=%v", allocation, want)
	}
	return allocation
}

func (h *placementProcessHarness) decode(t *testing.T, data []byte, target any) {
	t.Helper()
	if err := json.Unmarshal(data, target); err != nil {
		t.Fatal(err)
	}
}
