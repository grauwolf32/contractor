//go:build e2e

package e2e

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/localpki"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestHTTPAndCaidoAcrossHeterogeneousRuntimeProcesses(t *testing.T) {
	if testing.Short() {
		t.Skip("end-to-end process test")
	}
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Fatal("CONTRACTOR_TEST_DATABASE_URL is required")
	}
	repositoryRoot := repoRoot(t)
	temporaryRoot := t.TempDir()
	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Second)
	defer cancel()

	target := newHTTPCaidoTarget()
	t.Cleanup(target.close)
	forwardProxy := newHTTPCaidoForwardProxy(target.URL())
	t.Cleanup(forwardProxy.close)
	oldCaido := newFakeCaidoControl("old", true)
	t.Cleanup(oldCaido.close)
	newCaido := newFakeCaidoControl("new", false)
	t.Cleanup(newCaido.close)
	gateway := newHTTPCaidoGateway(llmGatewayToken, forwardProxy.TargetURL())
	t.Cleanup(gateway.close)

	configRoot := stageE2EConfiguration(
		t, filepath.Join(repositoryRoot, "configs"),
		filepath.Join(temporaryRoot, "configs"), gateway.URL(),
	)
	writeHTTPAnalysisE2EWorkflow(t, configRoot)

	isolateURL := isolatedDatabase(t, ctx, databaseURL)
	serverBinary := filepath.Join(temporaryRoot, "contractor-server")
	runChecked(t, repositoryRoot, nil, "go", "build", "-o", serverBinary, "./cmd/contractor-server")
	runChecked(t, repositoryRoot, map[string]string{
		"CONTRACTOR_DATABASE_URL": isolateURL,
	}, serverBinary, "migrate")

	pkiRoot := filepath.Join(temporaryRoot, "pki")
	generator := localpki.Generator{}
	caPaths, err := generator.InitCA(pkiRoot, false)
	if err != nil {
		t.Fatalf("initialize test CA: %v", err)
	}
	leaf := localpki.LeafOptions{IPAddresses: []net.IP{net.ParseIP("127.0.0.1")}}
	controlPlanePaths, err := generator.IssueControlPlane(pkiRoot, localpki.ControlPlaneOptions{
		LeafOptions: leaf, URI: "urn:contractor:control-plane:http-caido-e2e",
	})
	if err != nil {
		t.Fatalf("issue Control Plane certificate: %v", err)
	}
	httpIdentity, err := generator.IssueAgent(pkiRoot, "http-caido-http-agent", leaf)
	if err != nil {
		t.Fatalf("issue HTTP Runtime Agent certificate: %v", err)
	}
	caidoIdentity, err := generator.IssueAgent(pkiRoot, "http-caido-caido-agent", leaf)
	if err != nil {
		t.Fatalf("issue Caido Runtime Agent certificate: %v", err)
	}

	publicAddress, privateAddress := freeAddress(t), freeAddress(t)
	httpRuntimeAddress, caidoRuntimeAddress := freeAddress(t), freeAddress(t)
	caidoAdvertisedAddress := freeAddress(t)
	publicBaseURL := "http://" + publicAddress
	privateBaseURL := "https://" + privateAddress
	httpRuntimeBaseURL := "https://" + httpRuntimeAddress
	caidoRuntimeBaseURL := "https://" + caidoRuntimeAddress
	caidoAdvertisedBaseURL := "https://" + caidoAdvertisedAddress
	userID := "http-caido-e2e-user-" + randomHex(t, 8)
	masterKeyFile := filepath.Join(temporaryRoot, "credential-master-key")
	masterKey := base64.StdEncoding.EncodeToString(bytes.Repeat([]byte{0x4c}, 32))
	if err := os.WriteFile(masterKeyFile, []byte(masterKey), 0o600); err != nil {
		t.Fatal(err)
	}
	server := startProcess(t, "Go Server HTTP/Caido", repositoryRoot, map[string]string{
		"CONTRACTOR_DATABASE_URL":            isolateURL,
		"CONTRACTOR_CONFIG_ROOT":             configRoot,
		"CONTRACTOR_PUBLIC_LISTEN":           publicAddress,
		"CONTRACTOR_PRIVATE_LISTEN":          privateAddress,
		"CONTRACTOR_PRIVATE_URL":             privateBaseURL,
		"CONTRACTOR_CA_FILE":                 caPaths.Certificate,
		"CONTRACTOR_CONTROL_PLANE_CERT_FILE": controlPlanePaths.Certificate,
		"CONTRACTOR_CONTROL_PLANE_KEY_FILE":  controlPlanePaths.PrivateKey,
		"CONTRACTOR_LLM_GATEWAY_TOKEN":       llmGatewayToken,
		"CONTRACTOR_PUBLIC_USER_ID":          userID,
		"CONTRACTOR_PUBLIC_BEARER_TOKEN":     publicToken,
		"CONTRACTOR_LOCAL_AUTH_FILE":         writeE2ELocalAuth(t, temporaryRoot, userID),
		"CONTRACTOR_BROWSER_ORIGINS":         "https://ui.contractor.invalid",
	}, serverBinary, "serve", "--credential-master-key-file", masterKeyFile)
	publicClient := &http.Client{Timeout: 8 * time.Second}
	waitForHTTP(t, ctx, server, publicClient, publicBaseURL+"/readyz", http.StatusOK)

	python := filepath.Join(repositoryRoot, "runtime", ".venv", "bin", "python")
	if info, statErr := os.Stat(python); statErr != nil || info.IsDir() {
		t.Fatalf("Python Runtime environment is missing at %s; run 'cd runtime && uv sync --locked'", python)
	}
	httpWorkRoot := filepath.Join(temporaryRoot, "runtime-http-work")
	caidoWorkRoot := filepath.Join(temporaryRoot, "runtime-caido-work")
	httpRuntime := startRuntimeWithAdapters(
		t, "Python Runtime Agent HTTP proxy", repositoryRoot, python,
		privateBaseURL, httpRuntimeAddress, httpWorkRoot, caPaths.Certificate,
		httpIdentity, []string{"http-proxy@1"},
	)
	releaseProxy := newRuntimeReleaseLossProxy(
		t, caidoAdvertisedAddress, caidoRuntimeBaseURL, caPaths.Certificate,
		caidoIdentity, controlPlanePaths,
	)
	caidoRuntime := startProcess(
		t, "Python Runtime Agent Caido", filepath.Join(repositoryRoot, "runtime"),
		map[string]string{"PYTHONUNBUFFERED": "1"},
		python, "-m", "contractor_runtime",
		"--control-plane-url", privateBaseURL,
		"--advertised-control-url", caidoAdvertisedBaseURL,
		"--advertised-a2a-url", caidoAdvertisedBaseURL,
		"--ca-file", caPaths.Certificate,
		"--certificate-file", caidoIdentity.Certificate,
		"--private-key-file", caidoIdentity.PrivateKey,
		"--listen", caidoRuntimeAddress,
		"--work-root", caidoWorkRoot,
		"--request-timeout-seconds", "15",
		"--shutdown-grace-seconds", "5",
		"--runtime-adapter", "caido-graphql@1",
	)
	t.Cleanup(func() {
		if !t.Failed() {
			return
		}
		targetCalls, targetFailures := target.snapshot()
		proxyCalls, proxyFailures := forwardProxy.snapshot()
		oldOperations, oldFailures := oldCaido.snapshot()
		newOperations, newFailures := newCaido.snapshot()
		releaseDrops, releaseFailures := releaseProxy.snapshot()
		t.Logf("HTTP/Caido fixture diagnostics: target=%d/%v proxy=%d/%v old=%v/%v new=%v/%v release=%d/%v gateway=%v",
			targetCalls, targetFailures, proxyCalls, proxyFailures, oldOperations, oldFailures,
			newOperations, newFailures, releaseDrops, releaseFailures, gateway.Failures())
		for _, process := range []*childProcess{server, httpRuntime, caidoRuntime} {
			t.Logf("%s logs:\n%s", process.name, process.logs.redacted(
				publicToken, llmGatewayToken, httpCaidoProxyCanary, httpCaidoTokenCanary,
				httpCaidoSessionCanary, httpCaidoRawCanary,
			))
		}
	})
	controlClient := newMTLSClient(t, caPaths.Certificate, controlPlanePaths)
	waitForHTTP(t, ctx, httpRuntime, controlClient, httpRuntimeBaseURL+"/healthz", http.StatusOK)
	waitForHTTP(t, ctx, caidoRuntime, controlClient, caidoAdvertisedBaseURL+"/healthz", http.StatusOK)
	waitForProcessLog(t, ctx, httpRuntime, "runtime agent registered")
	waitForProcessLog(t, ctx, caidoRuntime, "runtime agent registered")
	runtimes := []*childProcess{httpRuntime, caidoRuntime}

	agents, _ := waitForObservedRuntimeAgents(
		t, ctx, server, runtimes, publicClient, publicBaseURL,
		func(items []observedRuntimeAgent) bool {
			if len(items) != 2 {
				return false
			}
			foundHTTP, foundCaido := false, false
			for _, item := range items {
				if item.SlotState != "idle" || item.ConfirmedLeaseUntil == nil {
					return false
				}
				switch strings.Join(item.SupportedRuntimeAdapters, ",") {
				case "http-proxy@1":
					foundHTTP = true
				case "caido-graphql@1":
					foundCaido = true
				}
			}
			return foundHTTP && foundCaido
		},
	)
	if len(agents) != 2 {
		t.Fatalf("heterogeneous Runtime snapshot = %+v", agents)
	}

	pool, err := pgxpool.New(ctx, isolateURL)
	if err != nil {
		t.Fatalf("open assertion database: %v", err)
	}
	t.Cleanup(pool.Close)
	store := runstore.NewPostgresStore(pool)
	operations := &runtimeOperations{t: t, client: publicClient, baseURL: publicBaseURL}
	operations.createRuntimeCredential("http-caido-proxy", "http-proxy-bearer@1", map[string]any{
		"token": httpCaidoProxyCanary,
	})
	operations.createRuntimeCredential("http-caido-control", "caido-bearer@1", map[string]any{
		"token": httpCaidoTokenCanary,
	})
	httpConfig := operations.publishRuntimeConfig("http-caido-http", "1", map[string]any{
		"worker": map[string]any{"httpProxy": map[string]any{
			"adapter": "http-proxy@1", "proxyUrl": forwardProxy.URL(),
			"credential": "http-caido-proxy", "targets": []string{"tool-http"},
		}},
	})
	oldConfig := operations.publishRuntimeConfig("http-caido-old", "1", map[string]any{
		"worker": map[string]any{"caido": map[string]any{
			"adapter": "caido-graphql@1", "endpoint": oldCaido.URL(),
			"credential": "http-caido-control", "requestTimeoutSeconds": 15,
		}},
	})
	newConfig := operations.publishRuntimeConfig("http-caido-new", "1", map[string]any{
		"worker": map[string]any{"caido": map[string]any{
			"adapter": "caido-graphql@1", "endpoint": newCaido.URL(),
			"credential": "http-caido-control", "requestTimeoutSeconds": 15,
		}},
	})
	operations.createRuntimeLabel("http-route", httpConfig)
	operations.createRuntimeLabel("caido", oldConfig)
	operations.createRuntimeLabel("agent-caido", oldConfig)
	principals := operations.runtimeAgentPrincipals()
	httpPrincipal := principalWithAdapters(t, principals, "http-proxy@1")
	caidoPrincipal := principalWithAdapters(t, principals, "caido-graphql@1")
	caidoPrincipal = operations.replacePrincipalLabels(caidoPrincipal, []string{"agent-caido"})

	httpInitial := createHTTPCaidoRun(
		operations, "http-analysis-e2e@1", "http-caido-http-initial", forwardProxy.TargetURL(),
		[]string{"http-route"},
	)
	httpInitialStatus := waitForRunAcross(
		t, ctx, server, runtimes, gateway, publicClient, publicBaseURL, httpInitial.RunID,
	)
	httpInitialAllocation := onlyRunAllocation(t, ctx, store, httpInitial.RunID)
	assertHTTPCaidoAllocation(
		t, httpInitialAllocation, httpPrincipal.RuntimeAgentID, contracts.RuntimeAdapterHTTPProxy,
	)
	assertHTTPCaidoReport(t, operations, publicClient, publicBaseURL, httpInitialStatus, httpCaidoHTTPReport)
	assertHTTPBodyArtifact(t, operations, publicClient, publicBaseURL, httpInitial.RunID)

	oldRun := createHTTPCaidoRun(
		operations, "security-analysis@1", "http-caido-old-run", "target.example",
		[]string{"caido"},
	)
	select {
	case <-oldCaido.first:
	case <-ctx.Done():
		t.Fatalf("wait for blocked old Caido request: %v", ctx.Err())
	}
	operations.rebindRuntimeLabel("agent-caido", "1", newConfig)
	oldCaido.releaseFirst()
	oldStatus := waitForRunAcross(
		t, ctx, server, runtimes, gateway, publicClient, publicBaseURL, oldRun.RunID,
	)
	select {
	case <-releaseProxy.dropped:
	case <-ctx.Done():
		t.Fatalf("wait for lost release acknowledgement: %v", ctx.Err())
	}
	oldAllocation := onlyRunAllocation(t, ctx, store, oldRun.RunID)
	assertHTTPCaidoAllocation(
		t, oldAllocation, caidoPrincipal.RuntimeAgentID, contracts.RuntimeAdapterCaidoGraphQL,
	)
	assertAgentLabelPin(t, oldAllocation, "agent-caido", 1, oldConfig)
	assertHTTPCaidoReport(t, operations, publicClient, publicBaseURL, oldStatus, httpCaidoCaidoReport)

	// The Caido release acknowledgement remains unavailable here. A separate
	// HTTP Run must still claim and finish on the other physical slot.
	httpUnrelated := createHTTPCaidoRun(
		operations, "http-analysis-e2e@1", "http-caido-http-unrelated", forwardProxy.TargetURL(),
		[]string{"http-route"},
	)
	httpUnrelatedStatus := waitForRunAcross(
		t, ctx, server, runtimes, gateway, publicClient, publicBaseURL, httpUnrelated.RunID,
	)
	httpUnrelatedAllocation := onlyRunAllocation(t, ctx, store, httpUnrelated.RunID)
	assertHTTPCaidoAllocation(
		t, httpUnrelatedAllocation, httpPrincipal.RuntimeAgentID, contracts.RuntimeAdapterHTTPProxy,
	)
	assertHTTPCaidoReport(
		t, operations, publicClient, publicBaseURL, httpUnrelatedStatus, httpCaidoHTTPReport,
	)
	assertHTTPBodyArtifact(t, operations, publicClient, publicBaseURL, httpUnrelated.RunID)

	releaseProxy.allowReleaseAcknowledgement()
	waitForObservedRuntimeAgents(
		t, ctx, server, runtimes, publicClient, publicBaseURL,
		func(items []observedRuntimeAgent) bool {
			if len(items) != 2 {
				return false
			}
			for _, item := range items {
				if item.SlotState != "idle" || item.ObservedState != "idle" ||
					item.CurrentAllocationID != nil || item.AuthoritativeAllocationID != nil {
					return false
				}
			}
			return true
		},
	)

	newRun := createHTTPCaidoRun(
		operations, "security-analysis@1", "http-caido-new-run", "target.example",
		[]string{"caido"},
	)
	newStatus := waitForRunAcross(
		t, ctx, server, runtimes, gateway, publicClient, publicBaseURL, newRun.RunID,
	)
	newAllocation := onlyRunAllocation(t, ctx, store, newRun.RunID)
	assertHTTPCaidoAllocation(
		t, newAllocation, caidoPrincipal.RuntimeAgentID, contracts.RuntimeAdapterCaidoGraphQL,
	)
	assertAgentLabelPin(t, newAllocation, "agent-caido", 2, newConfig)
	assertHTTPCaidoReport(t, operations, publicClient, publicBaseURL, newStatus, httpCaidoCaidoReport)

	finalAgents, finalSnapshot := waitForObservedRuntimeAgents(
		t, ctx, server, runtimes, publicClient, publicBaseURL,
		func(items []observedRuntimeAgent) bool {
			if len(items) != 2 {
				return false
			}
			for _, item := range items {
				if item.SlotState != "idle" || item.ObservedState != "idle" ||
					item.CurrentAllocationID != nil || item.AuthoritativeAllocationID != nil {
					return false
				}
			}
			return true
		},
	)
	if len(finalAgents) != 2 || !workRootEmpty(httpWorkRoot) || !workRootEmpty(caidoWorkRoot) {
		t.Fatalf("Runtime slots were not cleanly reusable: agents=%+v httpEmpty=%t caidoEmpty=%t",
			finalAgents, workRootEmpty(httpWorkRoot), workRootEmpty(caidoWorkRoot))
	}
	operations.evidence = append(operations.evidence, []byte(finalSnapshot))

	if calls, failures := target.snapshot(); calls != 2 || len(failures) != 0 {
		t.Fatalf("fake target = calls:%d failures:%v", calls, failures)
	}
	if calls, failures := forwardProxy.snapshot(); calls != 2 || len(failures) != 0 {
		t.Fatalf("forward proxy = calls:%d failures:%v", calls, failures)
	}
	if oldOperations, failures := oldCaido.snapshot(); !slices.Equal(oldOperations, []string{"RequestsByOffset", "CreateScope"}) || len(failures) != 0 {
		t.Fatalf("old Caido operations = %v failures=%v", oldOperations, failures)
	}
	if newOperations, failures := newCaido.snapshot(); !slices.Equal(newOperations, []string{"RequestsByOffset", "CreateScope"}) || len(failures) != 0 {
		t.Fatalf("new Caido operations = %v failures=%v", newOperations, failures)
	}
	if drops, failures := releaseProxy.snapshot(); drops == 0 || len(failures) != 0 {
		t.Fatalf("release response-loss proxy = drops:%d failures:%v", drops, failures)
	}
	if gateway.CompletedStages() != 4 || len(gateway.Failures()) != 0 {
		t.Fatalf("scripted Gateway = completed:%d failures:%v", gateway.CompletedStages(), gateway.Failures())
	}

	for _, runID := range []string{httpInitial.RunID, oldRun.RunID, httpUnrelated.RunID, newRun.RunID} {
		assertHTTPCaidoRetainedExecutionSafe(t, ctx, pool, runID)
	}
	assertRuntimeLabelSecretsAbsent(
		t, ctx, pool, operations.evidence,
		append([]*childProcess{server}, runtimes...),
		[]string{
			httpCaidoProxyCanary, httpCaidoTokenCanary, httpCaidoSessionCanary,
			httpCaidoRawCanary,
		},
	)
}

func createHTTPCaidoRun(
	operations *runtimeOperations,
	workflow, idempotencyKey, target string,
	labels []string,
) createdLabeledRun {
	operations.t.Helper()
	body := map[string]any{
		"workflow": workflow,
		"parameters": map[string]string{
			"objective": "Collect bounded evidence and publish one report",
			"target":    target, "authorization_scope": "single test-owned loopback target",
		},
		"artifacts": map[string]artifactRef{}, "labels": labels,
	}
	headers := map[string]string{"Idempotency-Key": idempotencyKey}
	data := operations.request(http.MethodPost, "/v1/runs", body, http.StatusAccepted, headers)
	var result createdLabeledRun
	if err := json.Unmarshal(data, &result); err != nil || result.RunID == "" ||
		(result.State != "initializing" && result.State != "running") {
		operations.t.Fatalf("create HTTP/Caido Run = (%+v, %v): %s", result, err, data)
	}
	replay := operations.requestResponse(http.MethodPost, "/v1/runs", body, http.StatusAccepted, headers)
	operations.requireReplay(replay, "HTTP/Caido Run create")
	var replayed createdLabeledRun
	if err := json.Unmarshal(replay.body, &replayed); err != nil || replayed.RunID != result.RunID ||
		!samePinnedRuntimeConfiguration(result, replayed) {
		operations.t.Fatalf("HTTP/Caido Run replay changed pinned identity: first=%+v replay=%+v error=%v",
			result, replayed, err)
	}
	return result
}

func assertHTTPCaidoAllocation(
	t *testing.T,
	allocation runstore.StageAllocation,
	wantRuntimeAgentID string,
	wantAdapter contracts.RuntimeAdapterRef,
) {
	t.Helper()
	if allocation.RuntimeAgentID != wantRuntimeAgentID || allocation.RuntimeConfiguration == nil ||
		!slices.Equal(
			allocation.RuntimeConfiguration.Provenance.RuntimeAdapters,
			[]contracts.RuntimeAdapterRef{wantAdapter},
		) {
		t.Fatalf("allocation placement/configuration = %+v, want principal %s adapter %s",
			allocation, wantRuntimeAgentID, wantAdapter)
	}
}

func assertAgentLabelPin(
	t *testing.T,
	allocation runstore.StageAllocation,
	label string,
	revision uint64,
	config runtimeconfig.Ref,
) {
	t.Helper()
	if allocation.RuntimeConfiguration == nil ||
		len(allocation.RuntimeConfiguration.Provenance.AgentLabels) != 1 {
		t.Fatalf("allocation has no exact Agent label provenance: %+v", allocation)
	}
	pin := allocation.RuntimeConfiguration.Provenance.AgentLabels[0]
	if pin.Label != label || pin.BindingRevision != revision || pin.Config.Name != config.Name ||
		pin.Config.Version != config.Version || pin.Config.Digest != config.Digest {
		t.Fatalf("Agent label pin = %+v, want %s revision %d config %+v", pin, label, revision, config)
	}
}

func assertHTTPCaidoReport(
	t *testing.T,
	operations *runtimeOperations,
	client *http.Client,
	baseURL string,
	status runStatus,
	want string,
) {
	t.Helper()
	if status.State != "succeeded" || len(status.Attempts) != 1 ||
		status.Attempts[0].State != "succeeded" || status.Outputs["report"].Revision == nil {
		t.Fatalf("HTTP/Caido terminal status = %+v", status)
	}
	data, mediaType := download(
		t, client, baseURL+"/v1/runs/"+url.PathEscape(status.RunID)+"/outputs/report",
	)
	if string(data) != want || mediaType != "text/markdown" {
		t.Fatalf("HTTP/Caido report = (%q, %q), want (%q, text/markdown)", data, mediaType, want)
	}
	operations.evidence = append(operations.evidence, append([]byte(nil), data...))
}

func assertHTTPBodyArtifact(
	t *testing.T,
	operations *runtimeOperations,
	client *http.Client,
	baseURL, runID string,
) {
	t.Helper()
	data := operations.request(
		http.MethodGet,
		"/v1/runs/"+url.PathEscape(runID)+"/artifacts?namespace=http&limit=100",
		nil, http.StatusOK, nil,
	)
	var page struct {
		Items []struct {
			Ref       artifactRef `json:"artifact"`
			MediaType string      `json:"mediaType"`
		} `json:"items"`
	}
	if err := json.Unmarshal(data, &page); err != nil {
		t.Fatal(err)
	}
	bodyName := ""
	for _, item := range page.Items {
		if strings.HasPrefix(item.Ref.Name, "http.body.") {
			if bodyName != "" || item.Ref.Revision == nil ||
				item.MediaType != "application/vnd.contractor.http-body+json" {
				t.Fatalf("invalid HTTP body artifact inventory: %+v", page.Items)
			}
			bodyName = item.Ref.Name
		}
	}
	if bodyName == "" {
		t.Fatalf("Run %s has no reserved HTTP body artifact: %+v", runID, page.Items)
	}
	payload, mediaType := download(
		t, client, baseURL+"/v1/runs/"+url.PathEscape(runID)+"/artifacts/http/"+
			url.PathEscape(bodyName),
	)
	if mediaType != "application/vnd.contractor.http-body+json" {
		t.Fatalf("HTTP body media type = %q", mediaType)
	}
	var envelope struct {
		SchemaVersion string `json:"schemaVersion"`
		Kind          string `json:"kind"`
		ContentType   string `json:"contentType"`
		Text          string `json:"text"`
	}
	if err := json.Unmarshal(payload, &envelope); err != nil || envelope.SchemaVersion != "1.0" ||
		envelope.Kind != "text" || !strings.Contains(envelope.Text, httpCaidoRawCanary) {
		t.Fatalf("HTTP body artifact envelope = (%+v, %v)", envelope, err)
	}
}

func assertHTTPCaidoRetainedExecutionSafe(
	t *testing.T,
	ctx context.Context,
	pool *pgxpool.Pool,
	runID string,
) {
	t.Helper()
	queries := []string{
		`SELECT COALESCE(runtime_config_snapshot::text || state_reason_code || state_reason_message, '')
FROM workflow_runs WHERE run_id = $1`,
		`SELECT COALESCE(string_agg(event::text, E'\n'), '') FROM planner_events WHERE run_id = $1`,
		`SELECT COALESCE(string_agg(data::text, E'\n'), '') FROM workflow_run_events WHERE run_id = $1`,
		`SELECT COALESCE(string_agg(session.state::text, E'\n'), '')
FROM planner_sessions AS session JOIN stage_executions AS execution
ON execution.stage_execution_id = session.stage_execution_id WHERE execution.run_id = $1`,
		`SELECT COALESCE(string_agg(report.report::text, E'\n'), '')
FROM planner_execution_reports AS report JOIN stage_executions AS execution
ON execution.stage_execution_id = report.stage_execution_id WHERE execution.run_id = $1`,
		`SELECT COALESCE(string_agg(report.report::text, E'\n'), '')
FROM allocation_execution_reports AS report JOIN stage_executions AS execution
ON execution.stage_execution_id = report.stage_execution_id WHERE execution.run_id = $1`,
		`SELECT COALESCE(string_agg(metrics.metrics::text || metrics.summary::text, E'\n'), '')
FROM stage_metrics AS metrics JOIN stage_executions AS execution
ON execution.stage_execution_id = metrics.stage_execution_id WHERE execution.run_id = $1`,
	}
	for _, query := range queries {
		var retained string
		if err := pool.QueryRow(ctx, query, runID).Scan(&retained); err != nil {
			t.Fatalf("read retained HTTP/Caido execution surface: %v", err)
		}
		for _, canary := range []string{
			httpCaidoProxyCanary, httpCaidoTokenCanary,
			httpCaidoSessionCanary, httpCaidoRawCanary,
		} {
			if strings.Contains(retained, canary) {
				t.Fatalf("retained execution surface exposed HTTP/Caido canary")
			}
		}
	}
}
