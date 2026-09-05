//go:build e2e

package e2e

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/localpki"
)

type auditProgramAudit struct {
	AuditID           string `json:"auditId"`
	ProjectID         string `json:"projectId"`
	State             string `json:"state"`
	Revision          uint64 `json:"revision"`
	CurrentRoundID    string `json:"currentRoundId,omitempty"`
	SubmittedRunCount int    `json:"submittedRunCount"`
	OutstandingRuns   int    `json:"outstandingRunCount"`
}

type auditProgramItem struct {
	ItemKey          string `json:"itemKey"`
	State            string `json:"state"`
	FinalDisposition string `json:"finalDisposition,omitempty"`
	AcceptedResult   *struct {
		Ref artifactRef `json:"ref"`
	} `json:"acceptedResult,omitempty"`
	Attempts []struct {
		RunID                 string `json:"runId,omitempty"`
		RunDeleted            bool   `json:"runDeleted"`
		CollectionDisposition string `json:"collectionDisposition,omitempty"`
	} `json:"attempts"`
}

type auditProgramCoverage struct {
	ItemKey  string `json:"itemKey"`
	Coverage struct {
		Status    string   `json:"status"`
		Requested []string `json:"requested"`
		Completed []string `json:"completed"`
		Gaps      []string `json:"gaps"`
	} `json:"coverage"`
}

type auditProgramReport struct {
	Status  string          `json:"status"`
	Machine json.RawMessage `json:"machine"`
	Summary string          `json:"summary"`
}

func TestAuditProgramsAcrossProductionProcesses(t *testing.T) {
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
		LeafOptions: leaf, URI: "urn:contractor:control-plane:audit-programs-e2e",
	})
	if err != nil {
		t.Fatalf("issue Control Plane certificate: %v", err)
	}
	agentPaths, err := generator.IssueAgent(pkiRoot, "audit-programs-agent", leaf)
	if err != nil {
		t.Fatalf("issue Runtime Agent certificate: %v", err)
	}

	gateway := newBlockedDomainGateway(llmGatewayToken, auditProgramGatewayStages())
	t.Cleanup(gateway.close)
	configRoot := stageE2EConfiguration(
		t, filepath.Join(repositoryRoot, "configs"),
		filepath.Join(temporaryRoot, "configs"), gateway.URL(),
	)
	publicAddress, privateAddress, runtimeAddress := freeAddress(t), freeAddress(t), freeAddress(t)
	publicBaseURL := "http://" + publicAddress
	privateBaseURL := "https://" + privateAddress
	runtimeBaseURL := "https://" + runtimeAddress
	userID := "audit-programs-user-" + randomHex(t, 8)
	localAuthFile := writeE2ELocalAuth(t, temporaryRoot, userID)
	server := startProcess(t, "Go Server", repositoryRoot, map[string]string{
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
		"CONTRACTOR_LOCAL_AUTH_FILE":         localAuthFile,
		"CONTRACTOR_BROWSER_ORIGINS":         "https://ui.contractor.invalid",
	}, serverBinary, "serve")

	client := &http.Client{Timeout: 8 * time.Second}
	waitForHTTP(t, ctx, server, client, publicBaseURL+"/readyz", http.StatusOK)
	python := filepath.Join(repositoryRoot, "runtime", ".venv", "bin", "python")
	if info, statErr := os.Stat(python); statErr != nil || info.IsDir() {
		t.Fatalf("Python Runtime environment is missing at %s; run 'cd runtime && uv sync --locked'", python)
	}
	workRoot := filepath.Join(temporaryRoot, "runtime-work")
	runtimeProcess := startProcess(
		t, "Python Runtime Agent", filepath.Join(repositoryRoot, "runtime"),
		map[string]string{"PYTHONUNBUFFERED": "1"},
		python, "-m", "contractor_runtime",
		"--control-plane-url", privateBaseURL,
		"--advertised-control-url", runtimeBaseURL,
		"--advertised-a2a-url", runtimeBaseURL,
		"--ca-file", caPaths.Certificate,
		"--certificate-file", agentPaths.Certificate,
		"--private-key-file", agentPaths.PrivateKey,
		"--listen", runtimeAddress,
		"--work-root", workRoot,
		"--request-timeout-seconds", "12",
		"--shutdown-grace-seconds", "5",
	)
	controlClient := newMTLSClient(t, caPaths.Certificate, controlPlanePaths)
	waitForHTTP(t, ctx, runtimeProcess, controlClient, runtimeBaseURL+"/healthz", http.StatusOK)
	waitForProcessLog(t, ctx, runtimeProcess, "runtime agent registered")

	assertAuditProfilesCompatible(t, client, publicBaseURL)
	project := createProjectResource(t, client, publicBaseURL)
	source := uploadProjectScopeArtifact(
		t, client, publicBaseURL, project.ProjectID,
		"sources", "audit-fixture", "application/zip", auditProgramSourceArchive(t),
	)
	checklist := uploadProjectScopeArtifact(
		t, client, publicBaseURL, project.ProjectID,
		"checklists", "source-checklist", "application/yaml", readAuditFixture(t, "checklist.yaml"),
	)
	openAPI := uploadProjectScopeArtifact(
		t, client, publicBaseURL, project.ProjectID,
		"openapi", "audit-fixture", "application/yaml", readAuditFixture(t, "openapi.yaml"),
	)

	checklistAudit := runAuditProgram(
		t, ctx, server, runtimeProcess, gateway, client, publicBaseURL, project.ProjectID,
		"source-checklist", map[string]artifactRef{"source": source, "checklist": checklist},
	)
	checklistCoverage := getAuditProgramCoverage(t, client, publicBaseURL, checklistAudit.AuditID)
	if len(checklistCoverage) != 2 || checklistCoverage[0].Coverage.Status != "satisfied" ||
		checklistCoverage[1].Coverage.Status != "inconclusive" {
		t.Fatalf("checklist coverage is not truthful: %+v", checklistCoverage)
	}
	assertAuditProgramReport(t, client, publicBaseURL, checklistAudit.AuditID, "completed-with-gaps")
	deleteCollectedAuditRuns(t, ctx, client, publicBaseURL, checklistAudit.AuditID)

	openAPIAudit := runAuditProgram(
		t, ctx, server, runtimeProcess, gateway, client, publicBaseURL, project.ProjectID,
		"openapi-operation-trace", map[string]artifactRef{"source": source, "openapi": openAPI},
	)
	openAPICoverage := getAuditProgramCoverage(t, client, publicBaseURL, openAPIAudit.AuditID)
	if len(openAPICoverage) != 2 {
		t.Fatalf("OpenAPI coverage count = %d, want 2", len(openAPICoverage))
	}
	allGaps := []string{}
	for _, row := range openAPICoverage {
		allGaps = append(allGaps, row.Coverage.Gaps...)
	}
	if !containsAuditGap(allGaps, "unsupported-callback") ||
		!containsAuditGap(openAPIAuditBaselineGaps(t, client, publicBaseURL, openAPIAudit.AuditID), "unsupported-webhook") {
		t.Fatalf("OpenAPI unsupported surfaces are missing: coverage=%v", allGaps)
	}
	assertAuditProgramReport(t, client, publicBaseURL, openAPIAudit.AuditID, "completed-with-gaps")
	deleteCollectedAuditRuns(t, ctx, client, publicBaseURL, openAPIAudit.AuditID)

	if gateway.CompletedStages() != 4 || len(gateway.Failures()) != 0 {
		t.Fatalf("Audit gateway stages/failures = %d/%v, want 4/none", gateway.CompletedStages(), gateway.Failures())
	}
	for _, secret := range []string{publicToken, llmGatewayToken} {
		if strings.Contains(server.logs.redacted(), secret) || strings.Contains(runtimeProcess.logs.redacted(), secret) {
			t.Fatal("process logs contain a configured secret")
		}
	}
}

func auditProgramGatewayStages() []domainGatewayStage {
	tools := []string{
		"list_skills", "list_source_files", "load_skill", "load_skill_resource",
		"open_source_archive", "read_artifact", "read_source", "search_source",
		"submit_check_result",
	}
	result := func(name, assessment string, completed, gaps []string, evidence []map[string]string) domainGatewayStage {
		return domainGatewayStage{name: name, tools: tools, steps: []domainGatewayStep{
			toolGatewayStep("open_source_archive", stageRefArguments("source", nil)),
			toolGatewayStep("read_source", fixedArguments(map[string]any{
				"path": "app.py", "start_line": 1, "max_lines": 100,
			})),
			toolGatewayStep("submit_check_result", fixedArguments(map[string]any{
				"assessment": assessment,
				"summary":    "Deterministic fixture assessment based on source/app.py.",
				"completed":  completed,
				"gaps":       gaps,
				"evidence":   evidence,
			})),
			finalGatewayStep("Canonical Audit result package published", map[string]domainArtifactBinding{
				"result": {namespace: "audit-check", name: "result"},
			}),
		}}
	}
	return []domainGatewayStage{
		result("checklist/check-authz", "satisfied", []string{"source-trace"}, []string{}, []map[string]string{{
			"kind": "source-trace", "summary": "Authorization call precedes the fixture object response.",
		}}),
		result("checklist/check-error-path", "inconclusive", []string{}, []string{"missing-error-path"}, nil),
		result("openapi/deleteWidget", "satisfied", []string{"operation-resolution"}, []string{}, []map[string]string{{
			"kind": "source-trace", "summary": "DELETE operation maps to source/app.py.",
		}}),
		result("openapi/getWidget", "satisfied", []string{"operation-resolution"}, []string{}, []map[string]string{{
			"kind": "source-trace", "summary": "GET operation maps to source/app.py.",
		}}),
	}
}

func assertAuditProfilesCompatible(t *testing.T, client *http.Client, baseURL string) {
	t.Helper()
	var page struct {
		Items []struct {
			Ref struct {
				Name    string `json:"name"`
				Version string `json:"version"`
			} `json:"ref"`
			ServerCompatible bool `json:"serverCompatible"`
		} `json:"items"`
	}
	auditProgramGET(t, client, baseURL+"/v1/audit-profiles?limit=100", &page)
	wanted := map[string]bool{"source-checklist@1": false, "openapi-operation-trace@1": false}
	for _, profile := range page.Items {
		selector := profile.Ref.Name + "@" + profile.Ref.Version
		if _, exists := wanted[selector]; exists {
			wanted[selector] = profile.ServerCompatible
		}
	}
	for selector, compatible := range wanted {
		if !compatible {
			t.Fatalf("Audit profile %s is absent or incompatible: %+v", selector, page.Items)
		}
	}
}

func runAuditProgram(
	t *testing.T,
	ctx context.Context,
	server, runtimeProcess *childProcess,
	gateway *domainGateway,
	client *http.Client,
	baseURL, projectID, profile string,
	inputs map[string]artifactRef,
) auditProgramAudit {
	t.Helper()
	body, err := json.Marshal(map[string]any{
		"profile": map[string]string{"name": profile, "version": "1"},
		"inputs":  inputs,
		"scope": map[string]string{
			"objective": "Exercise a bounded, non-certifying Audit fixture.",
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	request, err := http.NewRequest(
		http.MethodPost,
		baseURL+"/v1/projects/"+url.PathEscape(projectID)+"/audits",
		bytes.NewReader(body),
	)
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Authorization", "Bearer "+publicToken)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Idempotency-Key", "audit-program-"+profile)
	response := do(t, client, request, http.StatusCreated)
	var draft auditProgramAudit
	decodeAuditProgramResponse(t, response, &draft)
	response.Body.Close()
	if draft.AuditID == "" || draft.State != "draft" || draft.Revision != 1 {
		t.Fatalf("create %s Audit = %+v", profile, draft)
	}

	startRequest, err := http.NewRequest(
		http.MethodPost, baseURL+"/v1/audits/"+url.PathEscape(draft.AuditID)+"/start", nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	startRequest.Header.Set("Authorization", "Bearer "+publicToken)
	startRequest.Header.Set("Idempotency-Key", "audit-program-start-"+profile)
	startRequest.Header.Set("If-Match", fmt.Sprintf("\"%d\"", draft.Revision))
	startResponse := do(t, client, startRequest, http.StatusOK)
	var started struct {
		Audit auditProgramAudit  `json:"audit"`
		Items []auditProgramItem `json:"items"`
	}
	decodeAuditProgramResponse(t, startResponse, &started)
	startResponse.Body.Close()
	if started.Audit.State != "active" || len(started.Items) != 2 {
		t.Fatalf("start %s Audit = %+v", profile, started)
	}
	gateway.releaseBlockedRequest()
	return waitForAuditProgram(t, ctx, server, runtimeProcess, gateway, client, baseURL, draft.AuditID)
}

func waitForAuditProgram(
	t *testing.T,
	ctx context.Context,
	server, runtimeProcess *childProcess,
	gateway *domainGateway,
	client *http.Client,
	baseURL, auditID string,
) auditProgramAudit {
	t.Helper()
	ticker := time.NewTicker(150 * time.Millisecond)
	defer ticker.Stop()
	for {
		var audit auditProgramAudit
		if auditProgramTryGET(ctx, client, baseURL+"/v1/audits/"+url.PathEscape(auditID), &audit) == nil {
			switch audit.State {
			case "completed":
				if audit.SubmittedRunCount != 2 || audit.OutstandingRuns != 0 {
					t.Fatalf("completed Audit counters = %+v", audit)
				}
				return audit
			case "failed", "cancelled":
				t.Fatalf("Audit reached %s: %+v\nserver:\n%s\nruntime:\n%s\ngateway: %v",
					audit.State, audit, server.logs.redacted(publicToken, llmGatewayToken),
					runtimeProcess.logs.redacted(publicToken, llmGatewayToken), gateway.Failures())
			}
		}
		for _, process := range []*childProcess{server, runtimeProcess} {
			if exited, processErr := process.exited(); exited {
				t.Fatalf("%s exited while Audit was active: %v\n%s", process.name, processErr,
					process.logs.redacted(publicToken, llmGatewayToken))
			}
		}
		select {
		case <-ctx.Done():
			t.Fatalf("wait for Audit: %v\nserver:\n%s\nruntime:\n%s\ngateway: %v", ctx.Err(),
				server.logs.redacted(publicToken, llmGatewayToken),
				runtimeProcess.logs.redacted(publicToken, llmGatewayToken), gateway.Failures())
		case <-ticker.C:
		}
	}
}

func getAuditProgramCoverage(
	t *testing.T, client *http.Client, baseURL, auditID string,
) []auditProgramCoverage {
	t.Helper()
	var page struct {
		Items []auditProgramCoverage `json:"items"`
	}
	auditProgramGET(t, client, baseURL+"/v1/audits/"+url.PathEscape(auditID)+"/coverage?limit=100", &page)
	return page.Items
}

func openAPIAuditBaselineGaps(
	t *testing.T, client *http.Client, baseURL, auditID string,
) []string {
	t.Helper()
	var audit struct {
		Baseline struct {
			Inventory struct {
				Gaps []string `json:"gaps"`
			} `json:"inventory"`
		} `json:"baseline"`
	}
	auditProgramGET(t, client, baseURL+"/v1/audits/"+url.PathEscape(auditID), &audit)
	return audit.Baseline.Inventory.Gaps
}

func assertAuditProgramReport(
	t *testing.T, client *http.Client, baseURL, auditID, conclusion string,
) {
	t.Helper()
	var report auditProgramReport
	auditProgramGET(t, client, baseURL+"/v1/audits/"+url.PathEscape(auditID)+"/report", &report)
	if report.Status != "ready" || len(report.Machine) == 0 ||
		!strings.Contains(report.Summary, "Conclusion: "+conclusion) ||
		!strings.Contains(report.Summary, "not a security or compliance certification") {
		t.Fatalf("Audit report is not a truthful non-certifying report: %+v", report)
	}
}

func deleteCollectedAuditRuns(
	t *testing.T, ctx context.Context, client *http.Client, baseURL, auditID string,
) {
	t.Helper()
	var page struct {
		Items []auditProgramItem `json:"items"`
	}
	auditProgramGET(t, client, baseURL+"/v1/audits/"+url.PathEscape(auditID)+"/items?limit=100", &page)
	if len(page.Items) != 2 {
		t.Fatalf("Audit item count = %d, want 2", len(page.Items))
	}
	for _, item := range page.Items {
		if item.State != "settled" || item.FinalDisposition != "accepted-result" ||
			item.AcceptedResult == nil || len(item.Attempts) != 1 ||
			item.Attempts[0].CollectionDisposition != "accepted-result" || item.Attempts[0].RunID == "" {
			t.Fatalf("Audit item was not durably collected: %+v", item)
		}
		waitForAuditRunDeletable(t, ctx, client, baseURL, item.Attempts[0].RunID)
		request, err := http.NewRequest(
			http.MethodDelete, baseURL+"/v1/runs/"+url.PathEscape(item.Attempts[0].RunID), nil,
		)
		if err != nil {
			t.Fatal(err)
		}
		request.Header.Set("Authorization", "Bearer "+publicToken)
		response := do(t, client, request, http.StatusNoContent)
		response.Body.Close()
	}
	auditProgramGET(t, client, baseURL+"/v1/audits/"+url.PathEscape(auditID)+"/items?limit=100", &page)
	for _, item := range page.Items {
		if len(item.Attempts) != 1 || !item.Attempts[0].RunDeleted {
			t.Fatalf("deleted child Run lost its Audit tombstone: %+v", item)
		}
	}
}

func waitForAuditRunDeletable(
	t *testing.T, ctx context.Context, client *http.Client, baseURL, runID string,
) {
	t.Helper()
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()
	for {
		var run runStatus
		if auditProgramTryGET(ctx, client, baseURL+"/v1/runs/"+url.PathEscape(runID), &run) == nil && run.Deletable {
			return
		}
		select {
		case <-ctx.Done():
			t.Fatalf("wait for Audit child Run %s deletion: %v", runID, ctx.Err())
		case <-ticker.C:
		}
	}
}

func auditProgramGET(t *testing.T, client *http.Client, target string, output any) {
	t.Helper()
	if err := auditProgramTryGET(context.Background(), client, target, output); err != nil {
		t.Fatal(err)
	}
}

func decodeAuditProgramResponse(t *testing.T, response *http.Response, output any) {
	t.Helper()
	if err := json.NewDecoder(response.Body).Decode(output); err != nil {
		t.Fatalf("decode HTTP %d response: %v", response.StatusCode, err)
	}
}

func auditProgramTryGET(ctx context.Context, client *http.Client, target string, output any) error {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, target, nil)
	if err != nil {
		return err
	}
	request.Header.Set("Authorization", "Bearer "+publicToken)
	response, err := client.Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("GET %s returned HTTP %d", target, response.StatusCode)
	}
	if err := json.NewDecoder(response.Body).Decode(output); err != nil {
		return fmt.Errorf("decode GET %s: %w", target, err)
	}
	return nil
}

func auditProgramSourceArchive(t *testing.T) []byte {
	t.Helper()
	return auditProgramZip(t, map[string][]byte{
		"app.py": readAuditFixture(t, filepath.Join("source", "app.py")),
	})
}

func auditProgramZip(t *testing.T, files map[string][]byte) []byte {
	t.Helper()
	names := make([]string, 0, len(files))
	for name := range files {
		names = append(names, name)
	}
	sort.Strings(names)
	var output bytes.Buffer
	writer := zip.NewWriter(&output)
	for _, name := range names {
		header := &zip.FileHeader{Name: name, Method: zip.Store}
		header.SetMode(0o600)
		header.SetModTime(time.Date(2026, 9, 6, 0, 0, 0, 0, time.UTC))
		entry, err := writer.CreateHeader(header)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := entry.Write(files[name]); err != nil {
			t.Fatal(err)
		}
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	return output.Bytes()
}
