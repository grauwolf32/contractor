//go:build e2e

package e2e

import (
	"archive/zip"
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
)

type compositionAuditWire struct {
	auditProgramAudit
	Phase       auditdomain.AuditPhase `json:"phase"`
	Preparation *struct {
		Roles map[string]struct {
			Status      string `json:"status"`
			ExecutionID string `json:"executionId"`
			RunID       string `json:"runId"`
			Outputs     map[string]struct {
				Artifact auditstore.ExactArtifact `json:"artifact"`
			} `json:"outputs"`
		} `json:"roles"`
	} `json:"preparation,omitempty"`
	GeneratedInventory *auditservice.GeneratedInventoryProjection `json:"generatedInventory,omitempty"`
}

func TestAuditCompositionAcrossProductionProcesses(t *testing.T) {
	if testing.Short() {
		t.Fatal("Audit composition acceptance cannot run in short mode")
	}
	t.Setenv("CONTRACTOR_SCAN_BROWSER", "1")
	gateway := newDomainGateway(llmGatewayToken)
	gateway.stages = append(compositionGatewayStages(), compositionGatewayStages()...)
	t.Cleanup(gateway.close)
	h := startConfiguredScanStackWithBudget(t, 14*time.Minute, "", func(h *scanProcessHarness, root string) {
		stageE2EConfiguration(t, filepath.Join(h.repositoryRoot, "configs"), root, gateway.URL())
		h.gatewayToken = llmGatewayToken
		h.runtimePathPrefix, _ = installDomainValidators(t, h.temporaryRoot)
		h.workspaceStorageLocal = true
		replaceScanConfig(t, filepath.Join(root, "agent-templates/audit_nuclei_scan.yaml"), "http-missing-security-headers", "contractor-local-fixture")
		original, err := os.ReadFile(filepath.Join(root, "audit-profiles/source_openapi_sqlmap_scan.yaml"))
		if err != nil {
			t.Fatal(err)
		}
		profile := strings.ReplaceAll(string(original), "source-openapi-sqlmap-scan", "source-openapi-nuclei-fixture")
		profile = strings.ReplaceAll(profile, "audit-openapi-sqlmap-scan@1", "audit-openapi-nuclei-scan@1")
		if err := os.WriteFile(filepath.Join(root, "audit-profiles/source_openapi_nuclei_fixture.yaml"), []byte(profile), 0600); err != nil {
			t.Fatal(err)
		}
	}, "sqlmap", "nuclei")

	t.Run("generated_sqlmap_and_deleted_preparation_run", func(t *testing.T) {
		var project projectResourceResponse
		scanAuditPOST(t, h, "/v1/projects", map[string]string{"kind": "project", "name": "Generated OpenAPI acceptance", "activeChecks": "approval-required"}, 0, http.StatusCreated, &project)
		settings, err := json.Marshal(map[string]any{
			"schema": "contractor.audit.openapi-scan-settings.v1", "scanner": "sqlmap", "server": h.targetURL,
			"authentication": map[string]string{"bearerAuth": "local-fixture-token"}, "testParameters": []string{"q"},
			"operations": map[string]any{"#/paths/~1widgets~1{widget_id}/get": map[string]any{"parameters": map[string]any{"path:widget_id": "7", "query:q": "Milo"}}},
		})
		if err != nil {
			t.Fatal(err)
		}
		inputs := map[string]artifactRef{
			"source":   uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "fixture", "source", "application/zip", compositionSourceArchive(t)),
			"settings": uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "fixture", "settings", "application/json", settings),
		}
		var draft compositionAuditWire
		scanAuditPOST(t, h, "/v1/projects/"+project.ProjectID+"/audits", map[string]any{
			"profile": map[string]string{"name": "source-openapi-sqlmap-scan", "version": "1"}, "inputs": inputs,
			"scope": map[string]string{"objective": "Analyze the controlled source, generate OpenAPI and check its exact request."},
		}, 0, http.StatusCreated, &draft)
		var started struct {
			Audit compositionAuditWire `json:"audit"`
			Round *json.RawMessage     `json:"round"`
			Items []auditProgramItem   `json:"items"`
		}
		scanAuditPOST(t, h, "/v1/audits/"+draft.AuditID+"/start", nil, draft.Revision, http.StatusOK, &started)
		if started.Audit.Phase != auditdomain.AuditPhasePreparing || started.Round != nil || len(started.Items) != 0 || started.Audit.CurrentRoundID != "" {
			t.Fatalf("start invented an inventory: %+v", started)
		}
		var prepared compositionAuditWire
		waitScanCondition(t, h, "accepted generated worklist", func() bool {
			if err := auditProgramTryGET(h.ctx, h.client, h.baseURL+"/v1/audits/"+draft.AuditID, &prepared); err != nil {
				return false
			}
			if prepared.State == "failed" {
				t.Fatalf("preparation failed: %+v", prepared)
			}
			return prepared.Phase == auditdomain.AuditPhaseRounds
		})
		if prepared.Preparation == nil || prepared.GeneratedInventory == nil || prepared.GeneratedInventory.ItemCount != 1 || gateway.CompletedStages() != 4 || len(gateway.Failures()) != 0 {
			t.Fatalf("generated journey did not complete four real preparation stages: %+v stages=%d failures=%v", prepared, gateway.CompletedStages(), gateway.Failures())
		}
		role := prepared.Preparation.Roles["generate-api"]
		api := role.Outputs["api"].Artifact
		if role.Status != "accepted" || !prepared.GeneratedInventory.Sources["source"].Ref.SameExact(api.Ref) || prepared.GeneratedInventory.SourceContentDigest != api.Digest || len(scanFixtureRequests(h)) != 0 {
			t.Fatalf("accepted source or approval boundary changed: %+v", prepared)
		}
		key := "prepare:" + role.ExecutionID + ":api"
		path := "/v1/audits/" + draft.AuditID + "/artifacts?" + url.Values{"key": {key}, "revision": {*api.Ref.Revision}}.Encode()
		before, _ := download(t, h.client, h.baseURL+path)
		deleteASVSBacktraceRun(t, h.ctx, h.client, h.baseURL, role.RunID)
		after, _ := download(t, h.client, h.baseURL+path)
		if !bytes.Equal(before, after) {
			t.Fatal("source Run deletion changed accepted preparation bytes")
		}
		var reviews struct {
			Items []auditProgramReview `json:"items"`
		}
		auditProgramGET(t, h.client, h.baseURL+"/v1/audits/"+draft.AuditID+"/reviews", &reviews)
		if len(reviews.Items) != 1 {
			t.Fatalf("expected one scanner approval: %+v", reviews)
		}
		review := reviews.Items[0]
		scanAuditPOST(t, h, "/v1/audits/"+draft.AuditID+"/reviews/"+review.RequestID+"/decisions", map[string]string{"action": "approve", "rationale": "Only the controlled loopback target."}, review.Revision, http.StatusOK, nil)
		waitScanAudit(t, h, draft.AuditID, 2)
		item := scanAuditItem(t, h, draft.AuditID)
		if len(item.Attempts) != 1 || item.AcceptedResult == nil {
			t.Fatalf("scanner did not publish an accepted result: %+v", item)
		}
		requests := scanFixtureRequests(h)
		if len(requests) == 0 {
			t.Fatal("real SQLMap never contacted the controlled target")
		}
		for _, request := range requests {
			if request.method != "GET" || !strings.HasPrefix(request.uri, "/widgets/7?") || request.authorization != "Bearer local-fixture-token" {
				t.Fatalf("scan changed prepared request: method=%s uri=%s authorizationMatches=%t", request.method, request.uri, request.authorization == "Bearer local-fixture-token")
			}
		}
		var coverage struct {
			Items []struct {
				Details  auditstore.CoverageDetails `json:"details"`
				Coverage auditstore.Coverage        `json:"coverage"`
			} `json:"items"`
		}
		auditProgramGET(t, h.client, h.baseURL+"/v1/audits/"+draft.AuditID+"/coverage", &coverage)
		if len(coverage.Items) != 1 || coverage.Items[0].Details.Scan == nil || coverage.Items[0].Details.Scan.Source.Digest != api.Digest || coverage.Items[0].Coverage.Status != auditstore.CoverageInconclusive {
			t.Fatalf("exact generated scan provenance or coverage missing: %+v", coverage)
		}
	})
	t.Run("browser_preparation_approval_retention_and_failure", func(t *testing.T) {
		if h.browserAPIURL == "" {
			configureScanBrowser(t, h)
		}
		beforeStages := gateway.CompletedStages()
		gateway.blockNextRequest()
		removeFault := installScanFault(t, h, "composition_inventory", "artifact_bindings", "INSERT", `IF NEW.scope_kind = 'project' AND NEW.name LIKE 'initial-inventory%' THEN RAISE EXCEPTION USING ERRCODE = '40001', MESSAGE = 'controlled inventory publication hold'; END IF;`)
		token := randomHex(t, 24)
		control := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method != http.MethodPost || r.Header.Get("Authorization") != "Bearer "+token {
				w.WriteHeader(http.StatusForbidden)
				return
			}
			switch r.URL.Path {
			case "/release-preparation":
				gateway.releaseBlockedRequest()
			case "/accept-inventory":
				removeFault()
			default:
				w.WriteHeader(http.StatusNotFound)
				return
			}
			w.WriteHeader(http.StatusNoContent)
		}))
		t.Cleanup(control.Close)
		runCompositionBrowser(t, h, control.URL, token)
		if gateway.CompletedStages() != beforeStages+4 || len(gateway.Failures()) != 0 {
			t.Fatalf("unexpected real source pipeline: stages=%d failures=%v", gateway.CompletedStages(), gateway.Failures())
		}
	})
}

func compositionGatewayStages() []domainGatewayStage {
	stages := domainGatewayStages()[:4]
	for i := range stages[2].steps {
		step := &stages[2].steps[i]
		if step.tool == "load_openapi" {
			*step = toolGatewayStep("initialize_openapi", fixedArguments(map[string]any{"title": "Widget Service", "version": "1.0.0", "target_name": "openapi", "expected_revision": nil}))
		}
		if step.tool == "upsert_openapi_path" {
			original := step.arguments
			step.arguments = func(request map[string]any) (map[string]any, error) {
				arguments, err := original(request)
				if err != nil {
					return nil, err
				}
				operation := arguments["path_item"].(map[string]any)["get"].(map[string]any)
				operation["parameters"] = append(operation["parameters"].([]any), map[string]any{"name": "q", "in": "query", "required": true, "schema": map[string]any{"type": "string"}})
				operation["security"] = []any{map[string]any{"bearerAuth": []any{}}}
				return arguments, nil
			}
		}
	}
	security := toolGatewayStep("upsert_openapi_component", fixedArguments(map[string]any{
		"section": "securitySchemes", "name": "bearerAuth", "component": map[string]any{"type": "http", "scheme": "bearer"}, "evidence_files": []any{"app.py"},
	}))
	stages[2].steps = append(stages[2].steps[:5], append([]domainGatewayStep{security}, stages[2].steps[5:]...)...)
	return stages
}

func compositionSourceArchive(t *testing.T) []byte {
	t.Helper()
	original := projectSourceArchive(t)
	reader, err := zip.NewReader(bytes.NewReader(original), int64(len(original)))
	if err != nil {
		t.Fatal(err)
	}
	var result bytes.Buffer
	writer := zip.NewWriter(&result)
	for _, file := range reader.File {
		stream, err := file.Open()
		if err != nil {
			t.Fatal(err)
		}
		data, err := io.ReadAll(stream)
		stream.Close()
		if err != nil {
			t.Fatal(err)
		}
		if file.Name == "app.py" {
			data = []byte(strings.Replace(string(data), "    widget_id: str,\n", "    widget_id: str,\n    q: str,\n", 1))
		}
		header := file.FileHeader
		entry, err := writer.CreateHeader(&header)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := entry.Write(data); err != nil {
			t.Fatal(err)
		}
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	return result.Bytes()
}
