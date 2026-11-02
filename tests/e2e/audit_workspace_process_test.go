//go:build e2e

package e2e

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestAuditWorkspaceAcrossProductionProcesses(t *testing.T) {
	gateway := newDomainGateway(llmGatewayToken)
	gateway.stages = []domainGatewayStage{workspaceGatewayStage("seed", "source"), workspaceGatewayStage("consumer", "previous")}
	t.Cleanup(gateway.close)
	h := startConfiguredScanStackWithBudget(t, 5*time.Minute, "", func(h *scanProcessHarness, root string) {
		stageE2EConfiguration(t, filepath.Join(h.repositoryRoot, "configs"), root, gateway.URL())
		h.gatewayToken = llmGatewayToken
		source := filepath.Join(h.repositoryRoot, "tests/e2e/testdata/audit-workspace")
		for _, folder := range []string{"workflows", "agent-templates", "instructions"} {
			entries, err := os.ReadDir(filepath.Join(source, folder))
			if err != nil {
				t.Fatal(err)
			}
			for _, entry := range entries {
				data, err := os.ReadFile(filepath.Join(source, folder, entry.Name()))
				if err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(filepath.Join(root, folder, entry.Name()), data, 0600); err != nil {
					t.Fatal(err)
				}
			}
		}
		base, err := os.ReadFile(filepath.Join(root, "audit-profiles/source_checklist.yaml"))
		if err != nil {
			t.Fatal(err)
		}
		profile := strings.Replace(string(base), "name: source-checklist", "name: workspace-fixture", 1)
		profile = strings.Replace(profile, "  workflows:\n", `  workflows:
    seed:
      kind: prepare
      ref: workspace-fixture@1
      maxRunAttempts: 1
      inputs:
        source: {source: audit-input, name: checklist}
      parameters: {}
      outputs: {material: material}
    consumer:
      kind: prepare
      ref: workspace-fixture@1
      maxRunAttempts: 1
      inputs:
        source: {source: audit-input, name: checklist}
        previous: {source: prepare-output, role: seed, name: material}
      parameters: {}
      outputs: {material: material}
`, 1)
		if err := os.WriteFile(filepath.Join(root, "audit-profiles/workspace.yaml"), []byte(profile), 0600); err != nil {
			t.Fatal(err)
		}
	})
	var project projectResourceResponse
	scanAuditPOST(t, h, "/v1/projects", map[string]string{"kind": "project", "name": "Exact Audit workspace", "activeChecks": "prohibited"}, 0, http.StatusCreated, &project)
	checklist := auditdomain.ChecklistDocument{Schema: auditdomain.ChecklistSchema, Items: []auditdomain.ChecklistEntry{{Key: "manual", Version: "1", Statement: "Controlled material", Applicability: "manual", AllowedMethods: []string{"code-review"}, RequiredEvidence: []string{"code-review"}, ReviewPolicy: "manual"}}}
	payload, _ := json.Marshal(checklist)
	inputs := map[string]artifactRef{
		"source":    uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "fixture", "source", "application/zip", compositionSourceArchive(t)),
		"checklist": uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "fixture", "checklist", "application/json", payload),
	}
	var draft compositionAuditWire
	scanAuditPOST(t, h, "/v1/projects/"+project.ProjectID+"/audits", map[string]any{"profile": map[string]string{"name": "workspace-fixture", "version": "1"}, "inputs": inputs, "scope": map[string]string{"objective": "Transfer one selected material through ordinary Runs."}}, 0, http.StatusCreated, &draft)
	scanAuditPOST(t, h, "/v1/audits/"+draft.AuditID+"/start", nil, draft.Revision, http.StatusOK, nil)
	var current compositionAuditWire
	waitScanCondition(t, h, "two accepted material outputs", func() bool {
		if auditProgramTryGET(h.ctx, h.client, h.baseURL+"/v1/audits/"+draft.AuditID, &current) != nil {
			return false
		}
		if current.State == "failed" {
			t.Fatalf("Audit failed: %+v gateway=%v", current, gateway.Failures())
		}
		return current.Preparation != nil && current.Preparation.Roles["consumer"].Status == "accepted"
	})
	source := current.Preparation.Roles["seed"]
	material := source.Outputs["material"].Artifact
	request := auditservice.MaterialPublicationRequest{AuditID: draft.AuditID, Source: material.Ref, Target: contracts.ArtifactRef{Namespace: "materials", Name: "selected"}}
	published := workspacePublicationPOST(t, h, project.ProjectID, "publish", request, nil, http.StatusCreated)
	t.Run("handoff", func(t *testing.T) {
		var rows, versions int
		if err := h.pool.QueryRow(h.ctx, `SELECT count(*),count(DISTINCT version_id) FROM artifact_binding_revisions
WHERE (scope_kind='audit' AND scope_id=$1 AND namespace=$2 AND name=$3)
 OR (scope_kind='run' AND scope_id=$4 AND namespace='inputs' AND name='previous')
 OR (scope_kind='run' AND scope_id=$5 AND namespace='outputs' AND name='material')`, draft.AuditID, material.Ref.Namespace, material.Ref.Name, current.Preparation.Roles["consumer"].RunID, source.RunID).Scan(&rows, &versions); err != nil || rows != 3 || versions != 1 {
			t.Fatalf("actual three scopes share version: rows=%d versions=%d %v", rows, versions, err)
		}
		if gateway.CompletedStages() != 2 || len(gateway.Failures()) != 0 {
			t.Fatalf("ordinary workflows: %d %v", gateway.CompletedStages(), gateway.Failures())
		}
		deleteASVSBacktraceRun(t, h.ctx, h.client, h.baseURL, source.RunID)
		var data []byte
		waitScanCondition(t, h, "retained material after source purge", func() bool {
			target := fmt.Sprintf("%s/v1/audits/%s/artifacts?key=prepare:%s:material&revision=%s", h.baseURL, draft.AuditID, source.ExecutionID, *material.Ref.Revision)
			req, err := http.NewRequestWithContext(h.ctx, http.MethodGet, target, nil)
			if err != nil {
				t.Fatal(err)
			}
			req.Header.Set("Authorization", "Bearer "+publicToken)
			response, err := h.client.Do(req)
			if err != nil {
				t.Fatal(err)
			}
			defer response.Body.Close()
			// Child deletion can advance the Audit while ReadArtifact checks its
			// snapshot. Retry only that classified optimistic-read conflict.
			if response.StatusCode == http.StatusConflict {
				var failure struct {
					Code      string `json:"code"`
					Retryable bool   `json:"retryable"`
				}
				if err := json.NewDecoder(response.Body).Decode(&failure); err != nil || failure.Code != "conflict" || !failure.Retryable {
					t.Fatalf("unexpected material read conflict: %+v %v", failure, err)
				}
				return false
			}
			if response.StatusCode != http.StatusOK {
				t.Fatalf("material read: %d %s", response.StatusCode, copyBounded(response.Body))
			}
			data, err = io.ReadAll(io.LimitReader(response.Body, 1<<20))
			if err != nil {
				t.Fatal(err)
			}
			return true
		})
		if !bytes.Equal(data, payload) {
			t.Fatal("source Run purge changed retained bytes")
		}
	})
	t.Run("replay", func(t *testing.T) {
		h.restartServer()
		replay := workspacePublicationPOST(t, h, project.ProjectID, "publish", request, nil, http.StatusOK)
		if !replay.Artifact.SameExact(published.Artifact) {
			t.Fatal("lost-response replay created a revision")
		}
	})
	t.Run("cas", func(t *testing.T) {
		workspacePublicationPOST(t, h, project.ProjectID, "collision", request, nil, http.StatusConflict)
		first := published.Artifact
		published = workspacePublicationPOST(t, h, project.ProjectID, "replace", request, first.Revision, http.StatusOK)
		workspacePublicationPOST(t, h, project.ProjectID, "stale", request, first.Revision, http.StatusConflict)
	})
	t.Run("authority", func(t *testing.T) {
		invalid := request
		invalid.AuditID = "foreign-audit"
		workspacePublicationPOST(t, h, project.ProjectID, "foreign", invalid, nil, http.StatusNotFound)
		invalid = request
		missing := "missing"
		invalid.Source.Revision = &missing
		workspacePublicationPOST(t, h, project.ProjectID, "missing", invalid, nil, http.StatusNotFound)
	})
	t.Run("publication", func(t *testing.T) {
		// Delete every remaining child only after its terminal collection/release.
		for _, role := range current.Preparation.Roles {
			if role.RunID != source.RunID {
				deleteASVSBacktraceRun(t, h.ctx, h.client, h.baseURL, role.RunID)
			}
		}
		auditProgramGET(t, h.client, h.baseURL+"/v1/audits/"+draft.AuditID, &current)
		req, _ := http.NewRequestWithContext(h.ctx, http.MethodDelete, h.baseURL+"/v1/audits/"+draft.AuditID, nil)
		req.Header.Set("Authorization", "Bearer "+publicToken)
		req.Header.Set("If-Match", fmt.Sprintf("\"%d\"", current.Revision))
		req.Header.Set("Idempotency-Key", "delete-workspace")
		response := do(t, h.client, req, http.StatusAccepted)
		response.Body.Close()
		waitScanCondition(t, h, "source Audit purge", func() bool {
			var n int
			return h.pool.QueryRow(h.ctx, `SELECT count(*) FROM audits WHERE audit_id=$1`, draft.AuditID).Scan(&n) == nil && n == 0
		})
		replay := workspacePublicationPOST(t, h, project.ProjectID, "publish", request, nil, http.StatusOK)
		if replay.Artifact.Revision == nil {
			t.Fatal("publication receipt disappeared with source")
		}
		data, _ := download(t, h.client, fmt.Sprintf("%s/v1/projects/%s/artifacts/%s/%s?revision=%s", h.baseURL, project.ProjectID, published.Artifact.Namespace, published.Artifact.Name, *published.Artifact.Revision))
		if !bytes.Equal(data, payload) {
			t.Fatal("Project publication lost independent bytes")
		}
		var edges int
		if err := h.pool.QueryRow(h.ctx, `SELECT count(*) FROM artifact_lineage WHERE target_scope_kind='project' AND target_scope_id=$1 AND source_scope_kind='audit' AND source_scope_id=$2`, project.ProjectID, draft.AuditID).Scan(&edges); err != nil || edges == 0 {
			t.Fatalf("historical lineage: %d %v", edges, err)
		}
	})
}

func workspacePublicationPOST(t *testing.T, h *scanProcessHarness, project, key string, request auditservice.MaterialPublicationRequest, expected *string, status int) auditservice.MaterialPublication {
	t.Helper()
	data, _ := json.Marshal(request)
	req, _ := http.NewRequestWithContext(h.ctx, http.MethodPost, h.baseURL+"/v1/projects/"+project+"/audit-material-publications", bytes.NewReader(data))
	req.Header.Set("Authorization", "Bearer "+publicToken)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Idempotency-Key", key)
	if expected != nil {
		req.Header.Set("If-Match", fmt.Sprintf("\"%s\"", *expected))
	}
	response := do(t, h.client, req, status)
	defer response.Body.Close()
	var result auditservice.MaterialPublication
	if status < 300 {
		decodeAuditProgramResponse(t, response, &result)
	}
	return result
}

func workspaceGatewayStage(name, input string) domainGatewayStage {
	return domainGatewayStage{name: name, tools: []string{"read_artifact", "write_artifact"}, steps: []domainGatewayStep{
		toolGatewayStep("read_artifact", fixedArguments(map[string]any{"namespace": "inputs", "name": input, "revision": nil})),
		toolGatewayStep("write_artifact", func(request map[string]any) (map[string]any, error) {
			data, ok := lastStringValue(request, "dataBase64")
			if !ok {
				return nil, errors.New("selected material was not read")
			}
			return map[string]any{"namespace": "worker", "name": "material", "media_type": "application/json", "data_base64": data, "expected_revision": nil}, nil
		}),
		finalGatewayStep("Selected exact material copied", map[string]domainArtifactBinding{"material": {namespace: "worker", name: "material"}}),
	}}
}
