//go:build e2e

package e2e

import (
	"bytes"
	"encoding/json"
	"net/http"
	"reflect"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/findingintake"
)

func TestFindingsCollectionsRetainOrdinaryAndAuditReceipts(t *testing.T) {
	ordinaryStage := findingsProducerStage("ordinary-with-hypothesis")
	original := ordinaryStage.steps[4].arguments
	ordinaryStage.steps[4].arguments = func(request map[string]any) (map[string]any, error) {
		value, err := original(request)
		value["hypothesis"] = "Database parsing may change when the route parameter contains quotes."
		return value, err
	}
	// Repeat the same tool call within the same invocation through the real intake.
	ordinaryStage.steps = append(ordinaryStage.steps[:5:5], append([]domainGatewayStep{ordinaryStage.steps[4]}, ordinaryStage.steps[5:]...)...)
	h := startFindingsHarness(t, []domainGatewayStage{findingsProducerStage("operation-a"),
		findingsProducerStage("operation-b"), ordinaryStage, findingsReaderStage(3, 1)})
	project := createProjectResource(t, h.client, h.baseURL)
	sourcePayload := auditProgramZip(t, map[string][]byte{"app.py": []byte(strings.Replace(findingsSource,
		"@app.get(\"/items/{item_id}\")", "@app.get(\"/items/{item_id}\")\n@app.post(\"/items/{item_id}\")", 1))})
	source := uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "sources", "shared-function", "application/zip", sourcePayload)
	getStart := strings.Index(findingsOpenAPI, "    get:")
	schema := findingsOpenAPI + strings.ReplaceAll(findingsOpenAPI[getStart:], "get:", "post:")
	schema = strings.Replace(schema, "    post:\n      operationId: getItem", "    post:\n      operationId: postItem", 1)
	openAPI := uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "openapi", "shared-function", "application/yaml", []byte(schema))
	audit := runAuditProgram(t, h.ctx, h.server, h.runtimeProcess, h.gateway, h.client, h.baseURL, project.ProjectID,
		"openapi-operation-trace@3", map[string]artifactRef{"source": source, "openapi": openAPI}, 2, 2, 0, "")
	var items struct {
		Items []auditProgramItem `json:"items"`
	}
	auditProgramGET(t, h.client, h.baseURL+"/v1/audits/"+audit.AuditID+"/items?limit=100", &items)
	var auditReceipts []findingintake.Receipt
	for _, item := range items.Items {
		auditReceipts = append(auditReceipts, findingsReceipts(t, h, item.Attempts[0].RunID)...)
	}
	t.Run("shared-function-preserves-operations", func(t *testing.T) {
		if len(items.Items) != 2 || len(auditReceipts) != 2 || auditReceipts[0].ReceiptID == auditReceipts[1].ReceiptID ||
			auditReceipts[0].Document.Subject != auditReceipts[1].Document.Subject || auditReceipts[0].Evidence[0].Digest != auditReceipts[1].Evidence[0].Digest {
			t.Fatalf("shared function lost operation receipts: %+v", auditReceipts)
		}
	})
	ordinaryRun := runOrdinaryFindingProducer(t, h, project.ProjectID, source, sourcePayload, openAPI, []byte(schema))
	ordinary := findingsReceipts(t, h, ordinaryRun)
	t.Run("ordinary-hypothesis-and-intake-replay", func(t *testing.T) {
		if len(ordinary) != 1 || ordinary[0].Origin.Audit != nil || ordinary[0].Document.Hypothesis == "" {
			t.Fatalf("ordinary receipt = %+v", ordinary)
		}
		assertFindingsResultReceipt(t, h, ordinaryRun, ordinary[0])
	})
	sources := []findingintake.CollectionSelection{
		{Kind: "run", ID: ordinaryRun, ReceiptIDs: []string{ordinary[0].ReceiptID}, Findings: []findingintake.CollectionFindingSelection{}},
		{Kind: "audit", ID: audit.AuditID, ReceiptIDs: []string{auditReceipts[0].ReceiptID, auditReceipts[1].ReceiptID}, Findings: []findingintake.CollectionFindingSelection{}},
	}
	publication := publishFindingsCollection(t, h, "retained-selection", sources, false)
	t.Run("snapshot-survives-source-deletion", func(t *testing.T) {
		deleteASVSBacktraceRun(t, h.ctx, h.client, h.baseURL, ordinaryRun)
		for _, receipt := range auditReceipts {
			deleteASVSBacktraceRun(t, h.ctx, h.client, h.baseURL, receipt.Origin.RunID)
		}
		replay := publishFindingsCollection(t, h, "retained-selection", sources, true)
		if !reflect.DeepEqual(replay.Artifact, publication.Artifact) || replay.SnapshotAt != publication.SnapshotAt || replay.EntryCount != 3 {
			t.Fatalf("replay drift: %+v %+v", publication, replay)
		}
		// Retained Audit refs support a new snapshot even after child Run deletion.
		retained := publishFindingsCollection(t, h, "retained-audit-only", sources[1:], false)
		if retained.EntryCount != 2 {
			t.Fatalf("Audit holds lost receipts: %+v", retained)
		}
	})
	t.Run("changed-selection-conflicts", func(t *testing.T) {
		body, _ := json.Marshal(findingintake.PublishCollectionRequest{ClientKey: "retained-selection", Sources: sources[1:]})
		request, _ := http.NewRequest(http.MethodPost, h.baseURL+"/v1/finding-collections", bytes.NewReader(body))
		request.Header.Set("Authorization", "Bearer "+publicToken)
		request.Header.Set("Content-Type", "application/json")
		response := do(t, h.client, request, http.StatusConflict)
		response.Body.Close()
	})
	t.Run("pagination-and-exact-evidence", func(t *testing.T) {
		readerRun := runFindingsReader(t, h, "retained-reader", publication.Artifact.Ref)
		report, _ := download(t, h.client, h.baseURL+"/v1/runs/"+readerRun+"/outputs/report")
		for _, receipt := range append(auditReceipts, ordinary...) {
			if !strings.Contains(string(report), receipt.ReceiptID) {
				t.Fatalf("report lost %s", receipt.ReceiptID)
			}
		}
		if len(findingsReceipts(t, h, readerRun)) != 0 {
			t.Fatal("reader acquired proposal emission")
		}
	})
	if h.gateway.CompletedStages() != 4 || len(h.gateway.Failures()) != 0 {
		t.Fatalf("gateway stages = %d: %v", h.gateway.CompletedStages(), h.gateway.Failures())
	}
}

func runOrdinaryFindingProducer(t *testing.T, h *findingsHarness, projectID string, source artifactRef, sourcePayload []byte, openAPI artifactRef, openAPIPayload []byte) string {
	t.Helper()
	inventory, err := auditdomain.BuildOpenAPIInventory(openAPIPayload, "application/yaml", auditdomain.InventoryOptions{
		Round: 1, WorkflowRole: "trace", SourceInputName: "openapi", SourceRef: exactContractRef(t, openAPI), ApprovalRequirement: auditdomain.ApprovalNone,
	})
	if err != nil || len(inventory.Tasks) == 0 {
		t.Fatalf("ordinary inventory: %v", err)
	}
	task := inventory.Tasks[0]
	taskRef := uploadProjectScopeArtifact(t, h.client, h.baseURL, projectID, "fixtures", "ordinary-task", auditdomain.PackageMediaType, task.Package)
	item := inventory.ExecutionManifest.Items[0]
	item.Ordinal = 0
	exactTask := exactContractRef(t, taskRef)
	item.TaskRef = &exactTask
	item.Inputs = []auditdomain.ExactInput{{Name: "source", Ref: exactContractRef(t, source), Digest: auditProgramDigest(sourcePayload)}}
	manifest := auditdomain.ExecutionManifest{Schema: auditdomain.ExecutionManifestSchema, Items: []auditdomain.ExecutionItem{item}}
	if err := auditdomain.ValidateDispatchExecutionManifest(manifest); err != nil {
		t.Fatal(err)
	}
	encoded, err := auditdomain.EncodeExecutionManifest(manifest)
	if err != nil {
		t.Fatal(err)
	}
	manifestRef := uploadProjectScopeArtifact(t, h.client, h.baseURL, projectID, "fixtures", "ordinary-manifest", "application/json", encoded)
	body, _ := json.Marshal(map[string]any{"workflow": "audit-openapi-operation-trace@2", "artifacts": map[string]artifactRef{"task": taskRef, "execution_manifest": manifestRef, "source": source}})
	runID := postProjectRun(t, h.client, h.baseURL, projectID, "ordinary-findings", body, false)
	waitForFindingsRun(t, h, runID, "succeeded")
	return runID
}
