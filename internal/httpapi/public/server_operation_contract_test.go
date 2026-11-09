package public

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/getkin/kin-openapi/openapi3"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/auth"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/configtest"
	"github.com/grauwolf32/contractor/internal/contracts"
	publicapi "github.com/grauwolf32/contractor/internal/publicclient/generated"
	"github.com/grauwolf32/contractor/internal/runservice"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runstore/runstoretest"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
)

func publicOperationWorkflow(t *testing.T) resolved.ResolvedWorkflow {
	t.Helper()
	root := configtest.CopyWithPolicies(t, publicCatalogFixture)
	data, err := os.ReadFile("../../config/testdata/server-operations/workflows/join.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "workflows", "sitemap_join.yaml"), data, 0600); err != nil {
		t.Fatal(err)
	}
	catalog, _, err := configload.LoadUnionReadOnly(root, filepath.Join(t.TempDir(), "managed"))
	if err != nil {
		t.Fatal(err)
	}
	workflow, err := catalog.Workflow("sitemap-join@1")
	if err != nil {
		t.Fatal(err)
	}
	return workflow
}

func TestPublicServerOperationProjectionContracts(t *testing.T) {
	workflow := publicOperationWorkflow(t)
	body, err := json.Marshal(workflowResourceReadModel(workflow))
	if err != nil {
		t.Fatal(err)
	}
	var value map[string]any
	if err := json.Unmarshal(body, &value); err != nil {
		t.Fatal(err)
	}
	document := loadPublicOpenAPI(t)
	if err := document.Components.Schemas["WorkflowResource"].Value.VisitJSON(value, openapi3.EnableJSONSchema2020()); err != nil {
		t.Fatalf("operation projection: %s: %v", body, err)
	}
	var typed publicapi.WorkflowResource
	if err := json.Unmarshal(body, &typed); err != nil {
		t.Fatal(err)
	}
	stage := typed.Stages["join-api"]
	if stage.Operation == nil || stage.Planner != nil || stage.Instructions != nil || len(stage.Agents) != 0 {
		t.Fatalf("generated client invents a Planner: %+v", stage)
	}
	stageValue := value["stages"].(map[string]any)["join-api"].(map[string]any)
	for _, member := range []string{"planner", "instructions"} {
		stageValue[member] = nil
		if err := document.Components.Schemas["WorkflowStageContract"].Value.VisitJSON(stageValue, openapi3.EnableJSONSchema2020()); err == nil {
			t.Fatalf("public schema accepts operation and %s", member)
		}
		delete(stageValue, member)
	}
}

func TestPublicServerOperationFailOnSuccessContracts(t *testing.T) {
	workflow := publicOperationWorkflow(t)
	stage := workflow.Stages[workflow.EntryStage]
	stage.On.Succeeded = resolved.TransitionAction{Kind: resolved.TransitionFail}
	workflow.Stages[workflow.EntryStage] = stage
	if err := resolved.ValidateWorkflowGraph(workflow); err != nil {
		t.Fatal(err)
	}
	body, err := json.Marshal(workflowResourceReadModel(workflow))
	if err != nil {
		t.Fatal(err)
	}
	var value map[string]any
	if err := json.Unmarshal(body, &value); err != nil {
		t.Fatal(err)
	}
	document := loadPublicOpenAPI(t)
	if err := document.Components.Schemas["WorkflowResource"].Value.VisitJSON(value, openapi3.EnableJSONSchema2020()); err != nil {
		t.Fatalf("valid fail-on-success projection rejected: %v", err)
	}
	var typed publicapi.WorkflowResource
	if err := json.Unmarshal(body, &typed); err != nil {
		t.Fatal(err)
	}
	succeeded, err := typed.Stages[workflow.EntryStage].On.Succeeded.AsWorkflowFailTransition()
	if err != nil || succeeded.Kind != publicapi.WorkflowFailTransitionKindFail {
		t.Fatalf("generated fail transition: %+v, %v", succeeded, err)
	}
	for _, invalid := range []map[string]any{
		{"kind": "retry", "maxAttempts": 2, "then": map[string]any{"kind": "fail"}},
		{"kind": "escalate", "maxAttempts": 2, "executionConfig": map[string]any{"effective": map[string]any{"agents": map[string]any{}}}, "then": map[string]any{"kind": "fail"}},
		{"kind": "unknown"},
	} {
		if err := document.Components.Schemas["WorkflowSucceededTransition"].Value.VisitJSON(invalid, openapi3.EnableJSONSchema2020()); err == nil {
			t.Fatalf("unsupported successful transition accepted: %v", invalid)
		}
	}
}

func TestPostgresPublicOperationStageAttemptContract(t *testing.T) {
	ctx := t.Context()
	pool := isolatedPublicPool(t, ctx)
	workflow := publicOperationWorkflow(t)
	stage := workflow.Stages[workflow.EntryStage]
	snapshot, err := json.Marshal(workflow)
	if err != nil {
		t.Fatal(err)
	}
	stageJSON, err := json.Marshal(stage)
	if err != nil {
		t.Fatal(err)
	}
	storage := artifactpgtest.NewStorage()
	runs := runstore.NewPostgresStore(pool, storage)
	if _, err := runstoretest.CreateRun(ctx, runs, runstore.CreateRunParams{
		RunID: "run-operation", OwnerID: "user-1", WorkflowName: workflow.Ref.Name, WorkflowVersion: workflow.Ref.Version,
		WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: snapshot, RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot(), LaneClass: resolved.LaneClassTool,
	}); err != nil {
		t.Fatal(err)
	}
	for _, transition := range [][2]runstore.WorkflowRunState{{runstore.RunInitializing, runstore.RunPending}, {runstore.RunPending, runstore.RunRunning}} {
		if _, err := runs.TransitionRun(ctx, "run-operation", transition[0], transition[1], runstore.Reason{Code: "admitted"}); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := runs.CreateStageExecution(ctx, runstore.CreateStageExecutionParams{
		RunID: "run-operation", StageExecutionID: "stage-operation", StageName: workflow.EntryStage, Attempt: 1, LaneClass: resolved.LaneClassTool,
		StageSpecSchemaVersion: contracts.APIVersion, StageSpecSnapshot: stageJSON, StageContextSchemaVersion: contracts.APIVersion,
	}); err != nil {
		t.Fatal(err)
	}
	details, err := runservice.NewRunDetailReader(runservice.PostgresRunDetailTransaction(pool, storage, nil))
	if err != nil {
		t.Fatal(err)
	}
	handlers := &runHandlers{RunDependencies: RunDependencies{RunDetails: details}}
	request := httptest.NewRequest(http.MethodGet, "/v1/runs/run-operation", nil)
	request.SetPathValue("runId", "run-operation")
	request = request.WithContext(auth.WithPrincipal(ctx, auth.Principal{UserID: "user-1"}))
	response := httptest.NewRecorder()
	serve(handlers.getRun)(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("read persisted operation attempt: %d %s", response.Code, response.Body.String())
	}
	var body map[string]any
	if err := json.Unmarshal(response.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	attempt := body["attempts"].([]any)[0].(map[string]any)
	config := attempt["executionConfig"].(map[string]any)
	agents, ok := config["agents"].(map[string]any)
	if !ok || len(agents) != 0 {
		t.Fatalf("operation agents must be an empty object: %v", config["agents"])
	}
	document := loadPublicOpenAPI(t)
	if err := document.Components.Schemas["RunStatus"].Value.VisitJSON(body, openapi3.EnableJSONSchema2020()); err != nil {
		t.Fatalf("persisted operation Run rejected by public contract: %v", err)
	}
	for _, variant := range []string{"missing", "null"} {
		delete(config, "agents")
		if variant == "null" {
			config["agents"] = nil
		}
		if err := document.Components.Schemas["StageExecutionConfig"].Value.VisitJSON(config, openapi3.EnableJSONSchema2020()); err == nil {
			t.Fatalf("%s agents accepted", variant)
		}
	}
}
