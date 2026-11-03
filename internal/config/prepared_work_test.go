package config

import (
	"encoding/json"
	"path/filepath"
	"testing"

	"github.com/grauwolf32/contractor/internal/apierror"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestSharedPreparedWorkCatalogBindings(t *testing.T) {
	var fixture struct {
		Cases []struct {
			Name, Document, Context string
			Valid                   bool
		} `json:"cases"`
		LoaderCases []struct {
			Name, Document, Context string
			Valid                   bool
		} `json:"loader_cases"`
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "audit-common", "prepared-work.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	snapshot := mustLoad(t, copyCoreFixture(t, "audit-common-catalog"))
	for _, tc := range append(fixture.Cases, fixture.LoaderCases...) {
		t.Run(tc.Name, func(t *testing.T) {
			err := ValidatePreparedWork(snapshot, []byte(tc.Document), []byte(tc.Context))
			if (err == nil) != tc.Valid {
				t.Fatalf("valid=%v error=%v", tc.Valid, err)
			}
			if err != nil {
				classified, ok := apierror.Of(err)
				if !ok || classified.ErrorKind() != apierror.AuditDocumentInvalid {
					t.Fatalf("wrong classification: %v", err)
				}
			}
		})
	}
}

func TestSharedOpenAPIPreparationOperationStage(t *testing.T) {
	var fixture struct {
		Workflow string `json:"workflow"`
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "server-operation-openapi-work.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	root := copyCoreFixture(t)
	writeFile(t, filepath.Join(root, "workflows", "prepare.yaml"), []byte(fixture.Workflow))
	snapshot := mustLoad(t, root)
	workflow, err := snapshot.Workflow("openapi-prepare-work@1")
	if err != nil {
		t.Fatal(err)
	}
	if workflow.Stages[workflow.EntryStage].Operation == nil {
		t.Fatal("operation became a Planner")
	}
}

func TestOpenAPIPreparationWorkflowHasNoRuntimeExecutor(t *testing.T) {
	snapshot := mustLoad(t, copyCoreFixture(t, "openapi-prepared-work"))
	workflow, err := snapshot.Workflow("openapi-prepare-work@1")
	if err != nil {
		t.Fatal(err)
	}
	stage := workflow.Stages[workflow.EntryStage]
	if stage.Operation == nil || len(stage.Agents) != 0 || stage.ExecutionConfig.Planner != nil {
		t.Fatal("preparation created Runtime/Planner configuration")
	}
}
