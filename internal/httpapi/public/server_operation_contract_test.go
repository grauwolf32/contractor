package public

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/getkin/kin-openapi/openapi3"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/configtest"
	publicapi "github.com/grauwolf32/contractor/internal/publicclient/generated"
)

func TestPublicServerOperationProjectionContracts(t *testing.T) {
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
