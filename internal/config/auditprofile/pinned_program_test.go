package auditprofile_test

import (
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestSharedPinnedProgramRecoversOnlyItsExactClosure(t *testing.T) {
	catalog := loadCatalog(t, copyCatalog(t, "audit-common-catalog"))
	var fixture struct{ Cases []struct{ Document string } }
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "audit-common", "programs.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	pinned, err := auditprofile.PinProgram(resolved.Selector{ID: "common", Version: "1"}, []byte(fixture.Cases[0].Document), catalog.Workflow)
	if err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(pinned)
	if err != nil {
		t.Fatal(err)
	}
	recovered, err := auditprofile.DecodePinnedProgram(data)
	if err != nil || recovered.Ref != pinned.Ref {
		t.Fatalf("recovery: %+v %v", recovered.Ref, err)
	}
	// Recovery has no catalog callback. A mutated catalog closure cannot
	// change any of the captured Workflow maps or slices.
	workflow := recovered.Workflows["inspect"]
	workflow.Inputs["source"] = resolved.ArtifactSlot{Required: false, MediaTypes: []string{"application/zip"}}
	if pinned.Workflows["inspect"].Inputs["source"].MediaTypes[0] != "text/plain" {
		t.Fatal("pin shares decoded input maps")
	}
	for _, mutate := range []func(map[string]any){
		func(v map[string]any) {
			v["program"].(map[string]any)["initial_work"].([]any)[0].(map[string]any)["objective"] = "Different work"
		},
		func(v map[string]any) { delete(v["workflows"].(map[string]any), "inspect") },
		func(v map[string]any) {
			v["workflows"].(map[string]any)["other"] = v["workflows"].(map[string]any)["inspect"]
		},
		func(v map[string]any) { v["inventory"] = map[string]any{} },
		func(v map[string]any) { v["ref"].(map[string]any)["digest"] = "sha256:bad" },
	} {
		var changed map[string]any
		if err := json.Unmarshal(data, &changed); err != nil {
			t.Fatal(err)
		}
		mutate(changed)
		bytes, _ := json.Marshal(changed)
		if _, err := auditprofile.DecodePinnedProgram(bytes); err == nil {
			t.Fatalf("accepted changed closure: %s", bytes)
		}
	}
	if _, err := auditprofile.PinProgram(resolved.Selector{ID: "common", Version: "1"}, []byte(fixture.Cases[0].Document), func(ref string) (resolved.ResolvedWorkflow, error) {
		workflow, err := catalog.Workflow(ref)
		workflow.Ref.Name = "different"
		return workflow, err
	}); err == nil {
		t.Fatal("accepted mismatched selected Workflow identity")
	}
}
