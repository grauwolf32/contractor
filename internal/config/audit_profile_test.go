package config

import (
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/config/auditprofile"
)

func validAuditProfileYAML(t *testing.T, name string) string {
	t.Helper()
	return strings.Replace(string(readFile(t, "testdata/audit-profile-template.yaml")), "PROFILE_NAME", name, 1)
}
func writeAuditProfile(t *testing.T, root, name, manifest string) {
	t.Helper()
	writeFile(t, filepath.Join(root, "audit-profiles", name+".yaml"), []byte(manifest))
}
func copyAuditProfileConfigTree(t *testing.T) string { t.Helper(); return copyCoreFixture(t) }
func TestAuditProfileLoadsResolvedWorkflowAndReturnsDeepCopies(t *testing.T) {
	root := copyAuditProfileConfigTree(t)
	for _, name := range []string{"z-review", "a-review"} {
		writeAuditProfile(t, root, name, validAuditProfileYAML(t, name))
	}
	snapshot := mustLoad(t, root)
	profiles := snapshot.AuditProfiles()
	if len(profiles) != 2 || profiles[0].Ref.Name != "a-review" || profiles[1].Ref.Name != "z-review" {
		t.Fatalf("profiles=%+v", profiles)
	}
	profile, err := snapshot.AuditProfile("a-review@1")
	if err != nil {
		t.Fatal(err)
	}
	profile.Program.Inputs["source"].MediaTypes[0] = "bad/type"
	profile.Program.InitialWork[0].Objective = "changed"
	binding := profile.Program.Workflows["inspect"]
	binding.Outputs["material"] = "changed"
	profile.Program.Workflows["inspect"] = binding
	profile.Workflows["inspect"].Inputs["source"].MediaTypes[0] = "bad/type"
	again, err := snapshot.AuditProfile("a-review@1")
	if err != nil {
		t.Fatal(err)
	}
	if again.Program.Inputs["source"].MediaTypes[0] != "text/plain" || again.Program.InitialWork[0].Objective != "Inspect selected source" || again.Program.Workflows["inspect"].Outputs["material"] != "result" || again.Workflows["inspect"].Inputs["source"].MediaTypes[0] != "text/plain" {
		t.Fatal("snapshot shares mutable program content")
	}
	for _, selector := range []string{"a-review", "missing@1"} {
		if _, err := snapshot.AuditProfile(selector); err == nil {
			t.Fatal("accepted", selector)
		}
	}
	data, err := json.Marshal(again)
	if err != nil {
		t.Fatal(err)
	}
	recovered, err := auditprofile.DecodePinnedProgram(data)
	if err != nil || recovered.Ref != again.Ref {
		t.Fatal("lost exact closure", err)
	}
}
func TestAuditProfileRejectsOldContractAndUnknownOrNullMembers(t *testing.T) {
	for name, change := range map[string]func(map[string]any){
		"legacy": func(s map[string]any) { s["inventory"] = map[string]any{} }, "unknown": func(s map[string]any) { s["Unknown"] = true }, "null": func(s map[string]any) { s["policy"] = nil }, "old enum": func(s map[string]any) { s["policy"].(map[string]any)["active_operations"] = "approval-required" }, "missing required input": func(s map[string]any) {
			s["workflows"].(map[string]any)["inspect"].(map[string]any)["inputs"] = map[string]any{}
		},
	} {
		t.Run(name, func(t *testing.T) {
			root := copyAuditProfileConfigTree(t)
			var value map[string]any
			if err := json.Unmarshal([]byte(validAuditProfileYAML(t, "invalid")), &value); err != nil {
				t.Fatal(err)
			}
			change(value["spec"].(map[string]any))
			data, _ := json.Marshal(value)
			writeAuditProfile(t, root, "invalid", string(data))
			if _, err := LoadUnion(root, filepath.Join(t.TempDir(), "managed")); err == nil {
				t.Fatal("invalid program accepted")
			}
		})
	}
}
func TestAuditProfileProgramAndWorkflowClosureBothAffectDigest(t *testing.T) {
	root := copyAuditProfileConfigTree(t)
	raw := validAuditProfileYAML(t, "digest")
	writeAuditProfile(t, root, "digest", raw)
	before, err := mustLoad(t, root).AuditProfile("digest@1")
	if err != nil {
		t.Fatal(err)
	}
	writeAuditProfile(t, root, "digest", strings.Replace(raw, "Inspect selected source", "Inspect another aspect", 1))
	after, err := mustLoad(t, root).AuditProfile("digest@1")
	if err != nil || after.Ref.Digest == before.Ref.Digest {
		t.Fatal("changed objective retained digest", err)
	}
	pin := before
	pin.Program.InitialWork[0].Objective = "forged"
	data, _ := json.Marshal(pin)
	if _, err := auditprofile.DecodePinnedProgram(data); err == nil {
		t.Fatal("persisted program accepted drift")
	}
}
