//go:build e2e

package e2e

import (
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/configload/configloadtest"
)

func TestAuditProgramsE2EConfigurationLoads(t *testing.T) {
	root := stageE2EConfiguration(t, filepath.Join(repoRoot(t), "configs"), filepath.Join(t.TempDir(), "configs"), "http://127.0.0.1:1/v1")
	snapshot, _, err := configload.LoadUnionReadOnly(root, filepath.Join(t.TempDir(), "managed"))
	if err != nil {
		t.Fatal(err)
	}
	for _, selector := range commonAuditProfileSelectors {
		if _, err := snapshot.AuditProfile(selector); err != nil {
			t.Fatal(err)
		}
	}
}

func TestAuditProgramCatalogRemovalConfigurationClosure(t *testing.T) {
	root := stageE2EConfiguration(t, filepath.Join(repoRoot(t), "configs"), filepath.Join(t.TempDir(), "configs"), "http://127.0.0.1:1/v1")
	before, _, err := configload.LoadUnionReadOnly(root, filepath.Join(t.TempDir(), "before-managed"))
	if err != nil {
		t.Fatal(err)
	}
	removeAuditProgramAuthoringEntries(t, root)
	after, _, err := configload.LoadUnionReadOnly(root, filepath.Join(t.TempDir(), "after-managed"))
	if err != nil {
		t.Fatalf("catalog replacement must retain a complete authoring closure: %v", err)
	}
	removed := 0
	for _, profile := range before.AuditProfiles() {
		selector := profile.Ref.Name + "@" + profile.Ref.Version
		current, err := after.AuditProfile(selector)
		if removedAuditProgramProfiles[selector] {
			removed++
			if err == nil {
				t.Fatalf("removed Audit profile still resolves: %s", selector)
			}
		} else {
			if err != nil || !reflect.DeepEqual(current, profile) {
				t.Fatalf("unrelated Audit profile must remain unchanged: %s: %v", selector, err)
			}
		}
	}
	if removed != 4 || len(after.AuditProfiles()) != len(before.AuditProfiles())-removed {
		t.Fatalf("unexpected removed profile closure: removed=%d before=%d after=%d", removed, len(before.AuditProfiles()), len(after.AuditProfiles()))
	}
}

func TestAuditInvestigationInstructionCatalog(t *testing.T) {
	snapshot, _, err := configload.LoadUnionReadOnly(filepath.Join(repoRoot(t), "configs"), filepath.Join(t.TempDir(), "managed"))
	if err != nil {
		t.Fatal(err)
	}
	for _, ref := range []string{"audit_source_checker@1", "audit_wstg_http_verifier@1", "audit_openapi_operation_tracer@1", "audit_openapi_operation_observer@1", "audit_asvs_source_verifier@1", "audit_standard_source_verifier@1", "audit_risk_source_checker@1", "source_findings_reviewer@1"} {
		template, err := configloadtest.AgentTemplate(snapshot, ref)
		if err != nil {
			t.Fatal(err)
		}
		finding := false
		for _, selection := range template.Toolsets {
			for _, tool := range selection.Tools {
				finding = finding || tool == "finding"
			}
		}
		if !finding {
			t.Fatalf("%s cannot publish an incidental evidenced problem", ref)
		}
		text := template.Instructions.Text
		for _, guidance := range []string{"A first finding or one", "publish before final result submission", "what remains unexamined or uncertain", "Zero findings is valid"} {
			if !strings.Contains(text, guidance) {
				t.Fatalf("%s omitted investigation guidance %q", ref, guidance)
			}
		}
		if strings.Contains(text, "expected_revision") {
			t.Fatalf("%s asks for a model revision", ref)
		}
	}
}
