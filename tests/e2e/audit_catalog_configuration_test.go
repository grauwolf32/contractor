//go:build e2e

package e2e

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/grauwolf32/contractor/internal/config"
)

func TestAuditProgramCatalogReplacementPreservesSharedInstructions(t *testing.T) {
	root := filepath.Join(t.TempDir(), "configs")
	if err := os.CopyFS(root, os.DirFS(filepath.Join(repoRoot(t), "configs"))); err != nil {
		t.Fatal(err)
	}
	before, err := config.Load(root, config.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	removeAuditProgramAuthoringEntries(t, root)
	after, err := config.Load(root, config.MVPDescriptors())
	if err != nil {
		t.Fatalf("replacement catalog must remain valid for Server restart: %v", err)
	}
	for _, program := range []struct {
		workflow, template, profile, instruction, standard string
	}{
		{"audit-asvs-source-verification", "audit_asvs_source_verifier", "owasp-asvs-5-0-l1-source-review", "audit-asvs-source-verifier-worker.md", "owasp-asvs-5.0.0"},
		{"audit-top10-source-risk", "audit_risk_source_checker", "owasp-top10-2025-source-risk", "audit-risk-source-checker-worker.md", "owasp-web-top10-2025"},
	} {
		t.Run(program.workflow, func(t *testing.T) {
			if _, err := before.Workflow(program.workflow + "@1"); err != nil {
				t.Fatal(err)
			}
			if _, err := after.Workflow(program.workflow + "@1"); err == nil {
				t.Fatal("original Workflow remains available")
			}
			if _, err := after.AgentTemplate(program.template + "@1"); err == nil {
				t.Fatal("original AgentTemplate remains available")
			}
			if _, err := after.AuditProfile(program.profile + "@1"); err == nil {
				t.Fatal("original AuditProfile remains available")
			}
			if _, err := os.Stat(filepath.Join(root, "audit-standards", program.standard)); !os.IsNotExist(err) {
				t.Fatalf("original standard authoring directory remains: %v", err)
			}
			original, err := before.Workflow(program.workflow + "@3")
			if err != nil {
				t.Fatal(err)
			}
			retained, err := after.Workflow(program.workflow + "@3")
			if err != nil || !reflect.DeepEqual(original, retained) {
				t.Fatalf("surviving Workflow changed during catalog replacement: %v", err)
			}
			if _, err := after.AuditProfile(program.profile + "@2"); err != nil {
				t.Fatalf("surviving AuditProfile unavailable: %v", err)
			}
			instructionRef := "instructions/" + program.instruction
			want, err := before.Instructions(instructionRef)
			if err != nil {
				t.Fatal(err)
			}
			got, err := after.Instructions(instructionRef)
			if err != nil || got != want {
				t.Fatalf("shared exact instructions changed: %v", err)
			}
		})
	}
}
