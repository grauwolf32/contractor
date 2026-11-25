//go:build e2e

package e2e

import (
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/configload/configloadtest"
	"github.com/grauwolf32/contractor/internal/contracts"
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
	for _, ref := range []string{"audit_source_checker@3", "audit_wstg_http_verifier@1", "audit_openapi_operation_tracer@1", "audit_openapi_operation_observer@1", "audit_asvs_source_verifier@3", "audit_standard_source_verifier@3", "audit_risk_source_checker@3", "source_findings_reviewer@1"} {
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

// The shipped Audit definitions must hydrate source before any model call.
// An archive opener otherwise fails on valid project names such as App::Identity.
func TestAuditSourceWorkspaceCatalog(t *testing.T) {
	snapshot, _, err := configload.LoadUnionReadOnly(filepath.Join(repoRoot(t), "configs"), filepath.Join(t.TempDir(), "managed"))
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"audit-source-check", "audit-top10-source-risk", "audit-asvs-source-verification", "audit-standard-source-review"} {
		t.Run(name, func(t *testing.T) {
			wf, err := snapshot.Workflow(name + "@3")
			if err != nil {
				t.Fatal(err)
			}
			stage := wf.Stages[wf.EntryStage]
			workspace := stage.Context.Workspace
			if workspace == nil || workspace.Mode != "overlay" || workspace.State != nil || workspace.Export != nil || len(workspace.Sources) != 1 || workspace.Sources[0].Artifact != "source" || workspace.Sources[0].Target != "" {
				t.Fatalf("source must be hydrated before model invocation, retaining source capture: %+v", workspace)
			}
			source := stage.Context.Artifacts["source"]
			if source.Namespace != "inputs" || source.Name != "source" || !source.Required {
				t.Fatalf("wrong source binding: %+v", source)
			}
			for _, binding := range stage.Agents {
				wantSkills := []contracts.ArtifactRef{
					{Namespace: "skills", Name: "trace"},
					{Namespace: "skills", Name: "vuln-scan"},
				}
				if !slices.Equal(binding.Template.Skills, wantSkills) {
					t.Fatalf("unexpected passive source Skills: %+v", binding.Template.Skills)
				}
				hasFilesystem := false
				for _, selection := range binding.Template.Toolsets {
					if selection.Ref.ToolsetID == "source-analysis" {
						t.Fatal("model must not initialize a source archive")
					}
					if selection.Ref.ToolsetID == "filesystem" {
						hasFilesystem = true
						if !slices.Equal(selection.Tools, []string{"glob", "grep", "ls", "read_file"}) {
							t.Fatalf("unexpected filesystem tools: %+v", selection.Tools)
						}
					}
				}
				if !hasFilesystem || binding.Template.Ref.Version != "3" {
					t.Fatal("versioned source readers missing")
				}
				for _, text := range []string{stage.Instructions.Text, binding.Template.Instructions.Text} {
					if !strings.Contains(text, "Runtime has already prepared") || !strings.Contains(text, "workspace=true") {
						t.Fatal("source instructions omit preparation or exact workspace evidence")
					}
				}
			}
		})
	}
	for _, name := range []string{"source-checklist", "owasp-top10-2025-source-risk", "owasp-asvs-5-0-l1-source-pilot", "owasp-asvs-5-0-l1-source-review", "owasp-wstg-4-2-source-review", "owasp-wstg-4-2-fast-source-review", "hypothesis-research"} {
		profile, err := snapshot.AuditProfile(name + "@3")
		if err != nil {
			t.Fatal(err)
		}
		if !strings.HasSuffix(profile.Program.Workflows["check"].Ref, "@3") {
			t.Fatalf("profile %s retains an unprepared source checker", name)
		}
	}
}
