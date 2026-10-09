//go:build e2e

package e2e

import (
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/config"
)

func TestAuditProgramsE2EConfigurationLoads(t *testing.T) {
	root := stageE2EConfiguration(t, filepath.Join(repoRoot(t), "configs"), filepath.Join(t.TempDir(), "configs"), "http://127.0.0.1:1/v1")
	installOrdinaryFindingFixture(t, root, "audit_asvs_source_verifier", "audit_asvs_source_verification", "fixture-ordinary-asvs")
	snapshot, err := config.Load(root, config.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	templates := map[string]string{
		"checklist":    "audit_source_checker@1",
		"openapi":      "audit_openapi_operation_observer@1",
		"top10":        "audit_risk_source_checker@1",
		"asvs":         "audit_asvs_source_verifier@1",
		"ordinary-run": "fixture-ordinary-asvs@1",
	}
	for _, stage := range auditProgramGatewayStages() {
		t.Run(stage.name, func(t *testing.T) {
			prefix, _, _ := strings.Cut(stage.name, "/")
			template, err := snapshot.AgentTemplate(templates[prefix])
			if err != nil {
				t.Fatal(err)
			}
			var tools []string
			for _, selection := range template.Toolsets {
				tools = append(tools, selection.Tools...)
			}
			if len(template.Skills) > 0 {
				tools = append(tools, "list_skills", "load_skill", "load_skill_resource")
			}
			slices.Sort(tools)
			want := slices.Clone(stage.tools)
			slices.Sort(want)
			if !slices.Equal(tools, want) {
				t.Fatalf("scripted Gateway tools = %v; catalog tools = %v", want, tools)
			}
		})
	}
}
