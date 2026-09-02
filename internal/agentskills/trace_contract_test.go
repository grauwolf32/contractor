package agentskills

import (
	"strings"
	"testing"
)

func TestTraceSkillNamesOnlyTheStructuredAnnotationContract(t *testing.T) {
	_, pkg, err := PackageDirectory("../../configs/skills/trace")
	if err != nil {
		t.Fatal(err)
	}
	if pkg.Digest != "sha256:2b71e3f49e9da6aee8e9724c60fe8dc22987907ac99d262c1d0068d4936b3ebc" {
		t.Fatalf("trace Skill digest = %s", pkg.Digest)
	}
	members := make(map[string]string, len(pkg.Resources)+1)
	for _, member := range pkg.Members() {
		members[member.Path] = string(member.Data())
	}
	root := members["SKILL.md"]
	annotations := members["references/annotations.md"]
	for _, required := range []string{
		"annotate_trace", "annotate_validate", "annotate_sink",
		"definition_line=0", "changed=false", "changed_paths", "diff",
		"rollback_changes", "Haskell, Lua", "Elixir",
	} {
		if !strings.Contains(root+annotations, required) {
			t.Errorf("trace annotation contract omits %q", required)
		}
	}
	for _, forbidden := range []string{
		"AgentTemplate", "AllocationSpec", "run_skill_script", "`restore`",
	} {
		if strings.Contains(root+annotations, forbidden) {
			t.Errorf("trace annotation contract contains forbidden token %q", forbidden)
		}
	}
}
