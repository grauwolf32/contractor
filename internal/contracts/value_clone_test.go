package contracts

import (
	"errors"
	"strings"
	"testing"
)

func TestArtifactRefValueMethods(t *testing.T) {
	one, two := "rev-1", "rev-2"
	exact := ArtifactRef{Namespace: "ns", Name: "name", Revision: &one}
	clone := exact.Clone()
	*clone.Revision = "changed"
	if *exact.Revision != "rev-1" {
		t.Fatal("Clone shares the Revision pointer")
	}

	same := ArtifactRef{Namespace: "ns", Name: "name", Revision: &[]string{"rev-1"}[0]}
	other := ArtifactRef{Namespace: "ns", Name: "name", Revision: &two}
	logical := ArtifactRef{Namespace: "ns", Name: "name"}
	if !exact.SameExact(same) || exact.SameExact(other) || logical.SameExact(logical) {
		t.Fatal("SameExact must match only equal pinned revisions")
	}
	if !exact.Equal(same) || exact.Equal(other) || exact.Equal(logical) || !logical.Equal(ArtifactRef{Namespace: "ns", Name: "name"}) {
		t.Fatal("Equal must compare optional revisions by value")
	}
	if exact.Key() == other.Key() || logical.Key() != (ArtifactRef{Namespace: "ns", Name: "name", Revision: new(string)}).Key() {
		t.Fatal("Key must separate revisions and key an absent revision as empty")
	}
}

func TestResolvedAgentTemplateCloneCopiesSkillRevisions(t *testing.T) {
	revision := "rev-1"
	source := ResolvedAgentTemplate{
		Toolsets: []ToolsetSelection{{Tools: []string{"read"}}},
		Skills:   []ArtifactRef{{Namespace: "skills", Name: "alpha", Revision: &revision}},
	}
	clone := source.Clone()
	*clone.Skills[0].Revision = "corrupted"
	clone.Toolsets[0].Tools[0] = "corrupted"
	if revision != "rev-1" || source.Toolsets[0].Tools[0] != "read" {
		t.Fatalf("cloned AgentTemplate aliases its source: %+v", source)
	}
}

func TestCheckPinnedSelection(t *testing.T) {
	value := map[string]string{"workflow": "copy@1"}
	if err := CheckPinnedSelection("", value); err != nil {
		t.Fatalf("empty expectation error = %v", err)
	}
	if err := CheckPinnedSelection("sha256:"+strings.Repeat("0", 64), value); !errors.Is(err, ErrPinnedSelectionChanged) {
		t.Fatalf("changed selection error = %v", err)
	}
	if err := CheckPinnedSelection("sha256:a4d3a5e1b3ae8e4e3b5a4d1c11b1ee4d8a8a0fd09e0c08cb8b8f6f0c7d3cd5f2", func() {}); err == nil {
		t.Fatal("unencodable selection was accepted")
	}
}
