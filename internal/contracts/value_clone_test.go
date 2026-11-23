package contracts

import (
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

// Each Clone of a value that holds refs, maps or a revision pointer must not
// alias its source: callers outside the hub rely on them instead of copies.
func TestWireValueClonesDoNotAliasTheirSource(t *testing.T) {
	revision := "r1"
	ref := ArtifactRef{Namespace: "skills", Name: "review", Revision: &revision}

	request := StageContentRequest{
		Parameters: map[string]string{"mode": "strict"}, Artifacts: map[string]ArtifactRef{"source": ref},
		ResultArtifacts: map[string]ArtifactRef{"report": {Namespace: "builder", Name: "report"}},
	}
	requestCopy := request.Clone()
	requestCopy.Parameters["mode"] = "changed"
	*requestCopy.Artifacts["source"].Revision = "changed"
	delete(requestCopy.ResultArtifacts, "report")

	exact := ExactArtifact{Ref: ref, Digest: "sha256:" + strings.Repeat("0", 64)}
	exactCopy := CloneExactArtifacts(map[string]ExactArtifact{"input": exact})["input"]
	*exactCopy.Ref.Revision = "changed"

	if request.Parameters["mode"] != "strict" || len(request.ResultArtifacts) != 1 || revision != "r1" {
		t.Fatalf("a clone aliases its source: request %+v, revision %q", request, revision)
	}
	if CloneExactArtifacts(nil) == nil {
		t.Fatal("CloneExactArtifacts never returns nil")
	}
}
