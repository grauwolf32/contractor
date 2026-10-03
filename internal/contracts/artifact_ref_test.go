package contracts

import "testing"

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
