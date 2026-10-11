package artifactpolicy

import (
	"encoding/json"
	"os"
	"testing"
)

func TestReservedMemoryBindingExcludesPurposeNamespaces(t *testing.T) {
	for _, test := range []struct {
		namespace string
		name      string
		want      bool
	}{
		{"analysis", "memory.note", true},
		{"analysis", "report", false},
		{"inputs", "memory.note", false},
		{"outputs", "memory.note", false},
		{"skills", "memory.note", false},
	} {
		if got := IsReservedMemoryBinding(test.namespace, test.name); got != test.want {
			t.Errorf("IsReservedMemoryBinding(%q, %q) = %t, want %t", test.namespace, test.name, got, test.want)
		}
	}
}

func TestPurposeReservedNamespaceIsClosed(t *testing.T) {
	for _, namespace := range []string{"inputs", "outputs", "skills", FindingProposalNamespace} {
		if !IsPurposeReservedNamespace(namespace) {
			t.Errorf("namespace %q is not purpose-reserved", namespace)
		}
	}
	for _, namespace := range []string{"", "analysis", "Skills", "skills-extra"} {
		if IsPurposeReservedNamespace(namespace) {
			t.Errorf("namespace %q is unexpectedly purpose-reserved", namespace)
		}
	}
}

func TestModelHiddenBindingMatchesSharedCases(t *testing.T) {
	data, err := os.ReadFile("../../api/testdata/v1alpha1/artifact-list-cases.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases struct {
		Bindings []struct {
			Namespace string `json:"namespace"`
			Name      string `json:"name"`
			Hidden    bool   `json:"hidden"`
		} `json:"bindings"`
	}
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	if len(cases.Bindings) == 0 {
		t.Fatal("shared model visibility cases are empty")
	}
	for _, test := range cases.Bindings {
		if got := IsModelHiddenBinding(test.Namespace, test.Name); got != test.Hidden {
			t.Errorf("IsModelHiddenBinding(%q, %q) = %t, want %t", test.Namespace, test.Name, got, test.Hidden)
		}
	}
}
