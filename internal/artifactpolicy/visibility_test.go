package artifactpolicy

import "testing"

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
