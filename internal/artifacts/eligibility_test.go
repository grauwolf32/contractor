package artifacts

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestGenericReplacementAvailabilityUsesNamespaceAdmission(t *testing.T) {
	for _, test := range []struct {
		name      string
		scope     Scope
		namespace string
		current   bool
		frozen    bool
		want      bool
	}{
		{"ordinary-user", Scope{kind: ScopeUser, id: "owner"}, "files", true, false, true},
		{"ordinary-project", Scope{kind: ScopeProject, id: "project"}, "files", true, false, true},
		{"historical", Scope{kind: ScopeUser, id: "owner"}, "files", false, false, false},
		{"frozen", Scope{kind: ScopeProject, id: "project"}, "files", true, true, false},
		{"managed-project", Scope{kind: ScopeProject, id: "project"}, "sitemap", true, false, false},
		{"managed-audit", Scope{kind: ScopeProject, id: "project"}, "audit-results", true, false, false},
		{"managed-user", Scope{kind: ScopeUser, id: "owner"}, "audit-standards", true, false, false},
		{"run-input-working-copy", Scope{kind: ScopeRun, id: "run"}, "inputs", true, false, true},
		{"run-output", Scope{kind: ScopeRun, id: "run"}, "outputs", true, false, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			store, err := NewService(nil).Scoped(test.scope)
			if err != nil {
				t.Fatal(err)
			}
			metadata := Metadata{Ref: ArtifactRef{Namespace: test.namespace, Name: "value"}, Current: test.current, Frozen: test.frozen}
			if got := store.CanReplace(metadata); got != test.want {
				t.Fatalf("CanReplace=%t want %t", got, test.want)
			}
			data, err := json.Marshal(metadata)
			if err != nil || strings.Contains(string(data), "canUploadNewVersion") {
				t.Fatalf("public-only availability leaked into storage/private metadata: %s %v", data, err)
			}
		})
	}
}
