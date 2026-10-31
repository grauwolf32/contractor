package auditprofile_test

import (
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/configload"
)

func TestRetainedDependencyAuthoringAndPersistedValidation(t *testing.T) {
	for _, test := range []struct {
		name, mapping string
		valid         bool
	}{
		{"named", "{source: retained-dependency, name: exact_source}", true},
		{"missing-name", "{source: retained-dependency}", false},
		{"role-forbidden", "{source: retained-dependency, name: exact_source, role: producer}", false},
		{"unknown-field", "{source: retained-dependency, name: exact_source, latest: true}", false},
		{"invalid-name", "{source: retained-dependency, name: 'bad / name'}", false},
	} {
		t.Run(test.name, func(t *testing.T) {
			root := copyCatalog(t, "audit-profile-catalog")
			manifest := strings.Replace(templateAuditProfileYAML(t, "retained-check"), "{source: audit-input, name: source}", test.mapping, 1)
			writeAuditProfile(t, root, "retained-check", manifest)
			snapshot, _, err := configload.LoadUnionReadOnly(root, filepath.Join(t.TempDir(), "managed"))
			if (err == nil) != test.valid {
				t.Fatalf("load valid=%v: %v", test.valid, err)
			}
			if !test.valid {
				return
			}
			profile, err := snapshot.AuditProfile("retained-check@1")
			if err != nil {
				t.Fatal(err)
			}
			if profile.Workflows["check"].Inputs["source"].Source != auditprofile.AuditInputFromRetainedDependency {
				t.Fatal("declared dependency mapping changed")
			}
			raw, err := json.Marshal(profile)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := auditprofile.DecodeResolvedAuditProfileSnapshot(raw); err != nil {
				t.Fatal(err)
			}
			for _, kind := range []auditprofile.AuditWorkflowRoleKind{auditprofile.AuditWorkflowDiscovery, auditprofile.AuditWorkflowAssessment} {
				candidate := profile.Clone()
				binding := candidate.Workflows["check"]
				binding.Kind = kind
				candidate.Workflows["check"] = binding
				redigest(t, &candidate)
				encoded, _ := json.Marshal(candidate)
				if _, err := auditprofile.DecodeResolvedAuditProfileSnapshot(encoded); err == nil {
					t.Fatalf("%s consumer accepted retained dependency", kind)
				}
			}
			candidate := profile.Clone()
			candidate.Inventory.Source = &auditprofile.AuditWorkflowInputMapping{Source: auditprofile.AuditInputFromRetainedDependency, Name: "exact_source"}
			redigest(t, &candidate)
			encoded, _ := json.Marshal(candidate)
			if _, err := auditprofile.DecodeResolvedAuditProfileSnapshot(encoded); err == nil {
				t.Fatal("inventory accepted item dependency mapping")
			}
		})
	}
}
