package auditprograms

import (
	"bytes"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestFastWSTGPreservesSelectedObjectivesAndEvidencePolicies(t *testing.T) {
	wantIDs := []string{
		"WSTG-v42-ATHN-02", "WSTG-v42-ATHN-04", "WSTG-v42-ATHN-09",
		"WSTG-v42-ATHZ-01", "WSTG-v42-ATHZ-02", "WSTG-v42-ATHZ-03", "WSTG-v42-ATHZ-04",
		"WSTG-v42-BUSL-09", "WSTG-v42-CONF-04",
		"WSTG-v42-INPV-01", "WSTG-v42-INPV-02", "WSTG-v42-INPV-05",
		"WSTG-v42-INPV-12", "WSTG-v42-INPV-18", "WSTG-v42-INPV-19", "WSTG-v42-SESS-05",
	}
	snapshot, err := config.Load(configRoot(), config.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	for _, variant := range []struct {
		profile, fullProfile, edition, fullEdition string
		approval                                   auditdomain.ApprovalRequirement
	}{
		{"owasp-wstg-4-2-fast-source-review@1", "owasp-wstg-4-2-source-review@1", "4.2-fast-source.1", "4.2", auditdomain.ApprovalNone},
		{"owasp-wstg-4-2-fast-active-http@1", "owasp-wstg-4-2-active-http@1", "4.2-fast-http.1", "4.2-http.1", auditdomain.ApprovalActiveCheck},
	} {
		t.Run(variant.profile, func(t *testing.T) {
			load := func(edition string) *auditstandards.Package {
				t.Helper()
				_, pkg, err := auditstandards.PackageDirectory(filepath.Join(configRoot(), "audit-standards", "owasp-wstg-"+edition))
				if err != nil {
					t.Fatal(err)
				}
				return pkg
			}
			fast, full := load(variant.edition), load(variant.fullEdition)
			if fast.Reference().Version != variant.edition || fast.Reference().Scheme != "owasp-wstg" ||
				fast.Document.Standard.Source != full.Document.Standard.Source ||
				fast.Document.Standard.License.ID != full.Document.Standard.License.ID ||
				fast.Document.Standard.License.Disclosure != full.Document.Standard.License.Disclosure ||
				!reflect.DeepEqual(fast.Document.EvidenceContracts, full.Document.EvidenceContracts) ||
				len(fast.Document.Entries) != 16 || len(full.Document.Entries) != 94 ||
				len(fast.Document.Mappings) != 16 || len(full.Document.Mappings) != 94 {
				t.Fatal("Fast edition lost provenance, changed evidence rules or replaced the full edition")
			}
			entries := map[string]auditstandards.Entry{}
			mappings := map[string]auditstandards.Mapping{}
			for _, entry := range full.Document.Entries {
				entries[entry.ID] = entry
			}
			for _, mapping := range full.Document.Mappings {
				mappings[mapping.Key] = mapping
			}
			for i, entry := range fast.Document.Entries {
				if entry.ID != wantIDs[i] || !reflect.DeepEqual(entry, entries[entry.ID]) ||
					!reflect.DeepEqual(fast.Document.Mappings[i], mappings[entry.ID]) {
					t.Fatalf("Fast check has a different objective or evidence policy: %s", entry.ID)
				}
			}
			profile, err := snapshot.AuditProfile(variant.profile)
			if err != nil {
				t.Fatal(err)
			}
			original, err := snapshot.AuditProfile(variant.fullProfile)
			if err != nil {
				t.Fatal(err)
			}
			if !auditservice.ProfileCompatibility(profile).ServerCompatible ||
				len(profile.Standards) != 1 || profile.Standards[0].Version != variant.edition ||
				profile.Mode != original.Mode || profile.Interaction != original.Interaction ||
				!reflect.DeepEqual(profile.Inputs, original.Inputs) || !reflect.DeepEqual(profile.Workflows, original.Workflows) ||
				profile.Execution.MaxRounds != 1 || profile.Execution.BatchSize != 1 ||
				profile.Execution.MaxItemsPerRound != 16 || profile.Execution.MaxItemsTotal != 16 ||
				profile.Execution.MaxSubmittedRuns != 16 || profile.Execution.MaxItemRunAttempts != 1 ||
				profile.Execution.DeadlineSeconds != 14400 {
				t.Fatal("Fast profile lost its fixed work budget or changed verification/approval behavior")
			}
			revision := "fast-wstg-r1"
			options := auditdomain.InventoryOptions{
				Round: 1, WorkflowRole: "check", SourceInputName: "standard",
				SourceRef:           contracts.ArtifactRef{Namespace: "audit-example", Name: "fast-wstg", Revision: &revision},
				ApprovalRequirement: variant.approval,
			}
			first, err := auditdomain.BuildStandardMappingInventory(*fast, options)
			if err != nil {
				t.Fatal(err)
			}
			second, err := auditdomain.BuildStandardMappingInventory(*fast, options)
			if err != nil || auditdomain.ValidateInventory(first) != nil || len(first.Tasks) != 16 ||
				!bytes.Equal(first.CanonicalInventory, second.CanonicalInventory) {
				t.Fatalf("Fast inventory is not deterministic and complete: %v", err)
			}
			for i, task := range first.Tasks {
				if task.Document.ItemKey != wantIDs[i] || task.Document.Standard.Version != variant.edition ||
					task.Item.ApprovalRequirement != variant.approval ||
					!reflect.DeepEqual(task.Document.Standard.EntryIDs, []string{wantIDs[i]}) {
					t.Fatalf("Fast inventory changed scope or approval: %s", task.Document.ItemKey)
				}
			}
		})
	}
}
