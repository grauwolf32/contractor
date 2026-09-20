package auditprograms

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestExpandedASVSIncludesEveryUpstreamLevelOneRequirement(t *testing.T) {
	t.Parallel()
	_, pkg, err := auditstandards.PackageDirectory(filepath.Join(configRoot(), "audit-standards", "owasp-asvs-5.0.0-l1-source.1"))
	if err != nil {
		t.Fatal(err)
	}
	var upstream map[string]string
	data, err := os.ReadFile(filepath.Join("testdata", "asvs-5.0.0-level1.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &upstream); err != nil {
		t.Fatal(err)
	}
	if len(upstream) != 70 || len(pkg.Document.Entries) != len(upstream) {
		t.Fatalf("ASVS Level 1 denominator: %d entries, %d upstream", len(pkg.Document.Entries), len(upstream))
	}
	for _, entry := range pkg.Document.Entries {
		if entry.Kind != "requirement" || entry.Level != "1" || upstream[entry.ID] != entry.Statement {
			t.Errorf("ASVS requirement differs from pinned upstream: %s", entry.ID)
		}
	}
	if pkg.Reference() != (auditstandards.Reference{Scheme: "owasp-asvs", Version: "5.0.0-l1-source.1"}) ||
		pkg.Document.Standard.Source.Revision != asvsRevision {
		t.Fatalf("ASVS package lost its distinct edition or upstream provenance: %+v", pkg.Document.Standard)
	}

	// The original immutable package and profile remain the five-item pilot.
	_, legacy := loadASVSPackage(t)
	if len(legacy.Document.Entries) != 5 || legacy.Digest == pkg.Digest {
		t.Fatal("expanded ASVS replaced the original published package")
	}
}

func TestWSTGUsesActiveVersionQualifiedScenariosWithoutRetiredAliases(t *testing.T) {
	t.Parallel()
	_, pkg, err := auditstandards.PackageDirectory(filepath.Join(configRoot(), "audit-standards", "owasp-wstg-4.2"))
	if err != nil {
		t.Fatal(err)
	}
	if pkg.Reference() != (auditstandards.Reference{Scheme: "owasp-wstg", Version: "4.2"}) ||
		pkg.Document.Standard.Source.Revision != "dd33419e10edb22b78d89325a6c2aad9f184e3a2" {
		t.Fatal("WSTG source release is not pinned")
	}
	categories := map[string]int{}
	seen := map[string]bool{}
	for _, entry := range pkg.Document.Entries {
		parts := strings.Split(entry.ID, "-")
		if len(parts) != 4 || parts[0] != "WSTG" || parts[1] != "v42" || seen[entry.ID] || entry.Statement == "" {
			t.Fatalf("invalid or duplicate WSTG scenario: %+v", entry)
		}
		seen[entry.ID] = true
		categories[parts[2]]++
		if entry.ID == "WSTG-v42-INPV-13" && entry.Title != "Testing for Format String Injection" {
			t.Fatal("removed buffer overflow stub replaced format string injection")
		}
	}
	want := map[string]int{"INFO": 9, "CONF": 11, "IDNT": 5, "ATHN": 10, "ATHZ": 4, "SESS": 9, "INPV": 18, "ERRH": 1, "CRYP": 4, "BUSL": 9, "CLNT": 13, "APIT": 1}
	if !reflect.DeepEqual(categories, want) {
		t.Fatalf("WSTG active scenario categories = %v", categories)
	}
	for _, retired := range []string{"WSTG-v42-INFO-09", "WSTG-v42-INPV-03", "WSTG-v42-ERRH-02"} {
		if seen[retired] {
			t.Errorf("merged alias remains a separate check: %s", retired)
		}
	}
	for _, mapping := range pkg.Document.Mappings {
		if strings.Contains(mapping.Objective, "/stable/") || !strings.Contains(mapping.Objective, "/v42/") {
			t.Errorf("WSTG scenario does not link to the exact release: %s", mapping.Key)
		}
	}
}

func TestExpandedSourcePresetsBuildExactRunnableInventories(t *testing.T) {
	t.Parallel()
	snapshot, err := config.Load(configRoot(), config.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		profile, directory string
		count              int
		manual             []string
	}{
		{"owasp-asvs-5-0-l1-source-review@1", "owasp-asvs-5.0.0-l1-source.1", 70,
			[]string{"v5.0.0-15.1.1", "v5.0.0-2.1.1", "v5.0.0-6.1.1", "v5.0.0-8.1.1"}},
		{"owasp-wstg-4-2-source-review@1", "owasp-wstg-4.2", 94, []string{}},
	} {
		t.Run(test.profile, func(t *testing.T) {
			profile, err := snapshot.AuditProfile(test.profile)
			if err != nil {
				t.Fatal(err)
			}
			payload, pkg, err := auditstandards.PackageDirectory(filepath.Join(configRoot(), "audit-standards", test.directory))
			if err != nil {
				t.Fatal(err)
			}
			if !auditservice.ProfileCompatibility(profile).ServerCompatible || len(profile.Standards) != 1 ||
				profile.Standards[0].Scheme != pkg.Reference().Scheme || profile.Standards[0].Version != pkg.Reference().Version ||
				profile.Interaction.ActiveChecks != config.AuditActiveChecksProhibited ||
				profile.Interaction.FindingConfirmation != config.AuditFindingHumanRequired ||
				profile.Execution.MaxItemsTotal != test.count || profile.Execution.MaxItemsPerRound != test.count ||
				profile.Execution.MaxSubmittedRuns < test.count*profile.Execution.MaxItemRunAttempts {
				t.Fatalf("preset is not a bounded compatible source review: %+v", profile)
			}
			if profile.Workflows["check"].Workflow.Ref.Name != "audit-standard-source-review" ||
				pkg.Document.Standard.License.ID != "CC-BY-SA-4.0" || pkg.Document.Standard.License.Disclosure != auditstandards.DisclosureFull {
				t.Fatal("preset lost its verifier or licensed disclosure")
			}
			revision := "source-review-standard-r1"
			options := auditdomain.InventoryOptions{
				Round: 1, WorkflowRole: "check", SourceInputName: "standard",
				SourceRef:           contracts.ArtifactRef{Namespace: "audit-example", Name: "standard", Revision: &revision},
				ApprovalRequirement: auditdomain.ApprovalNone,
			}
			if selected := profile.Inventory.StandardSelection; selected != nil {
				options.StandardSelection = &auditdomain.StandardSelection{Scope: selected.Scope, Levels: selected.Levels, EntryIDs: selected.EntryIDs}
			}
			first, err := auditdomain.BuildStandardMappingInventory(*pkg, options)
			if err != nil {
				t.Fatal(err)
			}
			second, err := auditdomain.BuildStandardMappingInventory(*pkg, options)
			if err != nil || auditdomain.ValidateInventory(first) != nil || len(first.Tasks) != test.count ||
				!bytes.Equal(first.CanonicalInventory, second.CanonicalInventory) {
				t.Fatalf("inventory is not deterministic and complete: %v, %d tasks", err, len(first.Tasks))
			}
			if _, err := auditstandards.Validate(payload, pkg.Reference()); err != nil {
				t.Fatal(err)
			}
			manual := []string{}
			seen := map[string]bool{}
			for _, task := range first.Tasks {
				id := task.Document.ItemKey
				if seen[id] || task.Document.Standard == nil || task.Document.Standard.MappingKey != id ||
					!reflect.DeepEqual(task.Document.Standard.EntryIDs, []string{id}) {
					t.Fatalf("check is duplicated or lost its causal origin: %s", id)
				}
				seen[id] = true
				if task.Item.ApprovalRequirement == auditdomain.ApprovalHumanReview {
					manual = append(manual, id)
				}
				if pkg.Reference().Scheme == "owasp-wstg" || strings.HasPrefix(id, "v5.0.0-12.") || id == "v5.0.0-13.4.1" {
					for _, assessment := range task.Document.Standard.EvidenceContract.Assessments {
						if assessment == "satisfied" {
							t.Errorf("live-dependent check can be marked satisfied from source alone: %s", id)
						}
					}
				}
			}
			sort.Strings(manual)
			if !reflect.DeepEqual(manual, test.manual) {
				t.Fatalf("documentary applicability reviews = %v, want %v", manual, test.manual)
			}
		})
	}
}

func TestWSTGHTTPPresetHasSameScenariosAndDistinctExecutionEvidence(t *testing.T) {
	root := configRoot()
	_, source, err := auditstandards.PackageDirectory(filepath.Join(root, "audit-standards", "owasp-wstg-4.2"))
	if err != nil {
		t.Fatal(err)
	}
	_, active, err := auditstandards.PackageDirectory(filepath.Join(root, "audit-standards", "owasp-wstg-4.2-http.1"))
	if err != nil {
		t.Fatal(err)
	}
	if len(active.Document.Entries) != 94 || active.Digest == source.Digest ||
		active.Document.Standard.Source != source.Document.Standard.Source {
		t.Fatal("WSTG HTTP edition must preserve upstream provenance with separate evidence rules")
	}
	for i, entry := range active.Document.Entries {
		original := source.Document.Entries[i]
		if entry.ID != original.ID || entry.Statement != original.Statement || entry.Title != original.Title ||
			!reflect.DeepEqual(entry.AllowedMethods, []string{"active-test"}) || active.Document.Mappings[i].Method != "active-test" {
			t.Fatalf("active WSTG changed its scenario or lost the HTTP method: %s", entry.ID)
		}
	}
	snapshot, err := config.Load(root, config.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	profile, err := snapshot.AuditProfile("owasp-wstg-4-2-active-http@1")
	if err != nil {
		t.Fatal(err)
	}
	binding := profile.Workflows["check"]
	if !auditservice.ProfileCompatibility(profile).ServerCompatible ||
		profile.Interaction.ActiveChecks != config.AuditActiveChecksApprovalRequired ||
		profile.Interaction.FindingConfirmation != config.AuditFindingHumanRequired ||
		profile.Standards[0].Version != active.Reference().Version ||
		len(profile.Inputs) != 1 || !profile.Inputs["context"].Required ||
		profile.Execution.MaxItemsTotal != 94 || profile.Execution.MaxItemsPerRound != 94 ||
		profile.Execution.MaxSubmittedRuns != 94 || profile.Execution.MaxItemRunAttempts != 1 ||
		binding.Workflow.Ref.Name != "audit-wstg-active-http" ||
		binding.Parameters["target"].Name != "target" ||
		binding.Parameters["authorization_scope"].Name != "authorizationScope" {
		t.Fatalf("HTTP preset lost its prerequisites or execution limits: %+v", profile)
	}
	revision := "wstg-http-r1"
	inventory, err := auditdomain.BuildStandardMappingInventory(*active, auditdomain.InventoryOptions{
		Round: 1, WorkflowRole: "check", SourceInputName: "standard",
		SourceRef:           contracts.ArtifactRef{Namespace: "audit-example", Name: "wstg-http", Revision: &revision},
		ApprovalRequirement: auditdomain.ApprovalActiveCheck,
	})
	if err != nil || len(inventory.Tasks) != 94 || auditdomain.ValidateInventory(inventory) != nil {
		t.Fatalf("HTTP inventory invalid: %v", err)
	}
	for _, task := range inventory.Tasks {
		if task.Item.ApprovalRequirement != auditdomain.ApprovalActiveCheck ||
			!reflect.DeepEqual(task.Document.Standard.EvidenceContract.Assessments, []string{"blocked", "inconclusive", "not-tested", "satisfied", "violated"}) {
			t.Fatalf("HTTP scenario lost its gate or assessment policy: %s", task.Document.ItemKey)
		}
	}
}
