package auditprograms

import (
	"bytes"
	"path/filepath"
	"reflect"
	"sort"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
)

const top10Revision = "66ebc4798d2ca72973967a20264bdeb70dcf0a13"

var top10Categories = []string{
	"A01:2025", "A02:2025", "A03:2025", "A04:2025", "A05:2025",
	"A06:2025", "A07:2025", "A08:2025", "A09:2025", "A10:2025",
}

func TestTop10PackageHasExactLicensedCategoryDenominator(t *testing.T) {
	t.Parallel()

	payload, pkg := loadTop10Package(t)
	again, second, err := auditstandards.PackageDirectory(top10PackageDirectory())
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(payload, again) || pkg.Digest != second.Digest {
		t.Fatal("Top 10 package bytes are not deterministic")
	}
	metadata := pkg.Document.Standard
	if metadata.Scheme != "owasp-web-top10" || metadata.Version != "2025" ||
		metadata.Source.Revision != top10Revision || metadata.License.ID != "CC-BY-SA-4.0" ||
		metadata.License.Disclosure != auditstandards.DisclosureFull {
		t.Fatalf("Top 10 package provenance = %+v", metadata)
	}
	entryIDs := make([]string, len(pkg.Document.Entries))
	for index, entry := range pkg.Document.Entries {
		entryIDs[index] = entry.ID
		if entry.Kind != "risk" || entry.Title == "" {
			t.Fatalf("Top 10 entry is not a bounded risk category: %+v", entry)
		}
	}
	mappingKeys := make([]string, len(pkg.Document.Mappings))
	for index, mapping := range pkg.Document.Mappings {
		mappingKeys[index] = mapping.Key
		if !reflect.DeepEqual(mapping.EntryIDs, []string{mapping.Key}) ||
			mapping.WorkflowRole != "check" || mapping.Method != "source-analysis" {
			t.Fatalf("Top 10 mapping is not exact: %+v", mapping)
		}
	}
	if !reflect.DeepEqual(entryIDs, top10Categories) || !reflect.DeepEqual(mappingKeys, top10Categories) {
		t.Fatalf("Top 10 denominator differs: entries=%v mappings=%v", entryIDs, mappingKeys)
	}
}

func TestTop10InventoryRetainsMappingsReviewAndEvidenceContracts(t *testing.T) {
	t.Parallel()

	_, pkg := loadTop10Package(t)
	revision := "retained-standard-r1"
	options := auditdomain.InventoryOptions{
		Round: 1, WorkflowRole: "check", SourceInputName: "standard",
		SourceRef: contracts.ArtifactRef{
			Namespace: "audit-example", Name: "standard-top10", Revision: &revision,
		},
		ApprovalRequirement: auditdomain.ApprovalNone,
	}
	first, err := auditdomain.BuildStandardMappingInventory(*pkg, options)
	if err != nil {
		t.Fatal(err)
	}
	second, err := auditdomain.BuildStandardMappingInventory(*pkg, options)
	if err != nil {
		t.Fatal(err)
	}
	if auditdomain.ValidateInventory(first) != nil || len(first.Tasks) != 10 ||
		!bytes.Equal(first.CanonicalInventory, second.CanonicalInventory) ||
		first.CanonicalInventoryDigest != second.CanonicalInventoryDigest {
		t.Fatalf("Top 10 inventory is not exact and deterministic: %+v", first)
	}
	manual := []string{}
	for index, generated := range first.Tasks {
		task := generated.Document
		if task.Standard == nil || task.Checklist == nil ||
			task.ItemKey != top10Categories[index] || task.Standard.MappingKey != task.ItemKey ||
			!reflect.DeepEqual(task.Standard.EntryIDs, []string{task.ItemKey}) ||
			task.Standard.EvidenceContract.ID != "bounded-source-risk" ||
			!reflect.DeepEqual(task.Checklist.RequiredEvidence, []string{"observation"}) {
			t.Fatalf("Top 10 task lost its exact contract: %+v", task)
		}
		if generated.Item.ApprovalRequirement == auditdomain.ApprovalHumanReview {
			manual = append(manual, task.ItemKey)
		}
	}
	if !reflect.DeepEqual(manual, []string{"A03:2025", "A06:2025"}) {
		t.Fatalf("Top 10 manual applicability gates = %v", manual)
	}
}

func TestTop10ProfileUsesOrdinaryCompatibleRunBoundary(t *testing.T) {
	t.Parallel()

	snapshot, err := config.Load(configRoot(), config.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	profile, err := snapshot.AuditProfile("owasp-top10-2025-source-risk@1")
	if err != nil {
		t.Fatal(err)
	}
	compatibility := auditservice.ProfileCompatibility(profile)
	if !compatibility.ServerCompatible || profile.Mode != config.AuditModeRiskAssessment ||
		profile.Inventory.Implementation != "standard-mappings@1" ||
		profile.Inventory.SourceInput != "" || profile.Execution.BatchSize != 1 ||
		profile.Interaction.FindingConfirmation != config.AuditFindingHumanRequired {
		t.Fatalf("Top 10 profile is not runnable: profile=%+v compatibility=%+v", profile, compatibility)
	}
	binding := profile.Workflows["check"]
	if binding.Workflow.Ref.Name != "audit-top10-source-risk" ||
		binding.Inputs["task"].Source != config.AuditInputFromItemPackage ||
		binding.Inputs["execution_manifest"].Source != config.AuditInputFromExecutionManifest ||
		binding.Inputs["source"].Source != config.AuditInputFromAudit {
		t.Fatalf("Top 10 Workflow boundary = %+v", binding)
	}
	tools := []string{}
	for _, stage := range binding.Workflow.Stages {
		for _, agent := range stage.Agents {
			for _, selection := range agent.Template.Toolsets {
				for _, tool := range selection.Tools {
					tools = append(tools, selection.Ref.ToolsetID+"@"+selection.Ref.Version+"/"+tool)
				}
			}
		}
	}
	sort.Strings(tools)
	for _, required := range []string{
		"audit-results@1/read_audit_task",
		"audit-results@1/submit_check_result",
		"security-findings@1/finding",
		"source-analysis@1/open_source_archive",
	} {
		if !contains(tools, required) {
			t.Fatalf("Top 10 Worker lacks %s: %v", required, tools)
		}
	}
}

func loadTop10Package(t *testing.T) ([]byte, *auditstandards.Package) {
	t.Helper()
	payload, pkg, err := auditstandards.PackageDirectory(top10PackageDirectory())
	if err != nil {
		t.Fatal(err)
	}
	return payload, pkg
}

func top10PackageDirectory() string {
	return filepath.Join(configRoot(), "audit-standards", "owasp-web-top10-2025")
}

func configRoot() string { return filepath.Join("..", "..", "..", "configs") }

func contains(values []string, candidate string) bool {
	index := sort.SearchStrings(values, candidate)
	return index < len(values) && values[index] == candidate
}
