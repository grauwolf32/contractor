package auditprograms

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
)

const asvsRevision = "5cf9b032440be53ce345ab3c130fda46ba1ce7a2"

var asvsRequirementIDs = []string{
	"v5.0.0-1.2.4",
	"v5.0.0-1.2.5",
	"v5.0.0-1.3.2",
	"v5.0.0-1.5.1",
	"v5.0.0-2.1.1",
}

func TestASVSPackageAndProfilePinExactSelectedDenominator(t *testing.T) {
	t.Parallel()

	payload, pkg := loadASVSPackage(t)
	again, second, err := auditstandards.PackageDirectory(asvsPackageDirectory())
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(payload, again) || pkg.Digest != second.Digest {
		t.Fatal("ASVS package bytes are not deterministic")
	}
	metadata := pkg.Document.Standard
	if metadata.Scheme != "owasp-asvs" || metadata.Version != "5.0.0" ||
		metadata.Source.Revision != asvsRevision || metadata.License.ID != "CC-BY-SA-4.0" ||
		metadata.License.Disclosure != auditstandards.DisclosureFull {
		t.Fatalf("ASVS package provenance = %+v", metadata)
	}
	entryIDs := make([]string, len(pkg.Document.Entries))
	for index, entry := range pkg.Document.Entries {
		entryIDs[index] = entry.ID
		if entry.Kind != "requirement" || entry.Level != "1" || entry.Statement == "" {
			t.Fatalf("ASVS entry is not an exact Level 1 requirement: %+v", entry)
		}
	}
	if !reflect.DeepEqual(entryIDs, asvsRequirementIDs) {
		t.Fatalf("ASVS packaged entries = %v", entryIDs)
	}

	snapshot, err := config.Load(configRoot(), config.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	profile, err := snapshot.AuditProfile("owasp-asvs-5-0-l1-source-review@1")
	if err != nil {
		t.Fatal(err)
	}
	selection := profile.Inventory.StandardSelection
	if selection == nil || selection.Scope != "ASVS 5.0 Level 1 source and documentation pilot (5 requirements)" ||
		!reflect.DeepEqual(selection.Levels, []string{"1"}) ||
		!reflect.DeepEqual(selection.EntryIDs, asvsRequirementIDs) ||
		profile.Mode != config.AuditModeRequirementsVerification ||
		!auditservice.ProfileCompatibility(profile).ServerCompatible {
		t.Fatalf("ASVS exact profile selection = %+v", profile)
	}
}

func TestASVSInventoryRetainsEveryRequirementExactlyOnce(t *testing.T) {
	t.Parallel()

	_, pkg := loadASVSPackage(t)
	revision := "asvs-retained-r1"
	options := asvsInventoryOptions(revision)
	first, err := auditdomain.BuildStandardMappingInventory(*pkg, options)
	if err != nil {
		t.Fatal(err)
	}
	second, err := auditdomain.BuildStandardMappingInventory(*pkg, options)
	if err != nil {
		t.Fatal(err)
	}
	if auditdomain.ValidateInventory(first) != nil || len(first.Tasks) != len(asvsRequirementIDs) ||
		!bytes.Equal(first.CanonicalInventory, second.CanonicalInventory) ||
		first.CanonicalInventoryDigest != second.CanonicalInventoryDigest {
		t.Fatalf("ASVS inventory is not exact and deterministic: %+v", first)
	}
	statements := make(map[string]string, len(pkg.Document.Entries))
	for _, entry := range pkg.Document.Entries {
		statements[entry.ID] = entry.Statement
	}
	manual := []string{}
	seen := make(map[string]bool, len(first.Tasks))
	for index, generated := range first.Tasks {
		task := generated.Document
		wantID := asvsRequirementIDs[index]
		if seen[task.ItemKey] || task.ItemKey != wantID || task.Standard == nil || task.Checklist == nil ||
			task.Standard.Scheme != "owasp-asvs" || task.Standard.Version != "5.0.0" ||
			task.Standard.MappingKey != wantID || !reflect.DeepEqual(task.Standard.EntryIDs, []string{wantID}) ||
			task.Checklist.Statement != statements[wantID] || task.Checklist.Version != "5.0.0" ||
			!reflect.DeepEqual(task.Checklist.RequiredEvidence, []string{"observation"}) {
			t.Fatalf("ASVS task %d lost exact authority: %+v", index, task)
		}
		seen[task.ItemKey] = true
		if generated.Item.ApprovalRequirement == auditdomain.ApprovalHumanReview {
			manual = append(manual, task.ItemKey)
		}
	}
	if !reflect.DeepEqual(manual, []string{"v5.0.0-2.1.1"}) {
		t.Fatalf("ASVS manual applicability set = %v", manual)
	}
}

func TestASVSSelectionRejectsMissingAndWrongVersionMappings(t *testing.T) {
	t.Parallel()

	_, pkg := loadASVSPackage(t)
	revision := "asvs-retained-r1"
	wrong := asvsInventoryOptions(revision)
	wrong.StandardSelection.EntryIDs[0] = "v5.0.1-1.2.4"
	if _, err := auditdomain.BuildStandardMappingInventory(*pkg, wrong); err == nil {
		t.Fatal("wrong-version selected requirement was accepted")
	}

	document := pkg.Document
	document.Mappings = document.Mappings[1:]
	directory := t.TempDir()
	manifest, err := json.Marshal(document)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(directory, auditstandards.ManifestPath), manifest, 0o600); err != nil {
		t.Fatal(err)
	}
	_, missing, err := auditstandards.PackageDirectory(directory)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := auditdomain.BuildStandardMappingInventory(*missing, asvsInventoryOptions(revision)); err == nil {
		t.Fatal("selected ASVS requirement without a mapping was accepted")
	}
}

func asvsInventoryOptions(revision string) auditdomain.InventoryOptions {
	return auditdomain.InventoryOptions{
		Round: 1, WorkflowRole: "check", SourceInputName: "standard",
		ApprovalRequirement: auditdomain.ApprovalNone,
		SourceRef: contracts.ArtifactRef{
			Namespace: "audit-example", Name: "standard-asvs", Revision: &revision,
		},
		StandardSelection: &auditdomain.StandardSelection{
			Scope:  "ASVS 5.0 Level 1 source and documentation pilot (5 requirements)",
			Levels: []string{"1"}, EntryIDs: append([]string{}, asvsRequirementIDs...),
		},
	}
}

func loadASVSPackage(t *testing.T) ([]byte, *auditstandards.Package) {
	t.Helper()
	payload, pkg, err := auditstandards.PackageDirectory(asvsPackageDirectory())
	if err != nil {
		t.Fatal(err)
	}
	return payload, pkg
}

func asvsPackageDirectory() string {
	return filepath.Join(configRoot(), "audit-standards", "owasp-asvs-5.0.0")
}
