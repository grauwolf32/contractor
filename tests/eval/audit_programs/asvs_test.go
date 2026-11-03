package auditprograms

import (
	"bytes"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditstandards"
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
