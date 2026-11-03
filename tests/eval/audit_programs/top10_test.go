package auditprograms

import (
	"bytes"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditstandards"
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
