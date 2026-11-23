package auditprograms

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditstandards"
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
