package auditprograms

import (
	"path/filepath"
	"testing"

	"github.com/grauwolf32/contractor/internal/configload"
)

func TestCommonCatalogExcludesRetiredAssertionProfiles(t *testing.T) {
	catalog, _, err := configload.LoadUnionReadOnly(configRoot(), filepath.Join(t.TempDir(), "managed"))
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"source-checklist", "owasp-asvs-5-0-l1-source-pilot", "owasp-asvs-5-0-l1-source-review", "owasp-top10-2025-source-risk", "owasp-wstg-4-2-active-http", "owasp-wstg-4-2-fast-active-http", "owasp-wstg-4-2-fast-source-review", "owasp-wstg-4-2-source-review"} {
		if _, err := catalog.AuditProfile(name + "@1"); err == nil {
			t.Fatalf("retired assertion profile %s is still executable", name)
		}
	}
	for _, name := range []string{"openapi-operation-observe", "openapi-operation-trace", "openapi-sqlmap-scan", "openapi-nuclei-scan", "source-openapi-sqlmap-scan", "web-scan"} {
		profile, err := catalog.AuditProfile(name + "@1")
		if err != nil || profile.Program.Schema != "contractor.audit.program.v1" {
			t.Fatalf("common profile %s: %v", name, err)
		}
	}
}
