package auditprograms

import (
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"path/filepath"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/configload"
)

func TestCommonCatalogAssertionProfilesUseExactHypothesisTasks(t *testing.T) {
	catalog, _, err := configload.LoadUnionReadOnly(configRoot(), filepath.Join(t.TempDir(), "managed"))
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"source-checklist", "owasp-asvs-5-0-l1-source-pilot", "owasp-asvs-5-0-l1-source-review", "owasp-top10-2025-source-risk", "owasp-wstg-4-2-active-http", "owasp-wstg-4-2-fast-active-http", "owasp-wstg-4-2-fast-source-review", "owasp-wstg-4-2-source-review"} {
		profile, err := catalog.AuditProfile(name + "@1")
		if err != nil || profile.Program.SeedSource == nil {
			t.Fatalf("common assertion profile %s: %v", name, err)
		}
		source := profile.Program.SeedSource
		binding := profile.Program.Workflows[source.Workflow]
		workflow := profile.Workflows[source.Workflow]
		if _, old := workflow.Inputs["task"]; old {
			t.Fatalf("%s still reads item packages", name)
		}
		if _, old := workflow.Inputs["execution_manifest"]; old {
			t.Fatalf("%s still reads execution manifests", name)
		}
		target, err := auditprofile.ResolveCompletionTarget(workflow, binding)
		if err != nil || target == nil {
			t.Fatalf("%s has no real common Runtime completion: %v", name, err)
		}
		if source.Standard != nil {
			_, pkg, err := auditstandards.PackageDirectory(filepath.Join(configRoot(), "audit-standards", source.Standard.Scheme+"-"+source.Standard.Version))
			if err != nil {
				t.Fatal(err)
			}
			plan, err := auditdomain.StandardAssertions(*pkg, *source)
			if err != nil {
				t.Fatal(err)
			}
			expected := len(pkg.Document.Entries)
			if len(source.EntryIDs) > 0 {
				expected = len(source.EntryIDs)
			}
			if len(plan.Hypotheses) != expected || len(plan.Work) != expected {
				t.Fatalf("%s changed selected membership: %d/%d expected %d", name, len(plan.Hypotheses), len(plan.Work), expected)
			}
			plan.Source = auditcontract.Material{ScopeKind: "audit", ScopeID: "catalog-audit", Ref: auditcontract.Material{}.Ref, Digest: "sha256:" + strings.Repeat("a", 64), MediaType: "application/json", SizeBytes: 1}
			revision := "catalog-r1"
			plan.Source.Ref.Namespace, plan.Source.Ref.Name, plan.Source.Ref.Revision = "inputs", "assertion-source", &revision
			if err := auditdomain.ValidateInitialAssertions(plan); err != nil {
				t.Fatalf("%s source selection is not executable: %v", name, err)
			}
			baseline, _ := strictjson.Canonical(map[string]any{"initial_assertions": plan})
			if _, err := auditdomain.ExecutionProgram(profile.Program, baseline); err != nil {
				t.Fatalf("%s selected program is not executable: %v", name, err)
			}
			for _, seed := range plan.Hypotheses {
				if seed.Scope["standard_version"] != source.Standard.Version || seed.Scope["evidence_contract"] == "" || seed.Scope["standard_mapping"] == "" || seed.Scope["applicability"] == "" {
					t.Fatalf("%s lost source provenance/sufficiency", name)
				}
			}
		}
	}
	for _, name := range []string{"openapi-operation-observe", "openapi-operation-trace", "openapi-sqlmap-scan", "openapi-nuclei-scan", "source-openapi-sqlmap-scan", "web-scan"} {
		profile, err := catalog.AuditProfile(name + "@1")
		if err != nil || profile.Program.Schema != "contractor.audit.program.v1" {
			t.Fatalf("common profile %s: %v", name, err)
		}
	}
}
