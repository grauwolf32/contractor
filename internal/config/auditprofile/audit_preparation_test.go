package auditprofile_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/clone"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
)

// copyAuditPreparationCatalog adds the scan check Workflows and the prepare
// Workflow that the shared prepared-openapi-scan profile selects.
func copyAuditPreparationCatalog(t *testing.T) string {
	t.Helper()
	root := copyCatalog(t, "audit-scan-catalog", "audit-preparation-catalog")
	writeAuditProfile(t, root, "prepared-openapi-scan", sharedAuditProfileYAML(t, "prepared-openapi-scan"))
	return root
}

func TestSharedAuditPreparationAttemptBudgetChangesDigestAndInvalidatesSnapshot(t *testing.T) {
	profile := loadProfile(t, copyAuditPreparationCatalog(t), "prepared-openapi-scan@1")
	if profile.Workflows["generate-api"].MaxRunAttempts != 2 {
		t.Fatal("prepare fixture no longer has two attempts")
	}
	digests := map[string]bool{profile.Ref.Digest: true}
	for _, attempts := range []int{1, 3} {
		candidate := profile.Clone()
		binding := candidate.Workflows["generate-api"]
		binding.MaxRunAttempts = attempts
		candidate.Workflows["generate-api"] = binding
		digest, err := auditprofile.ProfileDigest(candidate)
		if err != nil {
			t.Fatal(err)
		}
		if digests[digest] {
			t.Fatalf("maxRunAttempts=%d reused digest %s", attempts, digest)
		}
		digests[digest] = true
		raw, err := json.Marshal(candidate)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := auditprofile.DecodeResolvedAuditProfileSnapshot(raw); err == nil || !strings.Contains(err.Error(), "persisted AuditProfile digest is invalid") {
			t.Fatalf("changed maxRunAttempts=%d with stale digest = %v", attempts, err)
		}
	}
}

func TestSharedAuditPreparationDependenciesAndSnapshotValidation(t *testing.T) {
	profile := loadProfile(t, copyAuditPreparationCatalog(t), "prepared-openapi-scan@1")
	second := profile.Workflows["generate-api"]
	second.Inputs = clone.Map(second.Inputs)
	second.Inputs["existing_openapi"] = auditprofile.AuditWorkflowInputMapping{Source: auditprofile.AuditInputFromPreparation, Role: "generate-api", Name: "api"}
	profile.Workflows["refine-api"] = second
	profile.Inventory.Source.Role = "refine-api"
	profile.Workflows["scan"].Inputs["openapi"] = *profile.Inventory.Source
	if err := auditprofile.ValidateAuditPreparationProfile(profile); err != nil {
		t.Fatal(err)
	}
	if err := auditprofile.ValidateAuditTaskProfile(profile); err != nil {
		t.Fatal(err)
	}
	for _, mutate := range []func(*auditprofile.ResolvedAuditProfile){
		func(p *auditprofile.ResolvedAuditProfile) {
			b := p.Workflows["generate-api"]
			b.MaxRunAttempts = 0
			p.Workflows["generate-api"] = b
		},
		func(p *auditprofile.ResolvedAuditProfile) {
			p.Inventory.Source.Source = auditprofile.AuditInputFromItemPackage
		},
		func(p *auditprofile.ResolvedAuditProfile) {
			p.Workflows["generate-api"].Inputs["existing_openapi"] = auditprofile.AuditWorkflowInputMapping{Source: auditprofile.AuditInputFromPreparation, Role: "refine-api", Name: "api"}
		},
	} {
		candidate := profile.Clone()
		mutate(&candidate)
		redigest(t, &candidate)
		encoded, err := json.Marshal(candidate)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := auditprofile.DecodeResolvedAuditProfileSnapshot(encoded); err == nil {
			t.Fatal("invalid snapshot accepted despite a recomputed digest")
		}
	}
	encoded, err := json.Marshal(profile)
	if err != nil {
		t.Fatal(err)
	}
	obsolete := strings.Replace(string(encoded), `"implementation":"openapi-scans@1"`, `"implementation":"openapi-scans@1","sourceInput":"source"`, 1)
	if _, err := auditprofile.DecodeResolvedAuditProfileSnapshot([]byte(obsolete)); err == nil {
		t.Fatal("obsolete snapshot schema accepted")
	}
}
