package config

import (
	"encoding/json"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestAuditPreparationAuthoringFixtures(t *testing.T) {
	fixture := "../../api/testdata/audit-composition"
	profileYAML := string(readFile(t, filepath.Join(fixture, "prepared-openapi-scan.yaml")))
	var cases []struct {
		Name         string      `json:"name"`
		Valid        bool        `json:"valid"`
		Replacements [][2]string `json:"replacements"`
	}
	if err := json.Unmarshal(readFile(t, filepath.Join(fixture, "profile-cases.json")), &cases); err != nil {
		t.Fatal(err)
	}
	for _, tc := range cases {
		t.Run(tc.Name, func(t *testing.T) {
			root := copyAuditPreparationCatalog(t)
			data := profileYAML
			for _, replacement := range tc.Replacements {
				if !strings.Contains(data, replacement[0]) {
					t.Fatalf("fixture replacement does not match: %s", replacement[0])
				}
				data = strings.ReplaceAll(data, replacement[0], replacement[1])
			}
			writeAuditProfile(t, root, "prepared-openapi-scan", data)
			snapshot, err := Load(root, MVPDescriptors())
			if !tc.Valid {
				if err == nil {
					t.Fatal("invalid preparation profile accepted")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			profile, err := snapshot.AuditProfile("prepared-openapi-scan@1")
			if err != nil {
				t.Fatal(err)
			}
			encoded, err := json.Marshal(profile)
			if err != nil {
				t.Fatal(err)
			}
			restored, err := DecodeResolvedAuditProfileSnapshot(encoded)
			if err != nil || !reflect.DeepEqual(profile, restored) {
				t.Fatalf("snapshot round trip: %v", err)
			}
			profile.Inventory.Source.Name = "mutated"
			again, err := snapshot.AuditProfile("prepared-openapi-scan@1")
			if err != nil || again.Inventory.Source.Name != "api" {
				t.Fatal("inventory source mutation leaked")
			}
		})
	}
}

func TestAuditPreparationAttemptBudgetChangesDigestAndInvalidatesSnapshot(t *testing.T) {
	root := copyAuditPreparationCatalog(t)
	writeAuditProfile(t, root, "prepared-openapi-scan", string(readFile(t, "../../api/testdata/audit-composition/prepared-openapi-scan.yaml")))
	profile, err := mustLoad(t, root, MVPDescriptors()).AuditProfile("prepared-openapi-scan@1")
	if err != nil {
		t.Fatal(err)
	}
	if profile.Workflows["generate-api"].MaxRunAttempts != 2 {
		t.Fatal("prepare fixture no longer has two attempts")
	}
	digests := map[string]bool{profile.Ref.Digest: true}
	for _, attempts := range []int{1, 3} {
		candidate := cloneAuditProfile(profile)
		binding := candidate.Workflows["generate-api"]
		binding.MaxRunAttempts = attempts
		candidate.Workflows["generate-api"] = binding
		digest, err := auditProfileDigest(Selector{ID: profile.Ref.Name, Version: profile.Ref.Version}, candidate)
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
		if _, err := DecodeResolvedAuditProfileSnapshot(raw); err == nil || !strings.Contains(err.Error(), "persisted AuditProfile digest is invalid") {
			t.Fatalf("changed maxRunAttempts=%d with stale digest = %v", attempts, err)
		}
	}
}

// TestAuditProfilesWithoutPreparationKeepDigests pins frozen fixture digests:
// a profile without a prepare role keeps its digest when preparation fields
// join the algorithm.
func TestAuditProfilesWithoutPreparationKeepDigests(t *testing.T) {
	snapshot := mustLoad(t, copyCoreFixture(t, "audit-scan-catalog"), MVPDescriptors())
	for ref, digest := range map[string]string{
		"openapi-sqlmap-scan@1": "sha256:a011789cffbe65cee70e2c8f4965e693a3c6593ac05e6a1cff65cf5a8b6da390",
		"openapi-nuclei-scan@1": "sha256:eab2a7a3ea204906fe70d29b0887ec5d95b17b61e4ffc83ab8bbcd0f7478141b",
	} {
		profile, err := snapshot.AuditProfile(ref)
		if err != nil {
			t.Fatal(err)
		}
		if profile.Ref.Digest != digest {
			t.Fatalf("%s digest = %s, want %s", ref, profile.Ref.Digest, digest)
		}
	}
}

func TestAuditPreparationDependenciesAndSnapshotValidation(t *testing.T) {
	root := copyAuditPreparationCatalog(t)
	data := string(readFile(t, "../../api/testdata/audit-composition/prepared-openapi-scan.yaml"))
	writeAuditProfile(t, root, "prepared-openapi-scan", data)
	snapshot := mustLoad(t, root, MVPDescriptors())
	profile, err := snapshot.AuditProfile("prepared-openapi-scan@1")
	if err != nil {
		t.Fatal(err)
	}
	second := profile.Workflows["generate-api"]
	second.Inputs = cloneMap(second.Inputs)
	second.Inputs["existing_openapi"] = AuditWorkflowInputMapping{Source: AuditInputFromPreparation, Role: "generate-api", Name: "api"}
	profile.Workflows["refine-api"] = second
	profile.Inventory.Source.Role = "refine-api"
	profile.Workflows["scan"].Inputs["openapi"] = *profile.Inventory.Source
	if err := ValidateAuditPreparationProfile(profile); err != nil {
		t.Fatal(err)
	}
	if err := ValidateAuditTaskProfile(profile); err != nil {
		t.Fatal(err)
	}
	for _, mutate := range []func(*ResolvedAuditProfile){
		func(p *ResolvedAuditProfile) {
			b := p.Workflows["generate-api"]
			b.MaxRunAttempts = 0
			p.Workflows["generate-api"] = b
		},
		func(p *ResolvedAuditProfile) { p.Inventory.Source.Source = AuditInputFromItemPackage },
		func(p *ResolvedAuditProfile) {
			p.Workflows["generate-api"].Inputs["existing_openapi"] = AuditWorkflowInputMapping{Source: AuditInputFromPreparation, Role: "refine-api", Name: "api"}
		},
	} {
		candidate := cloneAuditProfile(profile)
		mutate(&candidate)
		candidate.Ref.Digest, err = auditProfileDigest(Selector{ID: candidate.Ref.Name, Version: candidate.Ref.Version}, candidate)
		if err != nil {
			t.Fatal(err)
		}
		encoded, err := json.Marshal(candidate)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := DecodeResolvedAuditProfileSnapshot(encoded); err == nil {
			t.Fatal("invalid snapshot accepted despite a recomputed digest")
		}
	}
	encoded, err := json.Marshal(profile)
	if err != nil {
		t.Fatal(err)
	}
	obsolete := strings.Replace(string(encoded), `"implementation":"openapi-scans@1"`, `"implementation":"openapi-scans@1","sourceInput":"source"`, 1)
	if _, err := DecodeResolvedAuditProfileSnapshot([]byte(obsolete)); err == nil {
		t.Fatal("obsolete snapshot schema accepted")
	}
}

// copyAuditPreparationCatalog adds the scan check Workflows and the prepare
// Workflow that the shared prepared-openapi-scan profile selects.
func copyAuditPreparationCatalog(t *testing.T) string {
	t.Helper()
	return copyCoreFixture(t, "audit-scan-catalog", "audit-preparation-catalog")
}
