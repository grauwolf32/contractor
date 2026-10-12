package config

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestSharedAuditPreparationAuthoringFixtures(t *testing.T) {
	profileYAML := string(contractstest.ReadFile(t, "api", "testdata", "audit-composition", "prepared-openapi-scan.yaml"))
	var cases []struct {
		Name         string      `json:"name"`
		Valid        bool        `json:"valid"`
		Replacements [][2]string `json:"replacements"`
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "audit-composition", "profile-cases.json"), &cases); err != nil {
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
			snapshot, err := Load(root)
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
			restored, err := auditprofile.DecodeResolvedAuditProfileSnapshot(encoded)
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

// TestAuditProfilesWithoutPreparationKeepDigests pins frozen fixture digests:
// a profile without a prepare role keeps its digest when preparation fields
// join the algorithm.
func TestAuditProfilesWithoutPreparationKeepDigests(t *testing.T) {
	snapshot := mustLoad(t, copyCoreFixture(t, "audit-scan-catalog"))
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

// copyAuditPreparationCatalog adds the scan check Workflows and the prepare
// Workflow that the shared prepared-openapi-scan profile selects.
func copyAuditPreparationCatalog(t *testing.T) string {
	t.Helper()
	return copyCoreFixture(t, "audit-scan-catalog", "audit-preparation-catalog")
}
