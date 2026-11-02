package config

import (
	"encoding/json"
	"path/filepath"
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
			snapshot, err := LoadUnionReadOnly(root, filepath.Join(t.TempDir(), "managed"))
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
// join the algorithm. The pins changed once when every resolved Agent binding
// gained its explicit contextManagement selection (WFL-R91), and again
// when the active Finding confirmation policy was removed (FND-R215).
func TestAuditProfilesWithoutPreparationKeepDigests(t *testing.T) {
	snapshot := mustLoad(t, copyCoreFixture(t, "audit-scan-catalog"))
	for ref, digest := range map[string]string{
		"openapi-sqlmap-scan@1": "sha256:89678cf01042060cb57f2829bfa6814f049e6d26dffd15295c32afe11e06dedc",
		"openapi-nuclei-scan@1": "sha256:51f6e62884a24c846f4ce3bc74b92c96bbbb0b284d588455ef1d64bf0701aeb0",
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
