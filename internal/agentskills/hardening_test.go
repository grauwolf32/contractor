package agentskills

import (
	"encoding/base64"
	"encoding/json"
	"os"
	"reflect"
	"strings"
	"testing"
)

func TestSharedPackageCorpusCoversHardeningClasses(t *testing.T) {
	fixtures := loadPackageFixtures(t)
	seen := make(map[string]bool, len(fixtures.Cases))
	for _, fixture := range fixtures.Cases {
		if seen[fixture.ID] {
			t.Fatalf("duplicate shared corpus case %q", fixture.ID)
		}
		seen[fixture.ID] = true
	}
	for _, required := range []string{
		"valid-full", "valid-deflate", "path-traversal", "path-absolute",
		"path-backslash", "path-unicode", "path-duplicate",
		"path-prefix-collision", "path-prefix-collision-reverse",
		"path-duplicate-manifest", "member-scripts", "member-symlink",
		"member-special-fifo", "manifest-alias", "manifest-custom-tag",
		"manifest-malformed-yaml", "manifest-nul", "archive-forged-size",
		"archive-crc-mismatch", "archive-truncated-payload",
		"archive-unsupported-method", "archive-encrypted",
	} {
		if !seen[required] {
			t.Errorf("shared validator corpus omits hardening case %q", required)
		}
	}
}

func FuzzValidateAgentSkillPackage(f *testing.F) {
	payload, err := os.ReadFile("../../testdata/agent-skills/cases.json")
	if err != nil {
		f.Fatal(err)
	}
	var fixtures packageFixtureSet
	if err := json.Unmarshal(payload, &fixtures); err != nil {
		f.Fatal(err)
	}
	for _, fixture := range fixtures.Cases {
		archive, err := base64.StdEncoding.DecodeString(fixture.ArchiveBase64)
		if err != nil {
			f.Fatal(err)
		}
		f.Add(archive)
	}
	f.Add([]byte("not a zip"))

	stableCodes := map[string]bool{
		CodeArchiveInvalid: true, CodePathInvalid: true, CodeMemberForbidden: true,
		CodeManifestInvalid: true, CodeNameMismatch: true, CodeLimitExceeded: true,
	}
	f.Fuzz(func(t *testing.T, archive []byte) {
		validated, err := Validate(archive, "")
		if err != nil {
			if !stableCodes[ErrorCode(err)] {
				t.Fatalf("unstable validation error %T: %v", err, err)
			}
			if strings.Contains(err.Error(), "\n") || len(err.Error()) > MaximumPathBytes+64 {
				t.Fatalf("unsafe validation diagnostic: %q", err)
			}
			return
		}
		if validated.StoredBytes != int64(len(archive)) ||
			validated.StoredBytes > MaximumArchiveBytes ||
			validated.ExpandedBytes > MaximumExpandedBytes {
			t.Fatalf("validator returned an out-of-bounds package: %+v", validated)
		}
		again, err := Validate(append([]byte(nil), archive...), "")
		if err != nil || again.Digest != validated.Digest ||
			!reflect.DeepEqual(again.Manifest, validated.Manifest) ||
			!reflect.DeepEqual(again.Resources, validated.Resources) {
			t.Fatalf("exact package validation is not deterministic: (%+v, %v)", again, err)
		}
	})
}
