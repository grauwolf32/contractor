package agentskills

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"sort"
	"strings"
	"testing"
)

var workerFacingServerVocabulary = regexp.MustCompile(`(?i)\bagent(?:[ _-]*)template\b`)

func TestMigratedAgentSkillsAreDeterministicAndWorkerFacing(t *testing.T) {
	t.Parallel()

	want := map[string]string{
		"auth":      "sha256:c7165c518840bf65cb2f139d9b06ed1aa240597356c20f081ab1ae2894a3f62f",
		"caido":     "sha256:ab88f0a1f411c67b5060bb338928d2b7bd13b5096ed076d6d9b7ac5ae2448499",
		"code-exec": "sha256:ae482885e234465206e603a258463508998478d9845f1c323cfd2d2b5a7bd4d0",
		"exploit":   "sha256:e44969fa40e36907273490e1f7e58743003d46612f1d5ac31bc9156c16276e3f",
		"likec4":    "sha256:84bc32ac3f6ca32d13785280090701e1d54a3e4f0cd373572236f0a2cd22b95e",
		"stride":    "sha256:92cb91b0952fb419021e89ec5d977ae36b1ab6439d9f36f2b5240412ea530043",
		"trace":     "sha256:245b3799afc85ab27cb55fdeb196f461e85e8a5c4ab5542e2a959b61fd5fec98",
		"vuln-scan": "sha256:504c68f2c72545ab190d9b79140ee74fcab7abee6d02a4039cdc41caada20b16",
		"vulns":     "sha256:92dc4640426c8aa5f6374eed1b53775552fe14daf1becd89d7f456886212d274",
	}
	root := filepath.Join("..", "..", "configs", "skills")
	entries, err := os.ReadDir(root)
	if err != nil {
		t.Fatal(err)
	}
	actual := make([]string, 0, len(entries))
	for _, entry := range entries {
		if !entry.IsDir() {
			t.Fatalf("bundled Skill root contains non-directory %q", entry.Name())
		}
		actual = append(actual, entry.Name())
	}
	sort.Strings(actual)
	wantNames := make([]string, 0, len(want))
	for name := range want {
		wantNames = append(wantNames, name)
	}
	sort.Strings(wantNames)
	if !reflect.DeepEqual(actual, wantNames) {
		t.Fatalf("bundled Skill inventory = %v, want %v", actual, wantNames)
	}

	for _, name := range actual {
		name := name
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			source := filepath.Join(root, name)
			firstBytes, first, err := PackageDirectory(source)
			if err != nil {
				t.Fatal(err)
			}
			secondBytes, second, err := PackageDirectory(source)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(firstBytes, secondBytes) || first.Digest != second.Digest ||
				first.Digest != want[name] || first.Manifest.Name != name {
				t.Fatalf("non-deterministic or unexpected package: %s/%s, want %s", first.Digest, second.Digest, want[name])
			}
			for _, member := range first.Members() {
				if filepath.Ext(member.Path) != ".md" {
					continue
				}
				if workerFacingServerVocabulary.Match(member.Data()) {
					t.Errorf("%s exposes the Server-side AgentTemplate abstraction", member.Path)
				}
			}
		})
	}
}

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
