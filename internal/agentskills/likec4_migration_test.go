package agentskills

import (
	"bytes"
	"os"
	"path"
	"path/filepath"
	"reflect"
	"regexp"
	"strings"
	"testing"
)

const likeC4SourceRevision = "9c76b56cf7b83377fb1dd5e4a17440fa27b723f3"

var likeC4ReferencePaths = []string{
	"references/cli.md",
	"references/configuration.md",
	"references/deployment.md",
	"references/dynamic-views.md",
	"references/examples.md",
	"references/identifier-validity.md",
	"references/include-predicates-wildcards.md",
	"references/model.md",
	"references/predicates.md",
	"references/relationships-bidirectional.md",
	"references/specification.md",
	"references/style-tokens-colors.md",
	"references/troubleshooting.md",
	"references/views.md",
}

func TestRepositoryLikeC4SkillMigrationIsCompleteAndDeterministic(t *testing.T) {
	t.Parallel()

	source := filepath.Join("..", "..", "configs", "skills", "likec4")
	firstArchive, first, err := PackageDirectory(source)
	if err != nil {
		t.Fatal(err)
	}
	secondArchive, second, err := PackageDirectory(source)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(firstArchive, secondArchive) || first.Digest != second.Digest {
		t.Fatal("unchanged LikeC4 source did not produce a byte-identical package")
	}
	const expectedDigest = "sha256:84bc32ac3f6ca32d13785280090701e1d54a3e4f0cd373572236f0a2cd22b95e"
	if first.Digest != expectedDigest {
		t.Fatalf("LikeC4 package digest = %s, want %s", first.Digest, expectedDigest)
	}
	if first.Manifest.Name != "likec4" ||
		first.Manifest.Metadata["source-revision"] != likeC4SourceRevision {
		t.Fatalf("LikeC4 manifest provenance = %+v", first.Manifest)
	}

	actualReferences := make([]string, len(first.Resources))
	for index, resource := range first.Resources {
		actualReferences[index] = resource.Path
	}
	if !reflect.DeepEqual(actualReferences, likeC4ReferencePaths) {
		t.Fatalf("LikeC4 resources = %v, want %v", actualReferences, likeC4ReferencePaths)
	}
	assertLikeC4ReferencesResolve(t, first)
	assertLikeC4MigrationInventory(t)
}

func assertLikeC4ReferencesResolve(t *testing.T, skill *Package) {
	t.Helper()
	members := make(map[string][]byte, len(skill.Resources)+1)
	for _, member := range skill.Members() {
		members[member.Path] = member.Data()
	}

	legacyTokens := []string{
		"skills_read", "skills_list", "index is always in memory",
		"/home/ruslan/", "validate_likec4(path=",
	}
	for memberPath, data := range members {
		text := string(data)
		for _, token := range legacyTokens {
			if strings.Contains(text, token) {
				t.Errorf("%s retains legacy token %q", memberPath, token)
			}
		}
	}

	manifest := string(members["SKILL.md"])
	resourceRef := regexp.MustCompile(`references/[a-z0-9][a-z0-9._/-]*\.md`)
	referenced := make(map[string]bool)
	for _, target := range resourceRef.FindAllString(manifest, -1) {
		if _, ok := members[target]; !ok {
			t.Errorf("SKILL.md references missing resource %q", target)
		}
		referenced[target] = true
	}
	for _, target := range likeC4ReferencePaths {
		if !referenced[target] {
			t.Errorf("SKILL.md reference index omits %q", target)
		}
	}

	markdownLink := regexp.MustCompile(`\]\(([^)#?]+\.md)(?:#[^)]*)?\)`)
	for memberPath, data := range members {
		for _, match := range markdownLink.FindAllSubmatch(data, -1) {
			target := path.Clean(path.Join(path.Dir(memberPath), string(match[1])))
			if _, ok := members[target]; !ok {
				t.Errorf("%s links to missing package member %q", memberPath, target)
			}
		}
	}
}

func assertLikeC4MigrationInventory(t *testing.T) {
	t.Helper()
	documentPath := filepath.Join(
		"..", "..", "docs", "migrations", "contractor-old-agent-skills-likec4.md",
	)
	data, err := os.ReadFile(documentPath)
	if err != nil {
		t.Fatal(err)
	}
	text := string(data)
	if !strings.Contains(text, likeC4SourceRevision) {
		t.Errorf("migration document omits source revision %s", likeC4SourceRevision)
	}
	if !strings.Contains(text, "contractor/skills/likec4/index.md") ||
		!strings.Contains(text, "configs/skills/likec4/SKILL.md") {
		t.Error("migration document omits the index.md to SKILL.md mapping")
	}
	for _, target := range likeC4ReferencePaths {
		source := "contractor/skills/likec4/" + target
		destination := "configs/skills/likec4/" + target
		if !strings.Contains(text, source) || !strings.Contains(text, destination) {
			t.Errorf("migration document does not account for %s", target)
		}
	}
}
