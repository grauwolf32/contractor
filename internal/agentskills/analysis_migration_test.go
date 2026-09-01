package agentskills

import (
	"bytes"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"go.yaml.in/yaml/v4"
)

const analysisSkillsSourceRevision = "9c76b56cf7b83377fb1dd5e4a17440fa27b723f3"

type migratedAnalysisSkill struct {
	name       string
	oldName    string
	digest     string
	references []string
}

var migratedAnalysisSkills = []migratedAnalysisSkill{
	{name: "stride", digest: "sha256:92cb91b0952fb419021e89ec5d977ae36b1ab6439d9f36f2b5240412ea530043"},
	{
		name: "trace", digest: "sha256:245b3799afc85ab27cb55fdeb196f461e85e8a5c4ab5542e2a959b61fd5fec98",
		references: []string{
			"references/annotations.md", "references/controls.md", "references/cwe-mapping.md",
			"references/finding-shapes.md", "references/frameworks.md", "references/sinks.md",
			"references/sources.md",
		},
	},
	{
		name: "vuln-scan", oldName: "vuln_scan",
		digest: "sha256:504c68f2c72545ab190d9b79140ee74fcab7abee6d02a4039cdc41caada20b16",
		references: []string{
			"references/absence-detection.md", "references/business-logic.md",
			"references/checklist.md", "references/grep-patterns.md",
			"references/miss-patterns.md", "references/php-wordpress.md",
			"references/secrets.md", "references/sink-patterns.md",
		},
	},
	{
		name: "vulns", digest: "sha256:92dc4640426c8aa5f6374eed1b53775552fe14daf1becd89d7f456886212d274",
		references: []string{
			"references/idor.md", "references/ssrf.md", "references/ssti.md", "references/xxe.md",
		},
	},
}

func TestMigratedAnalysisSkillsAreCompleteDeterministicAndClosed(t *testing.T) {
	t.Parallel()

	for _, specification := range migratedAnalysisSkills {
		specification := specification
		t.Run(specification.name, func(t *testing.T) {
			t.Parallel()
			source := filepath.Join("..", "..", "configs", "skills", specification.name)
			firstArchive, first, err := PackageDirectory(source)
			if err != nil {
				t.Fatal(err)
			}
			secondArchive, second, err := PackageDirectory(source)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(firstArchive, secondArchive) || first.Digest != second.Digest {
				t.Fatal("unchanged source did not produce a byte-identical package")
			}
			if first.Digest != specification.digest {
				t.Fatalf("package digest = %s, want %s", first.Digest, specification.digest)
			}
			if first.Manifest.Name != specification.name ||
				first.Manifest.Metadata["source-revision"] != analysisSkillsSourceRevision {
				t.Fatalf("manifest provenance = %+v", first.Manifest)
			}

			actualReferences := make([]string, len(first.Resources))
			for index, resource := range first.Resources {
				actualReferences[index] = resource.Path
			}
			if !slices.Equal(actualReferences, specification.references) {
				t.Fatalf("resources = %v, want %v", actualReferences, specification.references)
			}
			assertMigratedAnalysisReferenceClosure(t, specification, first)
		})
	}

	assertMigratedAnalysisInventory(t)
	assertMigratedAnalysisSkillsRemainUnassigned(t)
}

func assertMigratedAnalysisReferenceClosure(
	t *testing.T,
	specification migratedAnalysisSkill,
	pkg *Package,
) {
	t.Helper()
	members := make(map[string][]byte, len(pkg.Resources)+1)
	for _, member := range pkg.Members() {
		members[member.Path] = member.Data()
	}

	for memberPath, data := range members {
		text := string(data)
		for _, token := range []string{
			"skills_read", "skills_list", "run_skill_script", "scripts/",
			"report_vulnerability", "list_symbols", "read_file", "changed_paths",
			"AgentTemplate", "/home/ruslan/", "index is always in memory",
		} {
			if strings.Contains(text, token) {
				t.Errorf("%s retains unavailable platform token %q", memberPath, token)
			}
		}
		if specification.oldName != "" && strings.Contains(text, specification.oldName) {
			t.Errorf("%s retains pre-portable Skill name %q", memberPath, specification.oldName)
		}
	}

	resourceRef := regexp.MustCompile(`references/[a-z0-9][a-z0-9._/-]*\.md`)
	referencedByManifest := make(map[string]bool)
	for _, target := range resourceRef.FindAllString(string(members["SKILL.md"]), -1) {
		if _, ok := members[target]; !ok {
			t.Errorf("SKILL.md references missing resource %q", target)
		}
		referencedByManifest[target] = true
	}
	for memberPath, data := range members {
		for _, target := range resourceRef.FindAllString(string(data), -1) {
			if _, ok := members[target]; !ok {
				t.Errorf("%s references missing package member %q", memberPath, target)
			}
		}
	}
	for _, target := range specification.references {
		if !referencedByManifest[target] {
			t.Errorf("SKILL.md reference index omits %q", target)
		}
	}

	nativeCall := regexp.MustCompile(
		`load_skill_resource\(skill_name="([a-z0-9-]+)", file_path="(references/[a-z0-9._/-]+\.md)"\)`,
	)
	for memberPath, data := range members {
		for _, match := range nativeCall.FindAllSubmatch(data, -1) {
			if string(match[1]) != specification.name {
				t.Errorf("%s native call selects %q, want %q", memberPath, match[1], specification.name)
			}
			if _, ok := members[string(match[2])]; !ok {
				t.Errorf("%s native call selects missing resource %q", memberPath, match[2])
			}
		}
	}

	switch specification.name {
	case "stride":
		if !bytes.Contains(members["SKILL.md"], []byte("Model only — never verify exploitability")) {
			t.Error("STRIDE analysis boundary is absent")
		}
	case "vulns":
		root := string(members["SKILL.md"])
		for _, boundary := range []string{
			"Authorization is a precondition", "Non-destructive PoCs",
			"Stop probing once confirmed", "only as scope allows",
		} {
			if !strings.Contains(root, boundary) {
				t.Errorf("authorized-testing boundary %q is absent", boundary)
			}
		}
	}
}

func assertMigratedAnalysisInventory(t *testing.T) {
	t.Helper()
	documentPath := filepath.Join(
		"..", "..", "docs", "migrations", "contractor-old-agent-skills-analysis.md",
	)
	data, err := os.ReadFile(documentPath)
	if err != nil {
		t.Fatal(err)
	}
	text := string(data)
	if !strings.Contains(text, analysisSkillsSourceRevision) ||
		!strings.Contains(text, "Every one of the 23 source Markdown files") {
		t.Fatal("migration document omits pinned revision or inventory cardinality")
	}

	mappings := make([][2]string, 0, 23)
	for _, specification := range migratedAnalysisSkills {
		oldName := specification.oldName
		if oldName == "" {
			oldName = specification.name
		}
		mappings = append(mappings, [2]string{
			"contractor/skills/" + oldName + "/index.md",
			"configs/skills/" + specification.name + "/SKILL.md",
		})
		for _, reference := range specification.references {
			mappings = append(mappings, [2]string{
				"contractor/skills/" + oldName + "/" + reference,
				"configs/skills/" + specification.name + "/" + reference,
			})
		}
	}
	if len(mappings) != 23 {
		t.Fatalf("migration inventory has %d entries, want 23", len(mappings))
	}
	seenSources := make(map[string]bool, len(mappings))
	seenTargets := make(map[string]bool, len(mappings))
	for _, mapping := range mappings {
		if seenSources[mapping[0]] || seenTargets[mapping[1]] {
			t.Fatalf("duplicate migration mapping %v", mapping)
		}
		seenSources[mapping[0]], seenTargets[mapping[1]] = true, true
		if !strings.Contains(text, "`"+mapping[0]+"`") ||
			!strings.Contains(text, "`"+mapping[1]+"`") {
			t.Errorf("migration document omits mapping %s -> %s", mapping[0], mapping[1])
		}
	}
}

func assertMigratedAnalysisSkillsRemainUnassigned(t *testing.T) {
	t.Helper()
	root := filepath.Join("..", "..", "configs", "agent-templates")
	entries, err := os.ReadDir(root)
	if err != nil {
		t.Fatal(err)
	}
	targets := map[string]bool{"stride": true, "trace": true, "vuln-scan": true, "vulns": true}
	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".yaml" {
			continue
		}
		data, readErr := os.ReadFile(filepath.Join(root, entry.Name()))
		if readErr != nil {
			t.Fatal(readErr)
		}
		var document struct {
			Spec struct {
				Skills []contracts.ArtifactRef `yaml:"skills"`
			} `yaml:"spec"`
		}
		if err := yaml.Unmarshal(data, &document); err != nil {
			t.Fatalf("parse %s: %v", entry.Name(), err)
		}
		for _, skill := range document.Spec.Skills {
			if targets[skill.Name] {
				t.Errorf("AgentTemplate %s prematurely selects skills/%s", entry.Name(), skill.Name)
			}
		}
	}
}

func TestMigratedAnalysisSkillInventoryNamesAreUnique(t *testing.T) {
	names := make([]string, len(migratedAnalysisSkills))
	for index, specification := range migratedAnalysisSkills {
		names[index] = specification.name
	}
	sort.Strings(names)
	for index := 1; index < len(names); index++ {
		if names[index] == names[index-1] {
			t.Fatalf("duplicate migrated Agent Skill %q", names[index])
		}
	}
}
