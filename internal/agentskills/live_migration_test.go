package agentskills

import (
	"bytes"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"
)

const liveSkillsSourceRevision = "9c76b56cf7b83377fb1dd5e4a17440fa27b723f3"

type migratedLiveSkill struct {
	name       string
	digest     string
	references []string
}

var migratedLiveSkills = []migratedLiveSkill{
	{name: "auth", digest: "sha256:c7165c518840bf65cb2f139d9b06ed1aa240597356c20f081ab1ae2894a3f62f"},
	{name: "caido", digest: "sha256:676d2d4736054dad6556a5a9f8fac49e7ffd89858bf9761fbe2517634b3459c1"},
	{name: "code-exec", digest: "sha256:ae482885e234465206e603a258463508998478d9845f1c323cfd2d2b5a7bd4d0"},
	{
		name: "exploit", digest: "sha256:e44969fa40e36907273490e1f7e58743003d46612f1d5ac31bc9156c16276e3f",
		references: []string{
			"references/auth-bypass.md", "references/auth-discovery.md",
			"references/broken-auth.md", "references/cmdi.md", "references/idor.md",
			"references/info-disclosure.md", "references/mass-assignment.md",
			"references/nosqli.md", "references/path-traversal.md",
			"references/rate-limiting.md", "references/sqli.md", "references/ssrf.md",
			"references/ssti.md", "references/xss.md", "references/xxe.md",
		},
	},
}

func TestMigratedLiveSkillsAreCompleteDeterministicAndClosed(t *testing.T) {
	t.Parallel()

	for _, specification := range migratedLiveSkills {
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
				first.Manifest.Metadata["source-revision"] != liveSkillsSourceRevision {
				t.Fatalf("manifest provenance = %+v", first.Manifest)
			}

			actualReferences := make([]string, len(first.Resources))
			for index, resource := range first.Resources {
				actualReferences[index] = resource.Path
			}
			if !slices.Equal(actualReferences, specification.references) {
				t.Fatalf("resources = %v, want %v", actualReferences, specification.references)
			}
			assertMigratedLiveReferenceClosure(t, specification, first)
		})
	}

	assertMigratedLiveInventoryAndCompatibility(t)
}

func assertMigratedLiveReferenceClosure(
	t *testing.T,
	specification migratedLiveSkill,
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
			"AgentTemplate", "SandboxProfile", "/home/ruslan/",
			"ephemeral Kali container", "host network", "mounted read-only at /project",
			"pyjwt is available", "persists across calls within this run",
			"auth/creds", "auth/user1", "auth/user2", "auth/refresh",
			"auth/forged-jwt", "auth/oauth", "auth/bypass", "auth-creds",
		} {
			if strings.Contains(text, token) {
				t.Errorf("%s retains unavailable platform token %q", memberPath, token)
			}
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

	memoryCall := regexp.MustCompile(
		`(?:write_memory|read_memory|append_memory)\([^\n)]*name="([^"]+)"`,
	)
	validMemoryName := regexp.MustCompile(`^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$`)
	for memberPath, data := range members {
		for _, match := range memoryCall.FindAllSubmatch(data, -1) {
			name := string(match[1])
			if len(name) > 128 || !validMemoryName.MatchString(name) {
				t.Errorf("%s uses invalid memory note name %q", memberPath, name)
			}
		}
	}

	root := strings.Join(strings.Fields(strings.ToLower(string(members["SKILL.md"]))), " ")
	if !strings.Contains(root, "current worker invocation") {
		t.Error("root omits the conditional Worker-visible capability boundary")
	}
	if specification.name == "auth" || specification.name == "exploit" {
		for _, boundary := range []string{"authorization", "target scope", "non-destructive", "clean up"} {
			if !strings.Contains(root, boundary) {
				t.Errorf("root safety contract omits %q", boundary)
			}
		}
	}
	if specification.name == "auth" {
		for _, note := range []string{
			"auth_creds", "auth_endpoints", "auth_user1", "auth_user2",
			"auth_refresh", "auth_forged_jwt", "auth_oauth", "auth_bypass",
		} {
			if !strings.Contains(string(members["SKILL.md"]), note) {
				t.Errorf("auth migration omits valid note name %q", note)
			}
		}
	}
}

func assertMigratedLiveInventoryAndCompatibility(t *testing.T) {
	t.Helper()
	documentPath := filepath.Join(
		"..", "..", "docs", "migrations", "contractor-old-agent-skills-live.md",
	)
	data, err := os.ReadFile(documentPath)
	if err != nil {
		t.Fatal(err)
	}
	text := string(data)
	if !strings.Contains(text, liveSkillsSourceRevision) ||
		!strings.Contains(text, "19 Markdown knowledge") {
		t.Fatal("migration document omits pinned revision or inventory cardinality")
	}

	mappings := make([][2]string, 0, 19)
	for _, specification := range migratedLiveSkills {
		mappings = append(mappings, [2]string{
			"contractor/skills/" + specification.name + "/index.md",
			"configs/skills/" + specification.name + "/SKILL.md",
		})
		for _, reference := range specification.references {
			mappings = append(mappings, [2]string{
				"contractor/skills/" + specification.name + "/" + reference,
				"configs/skills/" + specification.name + "/" + reference,
			})
		}
		if !strings.Contains(text, specification.digest) {
			t.Errorf("migration document omits %s digest %s", specification.name, specification.digest)
		}
	}
	if len(mappings) != 19 {
		t.Fatalf("migration inventory has %d entries, want 19", len(mappings))
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

	for _, operation := range []string{
		"load_skill", "load_skill_resource", "read_memory", "write_memory",
		"http_request", "http_session_set", "get_vulnerability", "submit_verdict",
		"run_python", "execute_bash", "caido_replay", "caido_automate_run",
		"caido_history", "caido_request_detail", "caido_workflow_list",
		"caido_workflow_run", "caido_workflow_findings",
	} {
		if !strings.Contains(text, "`"+operation+"`") {
			t.Errorf("compatibility matrix omits operation %q", operation)
		}
	}
}
