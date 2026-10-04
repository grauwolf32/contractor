package agentskills

import (
	"bytes"
	"maps"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"slices"
	"strings"
	"testing"
)

// The pseudo skills are test-only Agent Skill sources laid out as an operator
// root. They exercise packaging code on realistic multi-file trees and never
// mirror the shipped catalog; their pinned digests change only with them or
// with the canonical archive encoding.
const pseudoSkillsRoot = "testdata/pseudo-skills"

type pseudoSkill struct {
	name      string
	digest    string
	manifest  Manifest
	resources []string
}

var pseudoSkills = []pseudoSkill{
	{
		name:   "pseudo-minimal",
		digest: "sha256:4f5c39df1adb911d164f2e4bf50ee60f9cf72aaa8dc81eb1c79c3cc0bb250013",
		manifest: Manifest{
			Name:        "pseudo-minimal",
			Description: "Test-only Agent Skill with a manifest and no resources.",
		},
		resources: []string{},
	},
	{
		name:   "pseudo-skill",
		digest: "sha256:a7f2d3908ff44469797408416e929b33ba06f063a3bfd136afeb1fd056026f36",
		manifest: Manifest{
			Name:          "pseudo-skill",
			Description:   "Test-only Agent Skill that exercises packaging, manifest metadata and resource resolution.",
			License:       "Test fixture only",
			Compatibility: "Contractor agentskills package tests",
			Metadata: map[string]string{
				"source-revision": "0123456789abcdef0123456789abcdef01234567",
				"fixture":         "pseudo",
			},
		},
		resources: []string{
			"assets/template.txt", "references/checklist.md", "references/patterns/matching.md",
		},
	},
}

func pseudoSkillSource(name string) string {
	return filepath.Join(pseudoSkillsRoot, SkillNamespace, name)
}

func TestPseudoSkillPackagingIsDeterministicAndPinned(t *testing.T) {
	t.Parallel()

	for _, specification := range pseudoSkills {
		t.Run(specification.name, func(t *testing.T) {
			t.Parallel()
			source := pseudoSkillSource(specification.name)
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
			if !reflect.DeepEqual(first.Manifest, specification.manifest) {
				t.Fatalf("manifest = %+v, want %+v", first.Manifest, specification.manifest)
			}

			resources := make([]string, len(first.Resources))
			expanded := int64(0)
			for index, resource := range first.Resources {
				resources[index] = resource.Path
				info, err := os.Stat(filepath.Join(source, filepath.FromSlash(resource.Path)))
				if err != nil || info.Size() != resource.Size {
					t.Fatalf("resource %s size = %d, source = (%v, %v)", resource.Path, resource.Size, info, err)
				}
				expanded += resource.Size
			}
			if !slices.Equal(resources, specification.resources) {
				t.Fatalf("resources = %v, want %v", resources, specification.resources)
			}
			manifest, err := os.Stat(filepath.Join(source, "SKILL.md"))
			if err != nil {
				t.Fatal(err)
			}
			if first.StoredBytes != int64(len(firstArchive)) || first.ExpandedBytes != expanded+manifest.Size() {
				t.Fatalf("package sizes = stored %d, expanded %d", first.StoredBytes, first.ExpandedBytes)
			}

			validated, err := Validate(firstArchive, specification.name)
			if err != nil || validated.Digest != first.Digest ||
				!reflect.DeepEqual(validated.Manifest, first.Manifest) ||
				!reflect.DeepEqual(validated.Resources, first.Resources) {
				t.Fatalf("canonical archive does not round-trip: (%+v, %v)", validated, err)
			}
			if _, err := Validate(firstArchive, "other-skill"); ErrorCode(err) != CodeNameMismatch {
				t.Fatalf("foreign expected name error = %v", err)
			}
		})
	}
}

func TestPseudoSkillReferencesResolveToExactMembers(t *testing.T) {
	source := pseudoSkillSource("pseudo-skill")
	_, pkg, err := PackageDirectory(source)
	if err != nil {
		t.Fatal(err)
	}
	members := pkg.Members()
	paths := make([]string, len(members))
	for index, member := range members {
		paths[index] = member.Path
	}
	if want := []string{
		"SKILL.md", "assets/template.txt", "references/checklist.md", "references/patterns/matching.md",
	}; !slices.Equal(paths, want) {
		t.Fatalf("members = %v, want %v", paths, want)
	}

	// Every package path named by a member, including the native resource
	// call, resolves to the exact source bytes of that member.
	memberRef := regexp.MustCompile(`(?:references|assets)/[a-z0-9][a-z0-9._/-]*\.(?:md|txt)`)
	nativeCall := regexp.MustCompile(`load_skill_resource\(skill_name="([a-z0-9-]+)", file_path="([^"]+)"\)`)
	referenced := make(map[string]bool)
	for _, member := range members {
		for _, target := range memberRef.FindAllString(string(member.Data()), -1) {
			referenced[target] = true
		}
		for _, match := range nativeCall.FindAllSubmatch(member.Data(), -1) {
			if string(match[1]) != pkg.Manifest.Name {
				t.Errorf("%s native call selects %q, want %q", member.Path, match[1], pkg.Manifest.Name)
			}
			referenced[string(match[2])] = true
		}
	}
	if want := []string{
		"assets/template.txt", "references/checklist.md", "references/patterns/matching.md",
	}; !slices.Equal(slices.Sorted(maps.Keys(referenced)), want) {
		t.Fatalf("referenced paths = %v, want %v", slices.Sorted(maps.Keys(referenced)), want)
	}
	for target := range referenced {
		member, ok := pkg.Member(target)
		if !ok {
			t.Errorf("referenced path %q is not a package member", target)
			continue
		}
		want, err := os.ReadFile(filepath.Join(source, filepath.FromSlash(target)))
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(member.Data(), want) || member.Size() != int64(len(want)) {
			t.Errorf("member %s differs from its source bytes", target)
		}
	}
	for _, missing := range []string{"references/missing.md", "references/patterns", "scripts/run.sh"} {
		if _, ok := pkg.Member(missing); ok {
			t.Errorf("non-member %q resolved", missing)
		}
	}
}

func TestBundledDiscoveryPackagesEveryPseudoSkill(t *testing.T) {
	plan, err := DiscoverBundled(pseudoSkillsRoot)
	if err != nil {
		t.Fatal(err)
	}
	want := make([]SeedMetadata, 0, len(pseudoSkills))
	for _, specification := range pseudoSkills {
		archive, pkg, err := PackageDirectory(pseudoSkillSource(specification.name))
		if err != nil {
			t.Fatal(err)
		}
		want = append(want, SeedMetadata{Name: pkg.Manifest.Name, Digest: pkg.Digest, Size: int64(len(archive))})
	}
	if got := plan.Packages(); !reflect.DeepEqual(got, want) {
		t.Fatalf("bundled discovery = %+v, want %+v", got, want)
	}
}

func TestPseudoSkillSourceViolationsAreRejected(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(t *testing.T, source string) string
		code   string
	}{
		{
			name: "directory name differs from manifest name",
			mutate: func(t *testing.T, source string) string {
				renamed := filepath.Join(filepath.Dir(source), "renamed-skill")
				if err := os.Rename(source, renamed); err != nil {
					t.Fatal(err)
				}
				return renamed
			},
			code: CodeNameMismatch,
		},
		{
			name: "scripts member",
			mutate: func(t *testing.T, source string) string {
				writePseudoSkillFile(t, source, "scripts/run.sh", []byte("#!/bin/sh\n"))
				return source
			},
			code: CodeMemberForbidden,
		},
		{
			name: "top-level member beside SKILL.md",
			mutate: func(t *testing.T, source string) string {
				writePseudoSkillFile(t, source, "notes.md", []byte("# Notes\n"))
				return source
			},
			code: CodeMemberForbidden,
		},
		{
			name: "non-portable reference name",
			mutate: func(t *testing.T, source string) string {
				writePseudoSkillFile(t, source, "references/Notes.md", []byte("# Notes\n"))
				return source
			},
			code: CodePathInvalid,
		},
		{
			name: "binary reference",
			mutate: func(t *testing.T, source string) string {
				writePseudoSkillFile(t, source, "references/blob.md", []byte{0xff, 0x00, 0x01})
				return source
			},
			code: CodeManifestInvalid,
		},
		{
			name: "binary asset",
			mutate: func(t *testing.T, source string) string {
				writePseudoSkillFile(t, source, "assets/blob.bin", []byte{0xff, 0x00, 0x01})
				return source
			},
		},
		{
			name: "reserved metadata key",
			mutate: func(t *testing.T, source string) string {
				editPseudoSkillManifest(t, source, "  fixture: pseudo\n", "  adk_fixture: pseudo\n")
				return source
			},
			code: CodeManifestInvalid,
		},
		{
			name: "manifest name differs from directory",
			mutate: func(t *testing.T, source string) string {
				editPseudoSkillManifest(t, source, "name: pseudo-skill\n", "name: pseudo-other\n")
				return source
			},
			code: CodeNameMismatch,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			root := t.TempDir()
			if err := os.CopyFS(root, os.DirFS(pseudoSkillsRoot)); err != nil {
				t.Fatal(err)
			}
			source := test.mutate(t, filepath.Join(root, SkillNamespace, "pseudo-skill"))
			_, _, err := PackageDirectory(source)
			if ErrorCode(err) != test.code {
				t.Fatalf("error = %v (%q), want %q", err, ErrorCode(err), test.code)
			}
			if test.code == "" {
				return
			}
			if _, err := DiscoverBundled(root); ErrorCode(err) != test.code {
				t.Fatalf("bundled discovery error = %v, want %q", err, test.code)
			}
		})
	}
}

func writePseudoSkillFile(t *testing.T, source, path string, data []byte) {
	t.Helper()
	target := filepath.Join(source, filepath.FromSlash(path))
	if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(target, data, 0o644); err != nil {
		t.Fatal(err)
	}
}

func editPseudoSkillManifest(t *testing.T, source, old, replacement string) {
	t.Helper()
	path := filepath.Join(source, "SKILL.md")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	edited := strings.Replace(string(data), old, replacement, 1)
	if edited == string(data) {
		t.Fatalf("pseudo skill manifest does not contain %q", old)
	}
	if err := os.WriteFile(path, []byte(edited), 0o644); err != nil {
		t.Fatal(err)
	}
}
