package agentskills

import (
	"archive/zip"
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

type packageFixtureSet struct {
	SchemaVersion string           `json:"schemaVersion"`
	Cases         []packageFixture `json:"cases"`
}

type packageFixture struct {
	ID            string `json:"id"`
	ArchiveBase64 string `json:"archiveBase64"`
	ExpectedName  string `json:"expectedName"`
	ExpectedCode  string `json:"expectedCode"`
	Expected      struct {
		Digest    string     `json:"digest"`
		Manifest  Manifest   `json:"manifest"`
		Resources []Resource `json:"resources"`
	} `json:"expected"`
}

func TestSharedPackageCorpus(t *testing.T) {
	fixtures := loadPackageFixtures(t)
	if fixtures.SchemaVersion != "1.0" || len(fixtures.Cases) < 20 {
		t.Fatalf("unexpected shared corpus: %#v", fixtures)
	}
	for _, fixture := range fixtures.Cases {
		t.Run(fixture.ID, func(t *testing.T) {
			payload, err := base64.StdEncoding.DecodeString(fixture.ArchiveBase64)
			if err != nil {
				t.Fatal(err)
			}
			validated, err := Validate(payload, fixture.ExpectedName)
			if fixture.ExpectedCode != "" {
				if ErrorCode(err) != fixture.ExpectedCode {
					t.Fatalf("error code = %q (%v), want %q", ErrorCode(err), err, fixture.ExpectedCode)
				}
				if err == nil || strings.Contains(err.Error(), "\n") || len(err.Error()) > MaximumPathBytes+64 {
					t.Fatalf("unsafe validation diagnostic: %q", err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if validated.Digest != fixture.Expected.Digest || !reflect.DeepEqual(validated.Manifest, fixture.Expected.Manifest) || !reflect.DeepEqual(validated.Resources, fixture.Expected.Resources) {
				t.Fatalf("result mismatch:\n got %#v\nwant %#v", validated, fixture.Expected)
			}
			manifest, ok := validated.Member("SKILL.md")
			if !ok || manifest.Size() == 0 {
				t.Fatal("validated manifest member missing")
			}
			copyOfData := manifest.Data()
			copyOfData[0] ^= 0xff
			if manifest.Data()[0] == copyOfData[0] {
				t.Fatal("member data is not defensively copied")
			}
		})
	}
}

func TestSkillNameAcceptsDigitsAfterTheFirstCharacter(t *testing.T) {
	payload := makeTestZIP(t, zip.Store, []testEntry{{
		"SKILL.md", skillDocument("likec4", "LikeC4.", ""), 0,
	}})
	if _, err := Validate(payload, "likec4"); err != nil {
		t.Fatalf("digit-bearing skill name rejected: %v", err)
	}
}

func TestPackageLimitsAreStreamedAndExact(t *testing.T) {
	manifest := []byte("---\nname: limits\ndescription: Limits.\n---\n# Limits\n")
	tests := []struct {
		name    string
		payload []byte
		code    string
	}{
		{name: "stored archive over maximum", payload: make([]byte, MaximumArchiveBytes+1), code: CodeLimitExceeded},
		{name: "resource at maximum", payload: makeTestZIP(t, zip.Deflate, []testEntry{{"SKILL.md", manifest, 0}, {"assets/exact.bin", make([]byte, MaximumResourceBytes), 0}})},
		{name: "resource over maximum", payload: makeTestZIP(t, zip.Deflate, []testEntry{{"SKILL.md", manifest, 0}, {"assets/over.bin", make([]byte, MaximumResourceBytes+1), 0}}), code: CodeLimitExceeded},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := Validate(test.payload, "limits")
			if ErrorCode(err) != test.code {
				t.Fatalf("error = %v (%q), want %q", err, ErrorCode(err), test.code)
			}
		})
	}
}

func TestPackageDirectoryIsDeterministicAndNoFollow(t *testing.T) {
	root := t.TempDir()
	source := filepath.Join(root, "example")
	if err := os.MkdirAll(filepath.Join(source, "references"), 0o755); err != nil {
		t.Fatal(err)
	}
	manifest := []byte("---\nname: example\ndescription: Example.\n---\n# Example\n")
	if err := os.WriteFile(filepath.Join(source, "SKILL.md"), manifest, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(source, "references", "guide.md"), []byte("guide\n"), 0o777); err != nil {
		t.Fatal(err)
	}
	first, firstPackage, err := PackageDirectory(source)
	if err != nil {
		t.Fatal(err)
	}
	second, secondPackage, err := PackageDirectory(source)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(first, second) || firstPackage.Digest != secondPackage.Digest {
		t.Fatal("unchanged source did not produce byte-identical archives")
	}
	reader, err := zip.NewReader(bytes.NewReader(first), int64(len(first)))
	if err != nil {
		t.Fatal(err)
	}
	for index, file := range reader.File {
		if file.Name == "" || file.Method != zip.Store || len(file.Extra) != 0 || file.Comment != "" || file.Mode().Perm() != 0o644 || file.ModifiedDate != 33 || file.ModifiedTime != 0 {
			t.Fatalf("member %d is not canonical: %#v", index, file.FileHeader)
		}
		if index > 0 && reader.File[index-1].Name >= file.Name {
			t.Fatalf("members are not lexicographic: %q then %q", reader.File[index-1].Name, file.Name)
		}
	}
	if err := os.Symlink("guide.md", filepath.Join(source, "references", "link.md")); err != nil {
		t.Fatal(err)
	}
	if _, _, err := PackageDirectory(source); ErrorCode(err) != CodeMemberForbidden {
		t.Fatalf("symlink error = %v", err)
	}
	if got, err := os.ReadFile(filepath.Join(source, "SKILL.md")); err != nil || !bytes.Equal(got, manifest) {
		t.Fatalf("source was modified: %v", err)
	}
}

func TestEntryCountAndExpandedAggregateLimits(t *testing.T) {
	manifest := []byte("---\nname: limits\ndescription: Limits.\n---\n# Limits\n")
	entries := make([]testEntry, 0, MaximumEntries)
	entries = append(entries, testEntry{"SKILL.md", manifest, 0})
	for index := 0; index < MaximumEntries-1; index++ {
		entries = append(entries, testEntry{fmt.Sprintf("assets/f%04d", index), nil, 0})
	}
	if _, err := Validate(makeTestZIP(t, zip.Store, entries), "limits"); err != nil {
		t.Fatalf("exact entry boundary rejected: %v", err)
	}
	entries = append(entries, testEntry{"assets/overflow", nil, 0})
	if _, err := Validate(makeTestZIP(t, zip.Store, entries), "limits"); ErrorCode(err) != CodeLimitExceeded {
		t.Fatalf("entry overflow error = %v", err)
	}

	aggregate := []testEntry{{"SKILL.md", manifest, 0}}
	block := bytes.Repeat([]byte("x"), MaximumResourceBytes)
	for index := 0; index < 31; index++ {
		aggregate = append(aggregate, testEntry{fmt.Sprintf("assets/b%02d", index), block, 0})
	}
	aggregate = append(aggregate, testEntry{"assets/exact", block[:MaximumResourceBytes-len(manifest)], 0})
	if _, err := Validate(makeTestZIP(t, zip.Deflate, aggregate), "limits"); err != nil {
		t.Fatalf("exact expanded boundary rejected: %v", err)
	}
	aggregate[len(aggregate)-1].data = block
	if _, err := Validate(makeTestZIP(t, zip.Deflate, aggregate), "limits"); ErrorCode(err) != CodeLimitExceeded {
		t.Fatalf("aggregate overflow error = %v", err)
	}
}

func TestManifestAndPathBoundaries(t *testing.T) {
	exactDescription := strings.Repeat("d", MaximumDescriptionBytes)
	exactName := strings.Repeat("a", 64)
	frontmatterPrefix := "name: boundary\ndescription: Boundary.\n"
	exactFrontmatter := frontmatterPrefix + "#" + strings.Repeat("x", MaximumFrontmatterBytes-len(frontmatterPrefix)-1)
	tests := []struct {
		name     string
		manifest []byte
		code     string
	}{
		{name: "name exact", manifest: skillDocument(exactName, "Exact.", "")},
		{name: "name over", manifest: skillDocument(strings.Repeat("a", 65), "Over.", ""), code: CodeLimitExceeded},
		{name: "description exact", manifest: skillDocument("boundary", exactDescription, "")},
		{name: "description over", manifest: skillDocument("boundary", exactDescription+"d", ""), code: CodeLimitExceeded},
		{name: "frontmatter exact", manifest: []byte("---\n" + exactFrontmatter + "\n---\n# Body\n")},
		{name: "frontmatter over", manifest: []byte("---\n" + exactFrontmatter + "#\n---\n# Body\n"), code: CodeLimitExceeded},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			expectedName := "boundary"
			if strings.HasPrefix(test.name, "name ") {
				expectedName = ""
			}
			_, err := Validate(makeTestZIP(t, zip.Deflate, []testEntry{{"SKILL.md", test.manifest, 0}}), expectedName)
			if ErrorCode(err) != test.code {
				t.Fatalf("error = %v (%q), want %q", err, ErrorCode(err), test.code)
			}
		})
	}

	base := skillDocument("boundary", "Boundary.", "")
	exactBody := append(append([]byte(nil), base...), bytes.Repeat([]byte("x"), MaximumManifestBytes-len(base))...)
	if _, err := Validate(makeTestZIP(t, zip.Deflate, []testEntry{{"SKILL.md", exactBody, 0}}), "boundary"); err != nil {
		t.Fatalf("exact SKILL.md boundary rejected: %v", err)
	}
	if _, err := Validate(makeTestZIP(t, zip.Deflate, []testEntry{{"SKILL.md", append(exactBody, 'x'), 0}}), "boundary"); ErrorCode(err) != CodeLimitExceeded {
		t.Fatalf("SKILL.md overflow error = %v", err)
	}

	path := "assets/" + strings.Repeat("a", 126) + "/" + strings.Repeat("b", 126) + "/" + strings.Repeat("c", 125) + "/" + strings.Repeat("d", 125)
	if len(path) != MaximumPathBytes {
		t.Fatalf("test path length = %d", len(path))
	}
	if _, err := Validate(makeTestZIP(t, zip.Store, []testEntry{{"SKILL.md", base, 0}, {path, nil, 0}}), "boundary"); err != nil {
		t.Fatalf("exact path boundary rejected: %v", err)
	}
	if _, err := Validate(makeTestZIP(t, zip.Store, []testEntry{{"SKILL.md", base, 0}, {path + "d", nil, 0}}), "boundary"); ErrorCode(err) != CodePathInvalid {
		t.Fatalf("path overflow error = %v", err)
	}
}

func TestStoredArchiveExactBoundary(t *testing.T) {
	manifest := skillDocument("archive-boundary", "Archive boundary.", "")
	entries := []testEntry{{"SKILL.md", manifest, 0}}
	block := bytes.Repeat([]byte("x"), MaximumResourceBytes)
	for index := 0; index < 15; index++ {
		entries = append(entries, testEntry{fmt.Sprintf("assets/b%02d", index), block, 0})
	}
	entries = append(entries, testEntry{"assets/fill", nil, 0})
	base := makeTestZIP(t, zip.Store, entries)
	fill := MaximumArchiveBytes - len(base)
	if fill <= 0 || fill > MaximumResourceBytes {
		t.Fatalf("invalid test fill size %d", fill)
	}
	entries[len(entries)-1].data = bytes.Repeat([]byte("z"), fill)
	payload := makeTestZIP(t, zip.Store, entries)
	if len(payload) != MaximumArchiveBytes {
		t.Fatalf("archive boundary = %d", len(payload))
	}
	if _, err := Validate(payload, "archive-boundary"); err != nil {
		t.Fatalf("exact archive boundary rejected: %v", err)
	}
	if _, err := Validate(append(payload, 0), "archive-boundary"); ErrorCode(err) != CodeLimitExceeded {
		t.Fatalf("archive overflow error = %v", err)
	}
}

func skillDocument(name, description, extra string) []byte {
	return []byte("---\nname: " + name + "\ndescription: " + description + "\n" + extra + "---\n# Body\n")
}

type testEntry struct {
	name string
	data []byte
	mode os.FileMode
}

func makeTestZIP(t *testing.T, method uint16, entries []testEntry) []byte {
	t.Helper()
	var buffer bytes.Buffer
	writer := zip.NewWriter(&buffer)
	for _, entry := range entries {
		header := &zip.FileHeader{Name: entry.name, Method: method}
		mode := entry.mode
		if mode == 0 {
			mode = 0o644
		}
		header.SetMode(mode)
		output, err := writer.CreateHeader(header)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := output.Write(entry.data); err != nil {
			t.Fatal(err)
		}
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	return buffer.Bytes()
}

func loadPackageFixtures(t *testing.T) packageFixtureSet {
	t.Helper()
	payload, err := os.ReadFile("../../testdata/agent-skills/cases.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixtures packageFixtureSet
	if err := json.Unmarshal(payload, &fixtures); err != nil {
		t.Fatal(err)
	}
	return fixtures
}
