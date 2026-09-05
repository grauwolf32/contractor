package auditdomain

import (
	"archive/zip"
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestAuditPackageRoundTripIsCanonicalAndDefensive(t *testing.T) {
	inputs := []PackageInput{
		{ID: "second", Path: "data/z.json", MediaType: "application/json", Data: []byte(`{"z":true}`)},
		{ID: "first", Path: "data/a.txt", MediaType: "text/plain", Data: []byte("alpha")},
	}
	first, validated, err := BuildPackage("example-1", PackageKindEvidence, "", inputs)
	if err != nil {
		t.Fatal(err)
	}
	second, again, err := BuildPackage("example-1", PackageKindEvidence, "", []PackageInput{inputs[1], inputs[0]})
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(first, second) || validated.Digest != again.Digest {
		t.Fatal("equivalent package inputs did not produce byte-identical archives")
	}
	if validated.Manifest.Members[0].Path != "data/a.txt" || validated.Manifest.Members[1].Path != "data/z.json" {
		t.Fatalf("manifest is not canonical: %+v", validated.Manifest.Members)
	}
	member, ok := validated.MemberByID("first")
	if !ok || string(member.Data()) != "alpha" {
		t.Fatalf("missing package member: %+v", member)
	}
	copy := member.Data()
	copy[0] = 'X'
	if string(member.Data()) != "alpha" {
		t.Fatal("validated member aliases caller-owned data")
	}
	reader, err := zip.NewReader(bytes.NewReader(first), int64(len(first)))
	if err != nil {
		t.Fatal(err)
	}
	for _, file := range reader.File {
		if file.Method != zip.Store || file.ModifiedDate != 33 || file.ModifiedTime != 0 || file.Mode().Perm() != 0o644 || len(file.Extra) != 0 || file.Comment != "" {
			t.Fatalf("non-canonical ZIP header: %+v", file.FileHeader)
		}
	}
}

func TestAuditPackageRejectsUnsafeOrInconsistentArchives(t *testing.T) {
	validManifest := PackageManifest{
		Schema: PackageSchema, PackageID: "test", Kind: PackageKindEvidence,
		Members: []PackageMemberManifest{{
			ID: "body", Path: "body.txt", MediaType: "text/plain", Size: 4, Digest: digestBytes([]byte("body")),
		}},
	}
	manifestBytes, err := canonicalJSON(validManifest)
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name    string
		entries []rawZIPEntry
		code    string
	}{
		{name: "traversal", entries: []rawZIPEntry{{"manifest.json", manifestBytes, 0}, {"../body.txt", []byte("body"), 0}}, code: CodePackagePathInvalid},
		{name: "symlink", entries: []rawZIPEntry{{"manifest.json", manifestBytes, 0}, {"body.txt", []byte("body"), 0o120777}}, code: CodeMemberForbidden},
		{name: "extra member", entries: []rawZIPEntry{{"manifest.json", manifestBytes, 0}, {"body.txt", []byte("body"), 0}, {"extra.txt", nil, 0}}, code: CodePackageInvalid},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := ValidatePackage(makeRawZIP(t, zip.Store, test.entries))
			if ErrorCode(err) != test.code {
				t.Fatalf("error = %v (%q), want %q", err, ErrorCode(err), test.code)
			}
		})
	}

	t.Run("digest mismatch", func(t *testing.T) {
		entries := []rawZIPEntry{{"manifest.json", manifestBytes, 0}, {"body.txt", []byte("else"), 0}}
		_, err := ValidatePackage(makeRawZIP(t, zip.Store, entries))
		if ErrorCode(err) != CodeDigestMismatch {
			t.Fatalf("error = %v", err)
		}
	})

	t.Run("expanded member limit", func(t *testing.T) {
		oversized := bytes.Repeat([]byte("x"), MaximumMemberBytes+1)
		_, err := ValidatePackage(makeRawZIP(t, zip.Deflate, []rawZIPEntry{{"manifest.json", manifestBytes, 0}, {"body.txt", oversized, 0}}))
		if ErrorCode(err) != CodeLimitExceeded {
			t.Fatalf("error = %v", err)
		}
	})
}

func TestAuditPackageManifestIsStrictAndCanonical(t *testing.T) {
	member := []byte("body")
	manifest := PackageManifest{
		Schema: PackageSchema, PackageID: "test", Kind: PackageKindEvidence,
		Members: []PackageMemberManifest{{ID: "body", Path: "body.txt", MediaType: "text/plain", Size: 4, Digest: digestBytes(member)}},
	}
	canonical, err := canonicalJSON(manifest)
	if err != nil {
		t.Fatal(err)
	}
	nonCanonical := []byte(strings.Replace(string(canonical), `{"kind":`, `{"schema":"contractor.audit.package.v1","kind":`, 1))
	// The replacement duplicates schema if it matched; either a duplicate key
	// or a non-canonical representation must be rejected.
	if bytes.Equal(nonCanonical, canonical) {
		var generic map[string]any
		if err := json.Unmarshal(canonical, &generic); err != nil {
			t.Fatal(err)
		}
		nonCanonical, err = json.MarshalIndent(generic, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
	}
	_, err = ValidatePackage(makeRawZIP(t, zip.Store, []rawZIPEntry{{"manifest.json", nonCanonical, 0}, {"body.txt", member, 0}}))
	if ErrorCode(err) != CodePackageInvalid {
		t.Fatalf("non-canonical manifest error = %v", err)
	}
}

type rawZIPEntry struct {
	name string
	data []byte
	mode uint32
}

func makeRawZIP(t *testing.T, method uint16, entries []rawZIPEntry) []byte {
	t.Helper()
	var buffer bytes.Buffer
	writer := zip.NewWriter(&buffer)
	for _, entry := range entries {
		header := &zip.FileHeader{Name: entry.name, Method: method}
		if entry.mode != 0 {
			header.CreatorVersion = 3 << 8
			header.ExternalAttrs = entry.mode << 16
		}
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

func FuzzAuditPackage(f *testing.F) {
	payload, _, err := BuildPackage("seed", PackageKindEvidence, "", []PackageInput{{
		ID: "body", Path: "body.txt", MediaType: "text/plain", Data: []byte("seed"),
	}})
	if err != nil {
		f.Fatal(err)
	}
	f.Add(payload)
	f.Add([]byte("not a zip"))
	f.Fuzz(func(t *testing.T, candidate []byte) {
		validated, err := ValidatePackage(candidate)
		if err != nil {
			if ErrorCode(err) == "" {
				t.Fatalf("unstable error type: %T: %v", err, err)
			}
			return
		}
		if validated.StoredBytes != int64(len(candidate)) || validated.StoredBytes > MaximumArchiveBytes || validated.ExpandedBytes > MaximumExpandedBytes || len(validated.Members()) > MaximumMembers {
			t.Fatalf("accepted out-of-bounds package: %+v", validated)
		}
	})
}
