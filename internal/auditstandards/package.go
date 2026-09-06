package auditstandards

import (
	"archive/zip"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"os"
	"path/filepath"
	"strings"
)

func PackageDirectory(source string) ([]byte, *Package, error) {
	clean := filepath.Clean(source)
	info, err := os.Lstat(clean)
	if err != nil || !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return nil, nil, validationError(CodeMemberForbidden, "")
	}
	entries, err := os.ReadDir(clean)
	if err != nil || len(entries) != 1 || entries[0].Name() != ManifestPath ||
		entries[0].Type()&os.ModeSymlink != 0 {
		return nil, nil, validationError(CodeMemberForbidden, "")
	}
	manifestInfo, err := entries[0].Info()
	if err != nil || !manifestInfo.Mode().IsRegular() || manifestInfo.Mode()&os.ModeSymlink != 0 ||
		manifestInfo.Size() > MaximumManifestBytes {
		return nil, nil, validationError(CodeMemberForbidden, ManifestPath)
	}
	manifest, err := readSourceManifestNoFollow(clean)
	if err != nil {
		if errors.Is(err, errSourceManifestLimit) {
			return nil, nil, validationError(CodeLimitExceeded, ManifestPath)
		}
		return nil, nil, validationError(CodeMemberForbidden, ManifestPath)
	}
	document, err := DecodeDocument(manifest)
	if err != nil {
		return nil, nil, err
	}
	canonical, err := canonicalDocument(document)
	if err != nil {
		return nil, nil, validationError(CodeManifestInvalid, ManifestPath)
	}
	payload, err := canonicalArchive(canonical)
	if err != nil {
		return nil, nil, err
	}
	validated, err := Validate(payload, document.Standard.Reference())
	if err != nil {
		return nil, nil, err
	}
	return payload, validated, nil
}

func Validate(payload []byte, expected Reference) (*Package, error) {
	if len(payload) == 0 || len(payload) > MaximumPackageBytes {
		return nil, validationError(CodeLimitExceeded, "")
	}
	reader, err := zip.NewReader(bytes.NewReader(payload), int64(len(payload)))
	if err != nil || len(reader.File) != 1 {
		return nil, validationError(CodeArchiveInvalid, "")
	}
	member := reader.File[0]
	if member.Name != ManifestPath || strings.Contains(member.Name, "\\") || strings.HasPrefix(member.Name, "/") ||
		strings.Contains(member.Name, "..") || member.FileInfo().IsDir() || member.Flags&1 != 0 ||
		member.Method != zip.Store || member.UncompressedSize64 > MaximumManifestBytes ||
		member.CompressedSize64 > MaximumPackageBytes {
		return nil, validationError(CodePathInvalid, member.Name)
	}
	mode := member.Mode()
	if mode&(^mode.Perm()) != 0 || mode.Perm() != 0 && mode.Perm() != 0o644 {
		return nil, validationError(CodeMemberForbidden, ManifestPath)
	}
	manifest, err := readZipMember(member)
	if err != nil {
		return nil, err
	}
	document, err := DecodeDocument(manifest)
	if err != nil {
		return nil, err
	}
	if expected.Scheme != "" || expected.Version != "" {
		if document.Standard.Scheme != expected.Scheme || document.Standard.Version != expected.Version {
			return nil, validationError(CodeIdentityMismatch, ManifestPath)
		}
	}
	canonical, err := canonicalDocument(document)
	if err != nil || !bytes.Equal(canonical, manifest) {
		return nil, validationError(CodeManifestInvalid, ManifestPath)
	}
	canonicalPayload, err := canonicalArchive(canonical)
	if err != nil || !bytes.Equal(canonicalPayload, payload) {
		return nil, validationError(CodeArchiveInvalid, "")
	}
	digest := sha256.Sum256(payload)
	return &Package{
		Document: document, Digest: "sha256:" + hex.EncodeToString(digest[:]),
		StoredBytes: int64(len(payload)), ExpandedBytes: int64(len(manifest)),
		payload: append([]byte(nil), payload...),
	}, nil
}

func canonicalDocument(document Document) ([]byte, error) {
	normalizeDocument(&document)
	if err := validateDocument(document); err != nil {
		return nil, err
	}
	return jsonMarshal(document)
}

// Kept behind a tiny seam so tests can assert package determinism without
// depending on encoder indentation or host filesystem metadata.
func jsonMarshal(value any) ([]byte, error) {
	return json.Marshal(value)
}

func canonicalArchive(manifest []byte) ([]byte, error) {
	var buffer bytes.Buffer
	writer := zip.NewWriter(&buffer)
	header := &zip.FileHeader{
		Name: ManifestPath, Method: zip.Store, CRC32: crc32.ChecksumIEEE(manifest),
		CompressedSize64: uint64(len(manifest)), UncompressedSize64: uint64(len(manifest)),
		CompressedSize: uint32(len(manifest)), UncompressedSize: uint32(len(manifest)),
		CreatorVersion: 3<<8 | 20, ReaderVersion: 20,
		ModifiedDate: 1<<5 | 1, ModifiedTime: 0,
		ExternalAttrs: uint32(0o100644) << 16,
	}
	entry, err := writer.CreateRaw(header)
	if err != nil {
		return nil, fmt.Errorf("create canonical Audit standard archive: %w", err)
	}
	if _, err := entry.Write(manifest); err != nil {
		return nil, fmt.Errorf("write canonical Audit standard archive: %w", err)
	}
	if err := writer.Close(); err != nil {
		return nil, fmt.Errorf("close canonical Audit standard archive: %w", err)
	}
	if buffer.Len() > MaximumPackageBytes {
		return nil, validationError(CodeLimitExceeded, "")
	}
	return append([]byte(nil), buffer.Bytes()...), nil
}

func readZipMember(member *zip.File) ([]byte, error) {
	reader, err := member.Open()
	if err != nil {
		return nil, validationError(CodeArchiveInvalid, ManifestPath)
	}
	defer reader.Close()
	data, err := io.ReadAll(io.LimitReader(reader, MaximumManifestBytes+1))
	if err != nil || len(data) > MaximumManifestBytes {
		if len(data) > MaximumManifestBytes {
			return nil, validationError(CodeLimitExceeded, ManifestPath)
		}
		return nil, validationError(CodeArchiveInvalid, ManifestPath)
	}
	var extra [1]byte
	if count, readErr := reader.Read(extra[:]); count != 0 || !errors.Is(readErr, io.EOF) {
		return nil, validationError(CodeArchiveInvalid, ManifestPath)
	}
	return data, nil
}
