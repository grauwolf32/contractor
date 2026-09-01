package agentskills

import (
	"archive/zip"
	"bytes"
	"fmt"
	"hash/crc32"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

var errSourceLimit = fmt.Errorf("source member exceeds limit")

// PackageDirectory validates a source tree and returns its canonical archive.
// The expected skill name is the source directory's base name.
func PackageDirectory(source string) ([]byte, *Package, error) {
	source = filepath.Clean(source)
	info, err := os.Lstat(source)
	if err != nil || !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return nil, nil, validationError(CodeMemberForbidden, "")
	}
	expectedName := filepath.Base(filepath.Clean(source))
	if !validSkillName(expectedName) {
		return nil, nil, validationError(CodeNameMismatch, "")
	}

	paths := make([]string, 0)
	var expanded int64
	walkedEntries := 0
	err = filepath.WalkDir(source, func(hostPath string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return validationError(CodeMemberForbidden, "")
		}
		if hostPath == source {
			return nil
		}
		walkedEntries++
		if walkedEntries > MaximumEntries {
			return validationError(CodeLimitExceeded, "")
		}
		relative, relErr := filepath.Rel(source, hostPath)
		if relErr != nil {
			return validationError(CodePathInvalid, "")
		}
		portable := filepath.ToSlash(relative)
		validated, directory, pathErr := validateMemberPath(portable + map[bool]string{true: "/"}[entry.IsDir()])
		if pathErr != nil {
			return pathErr
		}
		info, infoErr := entry.Info()
		if infoErr != nil {
			return validationError(CodeMemberForbidden, validated)
		}
		if info.Mode()&os.ModeSymlink != 0 || !info.IsDir() && !info.Mode().IsRegular() {
			return validationError(CodeMemberForbidden, validated)
		}
		if directory {
			if !allowedDirectory(validated) {
				return validationError(CodeMemberForbidden, validated)
			}
			return nil
		}
		if !allowedFile(validated) {
			return validationError(CodeMemberForbidden, validated)
		}
		limit := int64(MaximumResourceBytes)
		if validated == "SKILL.md" {
			limit = MaximumManifestBytes
		}
		if info.Size() > limit || info.Size() > MaximumExpandedBytes-expanded {
			return validationError(CodeLimitExceeded, validated)
		}
		expanded += info.Size()
		paths = append(paths, validated)
		return nil
	})
	if err != nil {
		return nil, nil, err
	}
	if len(paths) > MaximumEntries {
		return nil, nil, validationError(CodeLimitExceeded, "")
	}
	sort.Strings(paths)

	var archive bytes.Buffer
	writer := zip.NewWriter(&archive)
	for _, path := range paths {
		limit := int64(MaximumResourceBytes)
		if path == "SKILL.md" {
			limit = MaximumManifestBytes
		}
		data, readErr := readSourceFileNoFollow(source, path, limit)
		if readErr != nil {
			_ = writer.Close()
			if readErr == errSourceLimit {
				return nil, nil, validationError(CodeLimitExceeded, path)
			}
			return nil, nil, validationError(CodeMemberForbidden, path)
		}
		header := &zip.FileHeader{
			Name: path, Method: zip.Store, CRC32: crc32.ChecksumIEEE(data),
			CompressedSize64: uint64(len(data)), UncompressedSize64: uint64(len(data)),
			CompressedSize: uint32(len(data)), UncompressedSize: uint32(len(data)),
			CreatorVersion: 3<<8 | 20, ReaderVersion: 20,
			ModifiedDate: 1<<5 | 1, ModifiedTime: 0,
			ExternalAttrs: uint32(0100644) << 16,
		}
		entry, createErr := writer.CreateRaw(header)
		if createErr != nil {
			return nil, nil, fmt.Errorf("create canonical skill archive: %w", createErr)
		}
		if _, writeErr := entry.Write(data); writeErr != nil {
			return nil, nil, fmt.Errorf("create canonical skill archive: %w", writeErr)
		}
	}
	if err := writer.Close(); err != nil {
		return nil, nil, fmt.Errorf("close canonical skill archive: %w", err)
	}
	if archive.Len() > MaximumArchiveBytes {
		return nil, nil, validationError(CodeLimitExceeded, "")
	}
	payload := archive.Bytes()
	validated, err := Validate(payload, expectedName)
	if err != nil {
		return nil, nil, err
	}
	return append([]byte(nil), payload...), validated, nil
}

func sourcePathParts(path string) []string { return strings.Split(path, "/") }
