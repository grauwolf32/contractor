package agentskills

import (
	"archive/zip"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"io"
	"sort"
	"strings"
	"unicode/utf8"
)

func Validate(payload []byte, expectedName string) (*Package, error) {
	if len(payload) > MaximumArchiveBytes {
		return nil, validationError(CodeLimitExceeded, "")
	}
	reader, err := zip.NewReader(bytes.NewReader(payload), int64(len(payload)))
	if err != nil {
		return nil, validationError(CodeArchiveInvalid, "")
	}
	if len(reader.File) > MaximumEntries {
		return nil, validationError(CodeLimitExceeded, "")
	}

	seen := make(map[string]bool, len(reader.File))
	members := make([]Member, 0, len(reader.File))
	var expanded int64
	var declaredExpanded uint64
	manifestCount := 0
	for _, file := range reader.File {
		path, directory, pathErr := validateMemberPath(file.Name)
		if pathErr != nil {
			return nil, pathErr
		}
		if _, exists := seen[path]; exists {
			return nil, validationError(CodePathInvalid, path)
		}
		if hasPathCollision(seen, path, directory) {
			return nil, validationError(CodePathInvalid, path)
		}
		seen[path] = directory
		if file.Flags&1 != 0 || file.Method != zip.Store && file.Method != zip.Deflate {
			return nil, validationError(CodeArchiveInvalid, path)
		}
		mode := file.Mode()
		if directory {
			if !allowedDirectory(path) || !mode.IsDir() && mode != 0 {
				return nil, validationError(CodeMemberForbidden, path)
			}
			if file.UncompressedSize64 != 0 {
				return nil, validationError(CodeArchiveInvalid, path)
			}
			continue
		}
		if mode&(^mode.Perm()) != 0 || !allowedFile(path) {
			return nil, validationError(CodeMemberForbidden, path)
		}
		limit := int64(MaximumResourceBytes)
		if path == "SKILL.md" {
			manifestCount++
			limit = MaximumManifestBytes
		}
		if file.UncompressedSize64 > uint64(limit) || file.UncompressedSize64 > uint64(MaximumExpandedBytes)-uint64(expanded) {
			return nil, validationError(CodeLimitExceeded, path)
		}
		if file.UncompressedSize64 > uint64(MaximumExpandedBytes)-declaredExpanded {
			return nil, validationError(CodeLimitExceeded, path)
		}
		declaredExpanded += file.UncompressedSize64
		data, readErr := readBoundedMember(file, limit, MaximumExpandedBytes-expanded)
		if readErr != nil {
			if errors.Is(readErr, errMemberLimit) {
				return nil, validationError(CodeLimitExceeded, path)
			}
			return nil, validationError(CodeArchiveInvalid, path)
		}
		expanded += int64(len(data))
		if path == "SKILL.md" || strings.HasPrefix(path, "references/") {
			if !utf8.Valid(data) || bytes.IndexByte(data, 0) >= 0 {
				return nil, validationError(CodeManifestInvalid, path)
			}
		}
		members = append(members, Member{Path: path, data: data})
	}
	if manifestCount != 1 {
		return nil, validationError(CodeManifestInvalid, "")
	}
	sort.Slice(members, func(i, j int) bool { return members[i].Path < members[j].Path })
	var manifestBytes []byte
	resources := make([]Resource, 0, len(members)-1)
	for _, member := range members {
		if member.Path == "SKILL.md" {
			manifestBytes = member.data
			continue
		}
		resources = append(resources, Resource{Path: member.Path, Size: member.Size()})
	}
	manifest, err := parseManifest(manifestBytes, expectedName)
	if err != nil {
		return nil, err
	}
	digest := sha256.Sum256(payload)
	return &Package{
		Manifest: manifest, Digest: "sha256:" + hex.EncodeToString(digest[:]),
		Resources: resources, StoredBytes: int64(len(payload)), ExpandedBytes: expanded,
		members: members,
	}, nil
}

var errMemberLimit = errors.New("member exceeds bounded stream")

func readBoundedMember(file *zip.File, memberLimit, aggregateRemaining int64) ([]byte, error) {
	reader, err := file.Open()
	if err != nil {
		return nil, err
	}
	defer reader.Close()
	limit := min(memberLimit, aggregateRemaining)
	data, err := io.ReadAll(io.LimitReader(reader, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(data)) > limit {
		return nil, errMemberLimit
	}
	var extra [1]byte
	if count, readErr := reader.Read(extra[:]); count != 0 || readErr != io.EOF {
		if readErr == nil || count != 0 {
			return nil, errMemberLimit
		}
		return nil, readErr
	}
	return data, nil
}

func allowedDirectory(path string) bool {
	return path == "references" || path == "assets" || strings.HasPrefix(path, "references/") || strings.HasPrefix(path, "assets/")
}

func allowedFile(path string) bool {
	return path == "SKILL.md" || strings.HasPrefix(path, "references/") || strings.HasPrefix(path, "assets/")
}

func hasPathCollision(paths map[string]bool, path string, directory bool) bool {
	parts := strings.Split(path, "/")
	for index := 1; index < len(parts); index++ {
		if ancestorDirectory, exists := paths[strings.Join(parts[:index], "/")]; exists && !ancestorDirectory {
			return true
		}
	}
	if directory {
		return false
	}
	prefix := path + "/"
	for existing := range paths {
		if strings.HasPrefix(existing, prefix) {
			return true
		}
	}
	return false
}
