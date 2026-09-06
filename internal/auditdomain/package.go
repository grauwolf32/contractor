package auditdomain

import (
	"archive/zip"
	"bytes"
	"errors"
	"hash/crc32"
	"io"
	"path"
	"sort"
	"strings"
	"unicode/utf8"
)

// BuildPackage creates a byte-identical ZIP for identical logical inputs. The
// entry point is required only for an openapi-source package.
func BuildPackage(packageID string, kind PackageKind, entryPoint string, inputs []PackageInput) ([]byte, *Package, error) {
	if err := validateIdentifier(packageID, "package_id"); err != nil || !validPackageKind(kind) {
		return nil, nil, invalid(CodePackageInvalid, "manifest")
	}
	if len(inputs) == 0 || len(inputs) > MaximumMembers {
		return nil, nil, invalid(CodeLimitExceeded, "members")
	}
	ordered := make([]PackageInput, len(inputs))
	for index, input := range inputs {
		ordered[index] = PackageInput{
			ID: input.ID, Path: input.Path, MediaType: input.MediaType,
			Data: append([]byte(nil), input.Data...),
		}
	}
	sort.Slice(ordered, func(i, j int) bool { return ordered[i].Path < ordered[j].Path })
	manifest := PackageManifest{
		Schema: PackageSchema, PackageID: packageID, Kind: kind,
		EntryPoint: entryPoint, Members: make([]PackageMemberManifest, 0, len(ordered)),
	}
	seenIDs := make(map[string]struct{}, len(ordered))
	seenPaths := make(map[string]struct{}, len(ordered))
	var expanded int64
	for _, input := range ordered {
		if err := validateIdentifier(input.ID, "members.id"); err != nil {
			return nil, nil, invalid(CodePackageInvalid, "members.id")
		}
		validatedPath, err := validatePackagePath(input.Path)
		if err != nil || validatedPath == "manifest.json" {
			return nil, nil, invalid(CodePackagePathInvalid, "members.path")
		}
		if _, duplicate := seenIDs[input.ID]; duplicate {
			return nil, nil, invalid(CodePackageInvalid, "members.id")
		}
		if _, duplicate := seenPaths[validatedPath]; duplicate {
			return nil, nil, invalid(CodePackagePathInvalid, "members.path")
		}
		seenIDs[input.ID] = struct{}{}
		seenPaths[validatedPath] = struct{}{}
		if !validMediaType(input.MediaType) {
			return nil, nil, invalid(CodePackageInvalid, "members.media_type")
		}
		if len(input.Data) > MaximumMemberBytes || int64(len(input.Data)) > int64(MaximumExpandedBytes)-expanded {
			return nil, nil, invalid(CodeLimitExceeded, "members")
		}
		expanded += int64(len(input.Data))
		manifest.Members = append(manifest.Members, PackageMemberManifest{
			ID: input.ID, Path: validatedPath, MediaType: normalizedMediaType(input.MediaType),
			Size: int64(len(input.Data)), Digest: digestBytes(input.Data),
		})
	}
	if err := validateEntryPoint(manifest, seenPaths); err != nil {
		return nil, nil, err
	}
	manifestBytes, err := canonicalJSON(manifest)
	if err != nil || len(manifestBytes) > MaximumManifestBytes || int64(len(manifestBytes)) > int64(MaximumExpandedBytes)-expanded {
		return nil, nil, invalid(CodeLimitExceeded, "manifest.json")
	}

	var buffer bytes.Buffer
	writer := zip.NewWriter(&buffer)
	if err := writeCanonicalZIPMember(writer, "manifest.json", manifestBytes); err != nil {
		return nil, nil, invalid(CodePackageInvalid, "archive")
	}
	for _, input := range ordered {
		if err := writeCanonicalZIPMember(writer, input.Path, input.Data); err != nil {
			_ = writer.Close()
			return nil, nil, invalid(CodePackageInvalid, "archive")
		}
	}
	if err := writer.Close(); err != nil {
		return nil, nil, invalid(CodePackageInvalid, "archive")
	}
	if buffer.Len() > MaximumArchiveBytes {
		return nil, nil, invalid(CodeLimitExceeded, "archive")
	}
	payload := append([]byte(nil), buffer.Bytes()...)
	validated, err := ValidatePackage(payload)
	if err != nil {
		return nil, nil, err
	}
	return payload, validated, nil
}

// ValidatePackage validates and fully materializes a bounded Audit package.
// No caller ever receives an untrusted zip.File handle.
func ValidatePackage(payload []byte) (*Package, error) {
	if len(payload) == 0 || len(payload) > MaximumArchiveBytes {
		return nil, invalid(CodeLimitExceeded, "archive")
	}
	reader, err := zip.NewReader(bytes.NewReader(payload), int64(len(payload)))
	if err != nil {
		return nil, invalid(CodePackageInvalid, "archive")
	}
	if len(reader.File) == 0 || len(reader.File) > MaximumMembers+1 {
		return nil, invalid(CodeLimitExceeded, "members")
	}

	seen := make(map[string]struct{}, len(reader.File))
	rawMembers := make(map[string][]byte, len(reader.File))
	var expanded int64
	manifestCount := 0
	for _, file := range reader.File {
		memberPath, pathErr := validatePackagePath(strings.TrimSuffix(file.Name, "/"))
		if pathErr != nil || strings.HasSuffix(file.Name, "/") {
			return nil, invalid(CodePackagePathInvalid, "archive")
		}
		if _, duplicate := seen[memberPath]; duplicate || hasPackagePathCollision(seen, memberPath) {
			return nil, invalid(CodePackagePathInvalid, "archive")
		}
		seen[memberPath] = struct{}{}
		if file.Flags&1 != 0 || file.Method != zip.Store && file.Method != zip.Deflate {
			return nil, invalid(CodePackageInvalid, "archive")
		}
		mode := file.Mode()
		if !mode.IsRegular() || mode.Perm()&0o111 != 0 {
			return nil, invalid(CodeMemberForbidden, "archive")
		}
		limit := int64(MaximumMemberBytes)
		if memberPath == "manifest.json" {
			manifestCount++
			limit = MaximumManifestBytes
		}
		if file.UncompressedSize64 > uint64(limit) || file.UncompressedSize64 > uint64(int64(MaximumExpandedBytes)-expanded) {
			return nil, invalid(CodeLimitExceeded, "members")
		}
		data, readErr := readPackageMember(file, min(limit, int64(MaximumExpandedBytes)-expanded))
		if readErr != nil {
			if errors.Is(readErr, errPackageMemberLimit) {
				return nil, invalid(CodeLimitExceeded, "members")
			}
			return nil, invalid(CodePackageInvalid, "archive")
		}
		expanded += int64(len(data))
		rawMembers[memberPath] = data
	}
	if manifestCount != 1 {
		return nil, invalid(CodePackageInvalid, "manifest.json")
	}

	var manifest PackageManifest
	if _, err := decodeStrictJSON(rawMembers["manifest.json"], &manifest); err != nil {
		return nil, invalid(CodePackageInvalid, "manifest.json")
	}
	canonical, err := canonicalJSON(manifest)
	if err != nil || !bytes.Equal(canonical, rawMembers["manifest.json"]) {
		return nil, invalid(CodePackageInvalid, "manifest.json")
	}
	if err := validatePackageManifest(manifest, rawMembers); err != nil {
		return nil, err
	}

	members := make([]Member, 0, len(manifest.Members))
	for _, metadata := range manifest.Members {
		members = append(members, Member{
			metadata: metadata,
			data:     append([]byte(nil), rawMembers[metadata.Path]...),
		})
	}
	return &Package{
		Manifest: manifest, Digest: digestBytes(payload), StoredBytes: int64(len(payload)),
		ExpandedBytes: expanded, members: members,
	}, nil
}

func validatePackageManifest(manifest PackageManifest, rawMembers map[string][]byte) error {
	if manifest.Schema != PackageSchema {
		return invalid(CodeSchemaUnsupported, "manifest.schema")
	}
	if err := validateIdentifier(manifest.PackageID, "manifest.package_id"); err != nil || !validPackageKind(manifest.Kind) {
		return invalid(CodePackageInvalid, "manifest")
	}
	if len(manifest.Members) == 0 || len(manifest.Members) > MaximumMembers || len(rawMembers) != len(manifest.Members)+1 {
		return invalid(CodePackageInvalid, "manifest.members")
	}
	seenIDs := make(map[string]struct{}, len(manifest.Members))
	seenPaths := make(map[string]struct{}, len(manifest.Members))
	previousPath := ""
	for _, member := range manifest.Members {
		if err := validateIdentifier(member.ID, "manifest.members.id"); err != nil {
			return invalid(CodePackageInvalid, "manifest.members.id")
		}
		memberPath, err := validatePackagePath(member.Path)
		if err != nil || memberPath == "manifest.json" || memberPath <= previousPath {
			return invalid(CodePackagePathInvalid, "manifest.members.path")
		}
		previousPath = memberPath
		if _, duplicate := seenIDs[member.ID]; duplicate {
			return invalid(CodePackageInvalid, "manifest.members.id")
		}
		if _, duplicate := seenPaths[memberPath]; duplicate {
			return invalid(CodePackagePathInvalid, "manifest.members.path")
		}
		seenIDs[member.ID] = struct{}{}
		seenPaths[memberPath] = struct{}{}
		if !validMediaType(member.MediaType) || member.MediaType != normalizedMediaType(member.MediaType) || member.Size < 0 || member.Size > MaximumMemberBytes || !validDigest(member.Digest) {
			return invalid(CodePackageInvalid, "manifest.members")
		}
		data, exists := rawMembers[memberPath]
		if !exists || int64(len(data)) != member.Size || digestBytes(data) != member.Digest {
			return invalid(CodeDigestMismatch, "manifest.members")
		}
	}
	return validateEntryPoint(manifest, seenPaths)
}

func validateEntryPoint(manifest PackageManifest, paths map[string]struct{}) error {
	if manifest.Kind == PackageKindOpenAPISource {
		entry, err := validatePackagePath(manifest.EntryPoint)
		if err != nil {
			return invalid(CodePackageInvalid, "manifest.entrypoint")
		}
		if _, exists := paths[entry]; !exists {
			return invalid(CodeReferenceInvalid, "manifest.entrypoint")
		}
		return nil
	}
	if manifest.EntryPoint != "" {
		return invalid(CodePackageInvalid, "manifest.entrypoint")
	}
	return nil
}

func validPackageKind(kind PackageKind) bool {
	switch kind {
	case PackageKindWorklist, PackageKindTask, PackageKindTaskSet, PackageKindExecution, PackageKindCheckResults,
		PackageKindFindingProposal, PackageKindEvidence, PackageKindCoverage, PackageKindOpenAPISource:
		return true
	default:
		return false
	}
}

func validMediaType(value string) bool {
	normalized := normalizedMediaType(value)
	if normalized == "" || len(normalized) > 127 || strings.Count(normalized, "/") != 1 || strings.Contains(normalized, "*") {
		return false
	}
	for _, character := range []byte(normalized) {
		if character <= 0x20 || character >= 0x7f {
			return false
		}
	}
	return true
}

func validatePackagePath(raw string) (string, error) {
	if raw == "" || len(raw) > MaximumPathBytes || !utf8.ValidString(raw) || strings.Contains(raw, "\\") || strings.HasPrefix(raw, "/") || path.Clean(raw) != raw {
		return "", invalid(CodePackagePathInvalid, "path")
	}
	parts := strings.Split(raw, "/")
	if len(parts) > MaximumPathComponents {
		return "", invalid(CodePackagePathInvalid, "path")
	}
	for _, part := range parts {
		if part == "" || part == "." || part == ".." || len([]byte(part)) > 128 {
			return "", invalid(CodePackagePathInvalid, "path")
		}
		for _, character := range []byte(part) {
			if character >= 'A' && character <= 'Z' || character >= 'a' && character <= 'z' || character >= '0' && character <= '9' || character == '.' || character == '_' || character == '-' {
				continue
			}
			return "", invalid(CodePackagePathInvalid, "path")
		}
	}
	return raw, nil
}

func hasPackagePathCollision(paths map[string]struct{}, candidate string) bool {
	parts := strings.Split(candidate, "/")
	for index := 1; index < len(parts); index++ {
		if _, exists := paths[strings.Join(parts[:index], "/")]; exists {
			return true
		}
	}
	prefix := candidate + "/"
	for existing := range paths {
		if strings.HasPrefix(existing, prefix) {
			return true
		}
	}
	return false
}

func writeCanonicalZIPMember(writer *zip.Writer, memberPath string, data []byte) error {
	header := &zip.FileHeader{
		Name: memberPath, Method: zip.Store, CRC32: crc32.ChecksumIEEE(data),
		CompressedSize64: uint64(len(data)), UncompressedSize64: uint64(len(data)),
		CompressedSize: uint32(len(data)), UncompressedSize: uint32(len(data)),
		CreatorVersion: 3<<8 | 20, ReaderVersion: 20,
		ModifiedDate: 1<<5 | 1, ModifiedTime: 0,
		ExternalAttrs: uint32(0100644) << 16,
	}
	entry, err := writer.CreateRaw(header)
	if err != nil {
		return err
	}
	_, err = entry.Write(data)
	return err
}

var errPackageMemberLimit = errors.New("package member exceeds limit")

func readPackageMember(file *zip.File, limit int64) ([]byte, error) {
	reader, err := file.Open()
	if err != nil {
		return nil, err
	}
	defer reader.Close()
	data, err := io.ReadAll(io.LimitReader(reader, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(data)) > limit {
		return nil, errPackageMemberLimit
	}
	var extra [1]byte
	if count, readErr := reader.Read(extra[:]); count != 0 || !errors.Is(readErr, io.EOF) {
		if readErr == nil || count != 0 {
			return nil, errPackageMemberLimit
		}
		return nil, readErr
	}
	return data, nil
}
