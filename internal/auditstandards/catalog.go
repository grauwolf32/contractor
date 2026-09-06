package auditstandards

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
)

type seedPackage struct {
	metadata SeedMetadata
	payload  []byte
}

type SeedPlan struct {
	packages   []seedPackage
	totalBytes int64
}

func (p *SeedPlan) Packages() []SeedMetadata {
	if p == nil {
		return nil
	}
	result := make([]SeedMetadata, len(p.packages))
	for index, pkg := range p.packages {
		result[index] = pkg.metadata
	}
	return result
}

func (p *SeedPlan) TotalBytes() int64 {
	if p == nil {
		return 0
	}
	return p.totalBytes
}

type Catalog struct {
	service *artifacts.Service
}

func NewCatalog(service *artifacts.Service) (*Catalog, error) {
	if service == nil {
		return nil, errors.New("ArtifactService is required")
	}
	return &Catalog{service: service}, nil
}

// DiscoverBundled validates and freezes all immediate package directories
// below <operatorRoot>/audit-standards before any catalog mutation occurs.
func DiscoverBundled(operatorRoot string) (*SeedPlan, error) {
	root := filepath.Join(filepath.Clean(operatorRoot), CatalogNamespace)
	info, err := os.Lstat(root)
	if errors.Is(err, os.ErrNotExist) {
		return &SeedPlan{}, nil
	}
	if err != nil || !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return nil, validationError(CodeMemberForbidden, "")
	}
	entries, err := os.ReadDir(root)
	if err != nil {
		return nil, validationError(CodeLimitExceeded, "")
	}
	packageEntries := make([]os.DirEntry, 0, len(entries))
	for _, entry := range entries {
		// Keep the intentionally empty operator directory in source control
		// without turning the marker into package input. No other root-level
		// file is ignored, so misspelled or stray package content still fails
		// closed.
		if entry.Name() == ".gitkeep" {
			info, infoErr := entry.Info()
			if infoErr != nil || !info.Mode().IsRegular() || info.Mode()&os.ModeSymlink != 0 || info.Size() > 16 {
				return nil, validationError(CodeMemberForbidden, "")
			}
			marker, readErr := os.ReadFile(filepath.Join(root, entry.Name()))
			if readErr != nil || len(bytes.TrimSpace(marker)) != 0 {
				return nil, validationError(CodeMemberForbidden, "")
			}
			continue
		}
		packageEntries = append(packageEntries, entry)
	}
	if len(packageEntries) > MaximumPackages {
		return nil, validationError(CodeLimitExceeded, "")
	}
	sort.Slice(packageEntries, func(i, j int) bool { return packageEntries[i].Name() < packageEntries[j].Name() })
	plan := &SeedPlan{packages: make([]seedPackage, 0, len(packageEntries))}
	seen := make(map[string]struct{}, len(packageEntries))
	for _, entry := range packageEntries {
		if !portableDirectoryName(entry.Name()) || entry.Type()&os.ModeSymlink != 0 {
			return nil, validationError(CodeMemberForbidden, "")
		}
		entryInfo, infoErr := entry.Info()
		if infoErr != nil || !entryInfo.IsDir() || entryInfo.Mode()&os.ModeSymlink != 0 {
			return nil, validationError(CodeMemberForbidden, "")
		}
		payload, pkg, packageErr := PackageDirectory(filepath.Join(root, entry.Name()))
		if packageErr != nil {
			return nil, packageErr
		}
		identity := referenceKey(pkg.Reference())
		if _, duplicate := seen[identity]; duplicate {
			return nil, validationError(CodeIdentityMismatch, ManifestPath)
		}
		seen[identity] = struct{}{}
		if int64(len(payload)) > MaximumCatalogBytes-plan.totalBytes {
			return nil, validationError(CodeLimitExceeded, "")
		}
		plan.totalBytes += int64(len(payload))
		plan.packages = append(plan.packages, seedPackage{
			metadata: SeedMetadata{Reference: pkg.Reference(), Digest: pkg.Digest, Size: int64(len(payload))},
			payload:  append([]byte(nil), payload...),
		})
	}
	sort.Slice(plan.packages, func(i, j int) bool {
		return referenceKey(plan.packages[i].metadata.Reference) < referenceKey(plan.packages[j].metadata.Reference)
	})
	return plan, nil
}

// Initialize is create-only. A package identity can never be rebound to other
// bytes: drift is fatal because existing AuditProfiles name exact versions.
func (c *Catalog) Initialize(ctx context.Context, ownerID string, plan *SeedPlan) ([]SeedOutcome, error) {
	if c == nil || c.service == nil || strings.TrimSpace(ownerID) == "" {
		return nil, errors.New("Audit standard catalog is not configured")
	}
	if plan == nil {
		return nil, errors.New("Audit standard seed plan is required")
	}
	store, err := c.service.User(ownerID)
	if err != nil {
		return nil, err
	}
	outcomes := make([]SeedOutcome, 0, len(plan.packages))
	for _, bundled := range plan.packages {
		ref := contracts.ArtifactRef{
			Namespace: CatalogNamespace,
			Name:      ArtifactName(bundled.metadata.Reference),
		}
		current, readErr := store.Read(ctx, ref)
		if errors.Is(readErr, artifacts.ErrArtifactNotFound) {
			written, writeErr := c.service.WriteAuditStandardPackage(
				ctx, ownerID, ref, artifacts.Payload{MediaType: MediaType, Data: bundled.payload}, nil,
			)
			if writeErr == nil {
				if written.Ref.Revision == nil {
					return nil, errors.New("Audit standard write returned no exact revision")
				}
				outcomes = append(outcomes, SeedOutcome{
					Reference: bundled.metadata.Reference, Status: SeedCreated, Digest: bundled.metadata.Digest,
				})
				continue
			}
			if !errors.Is(writeErr, artifacts.ErrArtifactConflict) {
				return nil, fmt.Errorf("seed Audit standard %s@%s: %w",
					bundled.metadata.Reference.Scheme, bundled.metadata.Reference.Version, writeErr)
			}
			current, readErr = store.Read(ctx, ref)
		}
		if readErr != nil {
			return nil, fmt.Errorf("read Audit standard %s@%s: %w",
				bundled.metadata.Reference.Scheme, bundled.metadata.Reference.Version, readErr)
		}
		if err := verifyCurrent(current, bundled.metadata.Reference, bundled.metadata.Digest); err != nil {
			return nil, err
		}
		outcomes = append(outcomes, SeedOutcome{
			Reference: bundled.metadata.Reference, Status: SeedInSync, Digest: bundled.metadata.Digest,
		})
	}
	return outcomes, nil
}

func (c *Catalog) Resolve(ctx context.Context, ownerID string, ref Reference) (ResolvedPackage, error) {
	if err := validateReference(ref); err != nil {
		return ResolvedPackage{}, err
	}
	store, err := c.service.User(ownerID)
	if err != nil {
		return ResolvedPackage{}, err
	}
	read, err := store.Read(ctx, contracts.ArtifactRef{
		Namespace: CatalogNamespace, Name: ArtifactName(ref),
	})
	if errors.Is(err, artifacts.ErrArtifactNotFound) {
		return ResolvedPackage{}, ErrNotFound
	}
	if err != nil {
		return ResolvedPackage{}, err
	}
	pkg, err := Validate(read.Payload.Data, ref)
	if err != nil || read.Payload.MediaType != MediaType || read.Ref.Revision == nil {
		if err == nil {
			err = ErrDrift
		}
		return ResolvedPackage{}, fmt.Errorf("%w: %s@%s: %v", ErrDrift, ref.Scheme, ref.Version, err)
	}
	return ResolvedPackage{Package: *pkg, Source: ExactPackage{
		Artifact: read.Ref, Digest: pkg.Digest, MediaType: read.Payload.MediaType,
		SizeBytes: int64(len(read.Payload.Data)),
	}}, nil
}

func (c *Catalog) List(ctx context.Context, ownerID string) ([]ResolvedPackage, error) {
	store, err := c.service.User(ownerID)
	if err != nil {
		return nil, err
	}
	namespace := CatalogNamespace
	refs, err := store.List(ctx, &namespace)
	if err != nil {
		return nil, err
	}
	if len(refs) > MaximumPackages {
		return nil, validationError(CodeLimitExceeded, "")
	}
	result := make([]ResolvedPackage, 0, len(refs))
	for _, ref := range refs {
		read, readErr := store.Read(ctx, ref)
		if readErr != nil {
			return nil, readErr
		}
		pkg, validateErr := Validate(read.Payload.Data, Reference{})
		if validateErr != nil || read.Payload.MediaType != MediaType || read.Ref.Revision == nil ||
			ref.Name != ArtifactName(pkgReference(pkg)) {
			return nil, fmt.Errorf("%w: catalog binding %q is invalid", ErrDrift, ref.Name)
		}
		result = append(result, ResolvedPackage{Package: *pkg, Source: ExactPackage{
			Artifact: read.Ref, Digest: pkg.Digest, MediaType: read.Payload.MediaType,
			SizeBytes: int64(len(read.Payload.Data)),
		}})
	}
	sort.Slice(result, func(i, j int) bool {
		return referenceKey(result[i].Package.Reference()) < referenceKey(result[j].Package.Reference())
	})
	return result, nil
}

func (c *Catalog) Pin(
	ctx context.Context,
	ownerID string,
	projectID string,
	auditNamespace string,
	references []Reference,
) ([]PinnedPackage, error) {
	result := make([]PinnedPackage, 0, len(references))
	seen := make(map[string]struct{}, len(references))
	for _, ref := range references {
		key := referenceKey(ref)
		if _, duplicate := seen[key]; duplicate {
			return nil, validationError(CodeIdentityMismatch, ManifestPath)
		}
		seen[key] = struct{}{}
		resolved, err := c.Resolve(ctx, ownerID, ref)
		if err != nil {
			return nil, err
		}
		target := contracts.ArtifactRef{
			Namespace: auditNamespace,
			Name:      "standard-" + strings.TrimPrefix(ArtifactName(ref), "std-"),
		}
		retained, err := c.writeRetained(ctx, projectID, target, resolved)
		if err != nil {
			return nil, err
		}
		result = append(result, PinnedPackage{
			Reference: ref, Title: resolved.Package.Document.Standard.Title,
			Source:  resolved.Package.Document.Standard.Source,
			License: resolved.Package.Document.Standard.License,
			Catalog: resolved.Source, Retained: retained,
		})
	}
	return result, nil
}

// ResolvePinned reads the protected Project-scoped copy named by an Audit
// baseline and revalidates both its bytes and copied provenance. It never
// consults the mutable current catalog binding.
func (c *Catalog) ResolvePinned(
	ctx context.Context,
	projectID string,
	pinned PinnedPackage,
) (ResolvedPackage, error) {
	if c == nil || c.service == nil || strings.TrimSpace(projectID) == "" ||
		ValidatePinnedPackage(pinned) != nil {
		return ResolvedPackage{}, validationError(CodeManifestInvalid, ManifestPath)
	}
	store, err := c.service.Project(projectID)
	if err != nil {
		return ResolvedPackage{}, err
	}
	read, err := store.Read(ctx, pinned.Retained.Artifact)
	if errors.Is(err, artifacts.ErrArtifactNotFound) {
		return ResolvedPackage{}, ErrNotFound
	}
	if err != nil {
		return ResolvedPackage{}, err
	}
	if read.Ref.Revision == nil || pinned.Retained.Artifact.Revision == nil ||
		*read.Ref.Revision != *pinned.Retained.Artifact.Revision ||
		read.Payload.MediaType != pinned.Retained.MediaType ||
		int64(len(read.Payload.Data)) != pinned.Retained.SizeBytes ||
		digest(read.Payload.Data) != pinned.Retained.Digest {
		return ResolvedPackage{}, fmt.Errorf("%w: retained Audit standard identity", ErrDrift)
	}
	pkg, err := ValidateRetainedPayload(read.Payload.Data, pinned)
	if err != nil {
		return ResolvedPackage{}, err
	}
	return ResolvedPackage{Package: *pkg, Source: pinned.Retained}, nil
}

func (c *Catalog) writeRetained(
	ctx context.Context,
	projectID string,
	target contracts.ArtifactRef,
	resolved ResolvedPackage,
) (ExactPackage, error) {
	payload := artifacts.Payload{MediaType: MediaType, Data: resolved.Package.Payload()}
	written, err := c.service.WriteAuditArtifact(ctx, projectID, target, payload)
	if err == nil {
		return ExactPackage{Artifact: written.Ref, Digest: resolved.Package.Digest,
			MediaType: written.MediaType, SizeBytes: written.Size}, nil
	}
	if !errors.Is(err, artifacts.ErrArtifactConflict) {
		return ExactPackage{}, err
	}
	store, openErr := c.service.Project(projectID)
	if openErr != nil {
		return ExactPackage{}, openErr
	}
	current, readErr := store.Read(ctx, target)
	if readErr != nil || current.Ref.Revision == nil || current.Payload.MediaType != MediaType ||
		digest(current.Payload.Data) != resolved.Package.Digest {
		return ExactPackage{}, fmt.Errorf("%w: retained Audit standard collision", ErrDrift)
	}
	return ExactPackage{Artifact: current.Ref, Digest: resolved.Package.Digest,
		MediaType: current.Payload.MediaType, SizeBytes: int64(len(current.Payload.Data))}, nil
}

func Projection(resolved ResolvedPackage, detail bool) PackageProjection {
	document := resolved.Package.Document
	result := PackageProjection{
		Reference: document.Standard.Reference(), Title: document.Standard.Title,
		Description: document.Standard.Description, Source: document.Standard.Source,
		License: document.Standard.License, Digest: resolved.Package.Digest,
		Artifact: resolved.Source.Artifact, EntryCount: len(document.Entries),
		MappingCount: len(document.Mappings), EvidenceContractCount: len(document.EvidenceContracts),
	}
	if !detail || document.Standard.License.Disclosure == DisclosureMetadata {
		return result
	}
	result.Entries = make([]EntryProjection, len(document.Entries))
	for index, entry := range document.Entries {
		result.Entries[index] = EntryProjection{ID: entry.ID, Kind: entry.Kind}
		if document.Standard.License.Disclosure == DisclosureFull {
			applicability := entry.Applicability
			contract := entry.EvidenceContract
			result.Entries[index].Title = entry.Title
			result.Entries[index].Statement = entry.Statement
			result.Entries[index].Level = entry.Level
			result.Entries[index].Applicability = &applicability
			result.Entries[index].AllowedMethods = append([]string{}, entry.AllowedMethods...)
			result.Entries[index].EvidenceContract = &contract
		}
	}
	if document.Standard.License.Disclosure == DisclosureFull {
		result.EvidenceContracts = cloneContracts(document.EvidenceContracts)
		result.Mappings = cloneMappings(document.Mappings)
	}
	return result
}

func ArtifactName(ref Reference) string {
	sum := sha256.Sum256([]byte(referenceKey(ref)))
	return "std-" + hex.EncodeToString(sum[:16])
}

func verifyCurrent(current artifacts.ReadResult, expected Reference, expectedDigest string) error {
	if current.Ref.Revision == nil || current.Payload.MediaType != MediaType || digest(current.Payload.Data) != expectedDigest {
		return fmt.Errorf("%w: %s@%s", ErrDrift, expected.Scheme, expected.Version)
	}
	if _, err := Validate(current.Payload.Data, expected); err != nil {
		return fmt.Errorf("%w: %s@%s", ErrDrift, expected.Scheme, expected.Version)
	}
	return nil
}

func validateReference(ref Reference) error {
	if !schemePattern.MatchString(ref.Scheme) || !versionPattern.MatchString(ref.Version) {
		return validationError(CodeIdentityMismatch, ManifestPath)
	}
	return nil
}

func pkgReference(pkg *Package) Reference {
	if pkg == nil {
		return Reference{}
	}
	return pkg.Reference()
}

func referenceKey(ref Reference) string { return ref.Scheme + "\x00" + ref.Version }

func digest(data []byte) string {
	sum := sha256.Sum256(data)
	return "sha256:" + hex.EncodeToString(sum[:])
}

func portableDirectoryName(value string) bool {
	if len(value) == 0 || len(value) > 128 || value == "." || value == ".." {
		return false
	}
	for _, character := range []byte(value) {
		if character >= 'a' && character <= 'z' || character >= '0' && character <= '9' ||
			character == '.' || character == '_' || character == '-' {
			continue
		}
		return false
	}
	return true
}

func cloneContracts(source []EvidenceContract) []EvidenceContract {
	result := make([]EvidenceContract, len(source))
	for index, contract := range source {
		contract.Assessments = append([]string{}, contract.Assessments...)
		contract.EvidenceKinds = append([]string{}, contract.EvidenceKinds...)
		result[index] = contract
	}
	return result
}

func cloneMappings(source []Mapping) []Mapping {
	result := make([]Mapping, len(source))
	for index, mapping := range source {
		mapping.EntryIDs = append([]string{}, mapping.EntryIDs...)
		result[index] = mapping
	}
	return result
}
