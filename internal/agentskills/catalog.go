package agentskills

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"

	"github.com/grauwolf32/contractor/internal/artifacts"
)

const (
	SkillNamespace             = "skills"
	MaximumBundledSkills       = 128
	MaximumBundledPackageBytes = 256 << 20

	SeedCreated SeedStatus = "created"
	SeedInSync  SeedStatus = "in_sync"
	SeedDrift   SeedStatus = "seed_drift"
)

type SeedStatus string

type SeedMetadata struct {
	Name   string `json:"name"`
	Digest string `json:"digest"`
	Size   int64  `json:"size"`
}

type seedPackage struct {
	metadata SeedMetadata
	payload  []byte
}

// SeedPlan freezes an all-or-nothing discovery result. Package bytes are kept
// private and are never re-read from the source tree during initialization.
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

type SeedOutcome struct {
	Name          string     `json:"name"`
	Status        SeedStatus `json:"status"`
	BundledDigest string     `json:"bundledDigest"`
	CurrentDigest string     `json:"currentDigest"`
}

type ownerArtifactStore interface {
	Read(context.Context, artifacts.ArtifactRef) (artifacts.ReadResult, error)
	Write(context.Context, artifacts.ArtifactRef, artifacts.Payload, *string) (artifacts.WriteResult, error)
}

type Catalog struct {
	ownerStore func(string) (ownerArtifactStore, error)
}

func NewCatalog(service *artifacts.Service) (*Catalog, error) {
	if service == nil {
		return nil, errors.New("ArtifactService is required")
	}
	return &Catalog{ownerStore: func(ownerID string) (ownerArtifactStore, error) {
		return service.User(ownerID)
	}}, nil
}

// DiscoverBundled packages only immediate skill directories below
// <operatorRoot>/skills. A missing subtree is an empty valid plan.
func DiscoverBundled(operatorRoot string) (*SeedPlan, error) {
	skillsRoot := filepath.Join(filepath.Clean(operatorRoot), SkillNamespace)
	info, err := os.Lstat(skillsRoot)
	if errors.Is(err, os.ErrNotExist) {
		return &SeedPlan{}, nil
	}
	if err != nil || !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return nil, validationError(CodeMemberForbidden, "")
	}
	entries, err := os.ReadDir(skillsRoot)
	if err != nil {
		return nil, validationError(CodeMemberForbidden, "")
	}
	if len(entries) > MaximumBundledSkills {
		return nil, validationError(CodeLimitExceeded, "")
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].Name() < entries[j].Name() })
	plan := &SeedPlan{packages: make([]seedPackage, 0, len(entries))}
	for _, entry := range entries {
		safeName := ""
		if validSkillName(entry.Name()) {
			safeName = entry.Name()
		}
		entryInfo, infoErr := entry.Info()
		if infoErr != nil || !entryInfo.IsDir() || entryInfo.Mode()&os.ModeSymlink != 0 {
			return nil, validationError(CodeMemberForbidden, safeName)
		}
		payload, validated, packageErr := PackageDirectory(filepath.Join(skillsRoot, entry.Name()))
		if packageErr != nil {
			return nil, packageErr
		}
		if int64(len(payload)) > MaximumBundledPackageBytes-plan.totalBytes {
			return nil, validationError(CodeLimitExceeded, safeName)
		}
		plan.totalBytes += int64(len(payload))
		plan.packages = append(plan.packages, seedPackage{
			metadata: SeedMetadata{Name: validated.Manifest.Name, Digest: validated.Digest, Size: int64(len(payload))},
			payload:  append([]byte(nil), payload...),
		})
	}
	return plan, nil
}

// Initialize creates only absent owner bindings. Existing bindings are never
// updated; their exact current bytes and media type classify in_sync or drift.
func (c *Catalog) Initialize(
	ctx context.Context,
	ownerID string,
	plan *SeedPlan,
) ([]SeedOutcome, error) {
	if c == nil || c.ownerStore == nil {
		return nil, errors.New("SkillCatalog is not configured")
	}
	if plan == nil {
		return nil, errors.New("bundled Skill seed plan is required")
	}
	store, err := c.ownerStore(ownerID)
	if err != nil {
		return nil, fmt.Errorf("open owner ArtifactStore: %w", err)
	}
	result := make([]SeedOutcome, 0, len(plan.packages))
	for _, pkg := range plan.packages {
		outcome, initializeErr := initializeSeed(ctx, store, pkg)
		if initializeErr != nil {
			return nil, fmt.Errorf("initialize bundled skill %q: %w", pkg.metadata.Name, initializeErr)
		}
		result = append(result, outcome)
	}
	return result, nil
}

func initializeSeed(
	ctx context.Context,
	store ownerArtifactStore,
	pkg seedPackage,
) (SeedOutcome, error) {
	ref := artifacts.ArtifactRef{Namespace: SkillNamespace, Name: pkg.metadata.Name}
	current, err := store.Read(ctx, ref)
	if err == nil {
		return classifySeed(pkg, current), nil
	}
	if !errors.Is(err, artifacts.ErrArtifactNotFound) {
		return SeedOutcome{}, err
	}
	_, err = store.Write(ctx, ref, artifacts.Payload{MediaType: MediaType, Data: pkg.payload}, nil)
	if err == nil {
		return SeedOutcome{
			Name: pkg.metadata.Name, Status: SeedCreated,
			BundledDigest: pkg.metadata.Digest, CurrentDigest: pkg.metadata.Digest,
		}, nil
	}
	if !errors.Is(err, artifacts.ErrArtifactConflict) {
		return SeedOutcome{}, err
	}
	current, err = store.Read(ctx, ref)
	if err != nil {
		return SeedOutcome{}, err
	}
	return classifySeed(pkg, current), nil
}

func classifySeed(pkg seedPackage, current artifacts.ReadResult) SeedOutcome {
	digest := sha256.Sum256(current.Payload.Data)
	currentDigest := "sha256:" + hex.EncodeToString(digest[:])
	status := SeedDrift
	if current.Payload.MediaType == MediaType && currentDigest == pkg.metadata.Digest {
		status = SeedInSync
	}
	return SeedOutcome{
		Name: pkg.metadata.Name, Status: status,
		BundledDigest: pkg.metadata.Digest, CurrentDigest: currentDigest,
	}
}
