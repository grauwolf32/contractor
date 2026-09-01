package agentskills

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
)

const (
	CodeArtifactNotFound    = "skill_artifact_not_found"
	CodeMediaTypeInvalid    = "skill_media_type_invalid"
	CodeArtifactUnavailable = "skill_artifact_unavailable"
	CodeForkConflict        = "skill_fork_conflict"
	CodeForkFailed          = "skill_fork_failed"

	MaximumTemplateStoredBytes   = 64 << 20
	MaximumTemplateExpandedBytes = 128 << 20
	MaximumRunStoredBytes        = 256 << 20
	MaximumRunExpandedBytes      = 512 << 20
)

// RunSkillError is safe to persist and expose operationally. It contains only
// a stable classification and the validated logical Skill name, never package
// bytes, parser diagnostics, revisions, or host paths.
type RunSkillError struct {
	Code      string
	Name      string
	Retryable bool
}

func (e *RunSkillError) Error() string {
	if e.Name == "" {
		return e.Code
	}
	return fmt.Sprintf("%s: skills/%s", e.Code, e.Name)
}

func runSkillError(code, name string, retryable bool) error {
	return &RunSkillError{Code: code, Name: name, Retryable: retryable}
}

// SelectRunSources records one complete, sorted current-owner outcome without
// reading package bodies. Call it from the same REPEATABLE READ transaction as
// WorkflowRun creation and its idempotency row.
func (c *Catalog) SelectRunSources(
	ctx context.Context,
	ownerID string,
	refs []contracts.ArtifactRef,
) ([]contracts.RunSkillSnapshot, error) {
	if c == nil || c.ownerStore == nil {
		return nil, errors.New("SkillCatalog is not configured")
	}
	if err := validateLogicalRunRefs(refs); err != nil {
		return nil, err
	}
	store, err := c.ownerStore(ownerID)
	if err != nil {
		return nil, runSkillError(CodeArtifactUnavailable, "", true)
	}
	selected := make([]contracts.RunSkillSnapshot, 0, len(refs))
	var total int64
	for _, ref := range refs {
		metadata, readErr := store.Metadata(ctx, artifacts.ArtifactRef(ref))
		if errors.Is(readErr, artifacts.ErrArtifactNotFound) {
			selected = append(selected, contracts.RunSkillSnapshot{Name: ref.Name})
			continue
		}
		if readErr != nil {
			return nil, runSkillError(CodeArtifactUnavailable, ref.Name, true)
		}
		snapshot := contracts.RunSkillSnapshot{
			Name: ref.Name, Source: exactRefPointer(metadata.Ref),
			SourceDigest: metadata.Digest, SourceSize: metadata.Size,
		}
		if err := snapshot.Validate(); err != nil {
			return nil, runSkillError(CodeArtifactUnavailable, ref.Name, true)
		}
		if snapshot.SourceSize > MaximumRunStoredBytes-total {
			return nil, runSkillError(CodeLimitExceeded, ref.Name, false)
		}
		total += snapshot.SourceSize
		selected = append(selected, snapshot)
	}
	return selected, nil
}

// ValidateSelectedLimits applies the stored-byte limit to every retained
// AgentTemplate as well as to the de-duplicated Run union.
func ValidateSelectedLimits(selected []contracts.RunSkillSnapshot, templateSets [][]string) error {
	byName := make(map[string]contracts.RunSkillSnapshot, len(selected))
	var runStored int64
	for _, skill := range selected {
		if err := skill.Validate(); err != nil || skill.Initialized() {
			return runSkillError(CodeArtifactUnavailable, skill.Name, true)
		}
		byName[skill.Name] = skill
		if skill.SourceSize > MaximumRunStoredBytes-runStored {
			return runSkillError(CodeLimitExceeded, skill.Name, false)
		}
		runStored += skill.SourceSize
	}
	for _, names := range templateSets {
		var stored int64
		for _, name := range names {
			skill, ok := byName[name]
			if !ok {
				return runSkillError(CodeArtifactUnavailable, name, true)
			}
			if skill.SourceSize > MaximumTemplateStoredBytes-stored {
				return runSkillError(CodeLimitExceeded, name, false)
			}
			stored += skill.SourceSize
		}
	}
	return nil
}

// PinRunSources retains every exact non-missing owner version before the Run
// creation transaction commits. Missing markers are intentionally durable and
// have nothing to pin.
func (c *Catalog) PinRunSources(
	ctx context.Context,
	ownerID string,
	runID string,
	selected []contracts.RunSkillSnapshot,
) error {
	if c == nil || c.service == nil {
		return errors.New("SkillCatalog is not configured")
	}
	scope, err := artifacts.UserScope(ownerID)
	if err != nil {
		return runSkillError(CodeArtifactUnavailable, "", true)
	}
	for _, skill := range selected {
		if skill.Source == nil {
			continue
		}
		if pinErr := c.service.PinExact(
			ctx, scope, *skill.Source, artifacts.PinRunInput, runID+":skill:"+skill.Name,
		); pinErr != nil {
			return runSkillError(CodeArtifactUnavailable, skill.Name, true)
		}
	}
	return nil
}

// InitializeRun validates only the already selected exact sources, then forks
// the complete set into reserved RunScope bindings. The caller owns the
// transaction so any failure exposes neither partial forks nor snapshot data.
func (c *Catalog) InitializeRun(
	ctx context.Context,
	ownerID string,
	runID string,
	selected []contracts.RunSkillSnapshot,
	templateSets [][]string,
) ([]contracts.RunSkillSnapshot, error) {
	if c == nil || c.ownerStore == nil || c.service == nil {
		return nil, errors.New("SkillCatalog is not configured")
	}
	if err := ValidateSelectedLimits(selected, templateSets); err != nil {
		return nil, err
	}
	store, err := c.ownerStore(ownerID)
	if err != nil {
		return nil, runSkillError(CodeArtifactUnavailable, "", true)
	}
	initialized := make([]contracts.RunSkillSnapshot, len(selected))
	for index, skill := range selected {
		if skill.Source == nil {
			return nil, runSkillError(CodeArtifactNotFound, skill.Name, false)
		}
		read, readErr := store.Read(ctx, artifacts.ArtifactRef(*skill.Source))
		if readErr != nil {
			return nil, runSkillError(CodeArtifactUnavailable, skill.Name, true)
		}
		if read.Payload.MediaType != MediaType {
			return nil, runSkillError(CodeMediaTypeInvalid, skill.Name, false)
		}
		digest := sha256.Sum256(read.Payload.Data)
		actualDigest := "sha256:" + hex.EncodeToString(digest[:])
		if actualDigest != skill.SourceDigest || read.Ref.Revision == nil ||
			*read.Ref.Revision != *skill.Source.Revision {
			return nil, runSkillError(CodeArchiveInvalid, skill.Name, false)
		}
		validated, validationErr := Validate(read.Payload.Data, skill.Name)
		if validationErr != nil {
			code := ErrorCode(validationErr)
			if code == "" {
				code = CodeArchiveInvalid
			}
			return nil, runSkillError(code, skill.Name, false)
		}
		initialized[index] = skill
		initialized[index].PackageDigest = validated.Digest
		initialized[index].ExpandedBytes = validated.ExpandedBytes
	}
	if err := validateInitializedLimits(initialized, templateSets); err != nil {
		return nil, err
	}
	runScope, err := artifacts.RunScope(runID)
	if err != nil {
		return nil, runSkillError(CodeForkFailed, "", true)
	}
	for index := range initialized {
		skill := &initialized[index]
		fork, forkErr := c.service.ForkSkill(ctx, ownerID, *skill.Source, runID, skill.Name)
		if errors.Is(forkErr, artifacts.ErrArtifactConflict) {
			return nil, runSkillError(CodeForkConflict, skill.Name, false)
		}
		if forkErr != nil {
			return nil, runSkillError(CodeForkFailed, skill.Name, true)
		}
		if fork.SourceRef.Revision == nil || *fork.SourceRef.Revision != *skill.Source.Revision ||
			fork.TargetRef.Namespace != SkillNamespace || fork.TargetRef.Name != skill.Name ||
			fork.TargetRef.Revision == nil || fork.MediaType != MediaType || fork.Size != skill.SourceSize {
			return nil, runSkillError(CodeForkConflict, skill.Name, false)
		}
		skill.Artifact = exactRefPointer(fork.TargetRef)
		if pinErr := c.service.PinExact(
			ctx, runScope, *skill.Artifact, artifacts.PinRunInput,
			runID+":skill-fork:"+skill.Name,
		); pinErr != nil {
			return nil, runSkillError(CodeForkFailed, skill.Name, true)
		}
	}
	return initialized, nil
}

func validateLogicalRunRefs(refs []contracts.ArtifactRef) error {
	if len(refs) > contracts.MaxWorkflowRunSkills {
		return runSkillError(CodeLimitExceeded, "", false)
	}
	previous := ""
	for _, ref := range refs {
		if err := ref.ValidateAgentSkillRef(); err != nil {
			return err
		}
		if previous != "" && ref.Name <= previous {
			return fmt.Errorf("Run Skill refs must be sorted and unique")
		}
		previous = ref.Name
	}
	return nil
}

func validateInitializedLimits(skills []contracts.RunSkillSnapshot, templateSets [][]string) error {
	byName := make(map[string]contracts.RunSkillSnapshot, len(skills))
	var runExpanded int64
	for _, skill := range skills {
		byName[skill.Name] = skill
		if skill.ExpandedBytes > MaximumRunExpandedBytes-runExpanded {
			return runSkillError(CodeLimitExceeded, skill.Name, false)
		}
		runExpanded += skill.ExpandedBytes
	}
	for _, names := range templateSets {
		var expanded int64
		for _, name := range names {
			skill := byName[name]
			if skill.ExpandedBytes > MaximumTemplateExpandedBytes-expanded {
				return runSkillError(CodeLimitExceeded, name, false)
			}
			expanded += skill.ExpandedBytes
		}
	}
	return nil
}

func exactRefPointer(ref artifacts.ArtifactRef) *contracts.ArtifactRef {
	copy := contracts.ArtifactRef{Namespace: ref.Namespace, Name: ref.Name}
	if ref.Revision != nil {
		revision := *ref.Revision
		copy.Revision = &revision
	}
	return &copy
}
