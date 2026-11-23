package agentskills

import "github.com/grauwolf32/contractor/internal/contracts"

// RunSkillSnapshot is immutable Run provenance. Source is nil only for the
// durable missing marker selected during Run creation. Artifact remains nil
// until the exact package has been validated and forked into RunScope. An
// allocation receives only its contracts.ResolvedSkill projection.
type RunSkillSnapshot struct {
	Name          string                 `json:"name"`
	Source        *contracts.ArtifactRef `json:"source,omitempty"`
	SourceDigest  string                 `json:"sourceDigest,omitempty"`
	SourceSize    int64                  `json:"sourceSize,omitempty"`
	Artifact      *contracts.ArtifactRef `json:"artifact,omitempty"`
	PackageDigest string                 `json:"packageDigest,omitempty"`
	ExpandedBytes int64                  `json:"expandedBytes,omitempty"`
}

func (s RunSkillSnapshot) Validate() error {
	logical := contracts.ArtifactRef{Namespace: contracts.AgentSkillNamespace, Name: s.Name}
	if err := logical.ValidateAgentSkillRef(); err != nil {
		return err
	}
	if s.Source == nil {
		if s.SourceDigest != "" || s.SourceSize != 0 || s.Artifact != nil || s.PackageDigest != "" || s.ExpandedBytes != 0 {
			return contracts.Invalidf("missing Run Skill source marker has unexpected resolved fields")
		}
		return nil
	}
	if err := contracts.ValidateExactRunSkillRef(*s.Source, s.Name, "source"); err != nil {
		return err
	}
	if err := contracts.ValidateDigest("Run Skill sourceDigest", s.SourceDigest); err != nil {
		return err
	}
	if s.SourceSize < 1 || s.SourceSize > MaximumArchiveBytes {
		return contracts.Invalidf("Run Skill sourceSize must be between 1 and 16 MiB")
	}
	if s.Artifact == nil {
		if s.PackageDigest != "" || s.ExpandedBytes != 0 {
			return contracts.Invalidf("uninitialized Run Skill has package result fields")
		}
		return nil
	}
	if err := contracts.ValidateExactRunSkillRef(*s.Artifact, s.Name, "artifact"); err != nil {
		return err
	}
	if err := contracts.ValidateDigest("Run Skill packageDigest", s.PackageDigest); err != nil {
		return err
	}
	if s.PackageDigest != s.SourceDigest {
		return contracts.Invalidf("Run Skill source and package digests differ")
	}
	if s.ExpandedBytes < 1 || s.ExpandedBytes > MaximumExpandedBytes {
		return contracts.Invalidf("Run Skill expandedBytes must be between 1 and 32 MiB")
	}
	return nil
}

func (s RunSkillSnapshot) Initialized() bool { return s.Artifact != nil }

// Clone returns a copy that shares no ref or revision with s.
func (s RunSkillSnapshot) Clone() RunSkillSnapshot {
	if s.Source != nil {
		source := s.Source.Clone()
		s.Source = &source
	}
	if s.Artifact != nil {
		artifact := s.Artifact.Clone()
		s.Artifact = &artifact
	}
	return s
}

// CloneRunSkills returns a copy of skills that shares no ref or revision with
// the source; nil stays nil.
func CloneRunSkills(skills []RunSkillSnapshot) []RunSkillSnapshot {
	if skills == nil {
		return nil
	}
	result := make([]RunSkillSnapshot, len(skills))
	for index, skill := range skills {
		result[index] = skill.Clone()
	}
	return result
}
