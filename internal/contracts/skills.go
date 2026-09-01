package contracts

// ResolvedSkill is the allocation-local projection of an immutable Run Skill
// snapshot. It deliberately carries no UserScope source authority or package
// bytes: Runtime can read only this exact RunScope artifact through the
// allocation-bound private Artifact API.
type ResolvedSkill struct {
	Name          string      `json:"name"`
	Artifact      ArtifactRef `json:"artifact"`
	PackageDigest string      `json:"packageDigest"`
}

func (s ResolvedSkill) Validate() error {
	logical := ArtifactRef{Namespace: AgentSkillNamespace, Name: s.Name}
	if err := logical.ValidateAgentSkillRef(); err != nil {
		return err
	}
	if err := validateExactRunSkillRef(s.Artifact, s.Name, "artifact"); err != nil {
		return err
	}
	return validateDigest("resolved Skill packageDigest", s.PackageDigest)
}

// ValidateResolvedSkills enforces the exact, canonical projection selected for
// one AgentTemplate. A non-nil empty slice is mandatory even when the template
// has no Skills so private-wire upgrades fail closed.
func ValidateResolvedSkills(template ResolvedAgentTemplate, skills []ResolvedSkill) error {
	if skills == nil {
		return invalidf("resolvedSkills is required")
	}
	if len(skills) > MaxAgentTemplateSkills {
		return invalidf("resolvedSkills exceeds %d entries", MaxAgentTemplateSkills)
	}
	if len(skills) != len(template.Skills) {
		return invalidf("resolvedSkills must exactly match AgentTemplate skills")
	}
	previous := ""
	for index, skill := range skills {
		if err := skill.Validate(); err != nil {
			return err
		}
		if skill.Name <= previous {
			return invalidf("resolvedSkills must be sorted and unique")
		}
		if template.Skills[index].Name != skill.Name {
			return invalidf("resolvedSkills must exactly match AgentTemplate skills")
		}
		previous = skill.Name
	}
	return nil
}

func CloneResolvedSkills(source []ResolvedSkill) []ResolvedSkill {
	if source == nil {
		return nil
	}
	result := make([]ResolvedSkill, len(source))
	for index, skill := range source {
		result[index] = skill
		if skill.Artifact.Revision != nil {
			revision := *skill.Artifact.Revision
			result[index].Artifact.Revision = &revision
		}
	}
	return result
}

// RunSkillSnapshot is immutable Run provenance. Source is nil only for the
// durable missing marker selected during Run creation. Artifact remains nil
// until the exact package has been validated and forked into RunScope.
type RunSkillSnapshot struct {
	Name          string       `json:"name"`
	Source        *ArtifactRef `json:"source,omitempty"`
	SourceDigest  string       `json:"sourceDigest,omitempty"`
	SourceSize    int64        `json:"sourceSize,omitempty"`
	Artifact      *ArtifactRef `json:"artifact,omitempty"`
	PackageDigest string       `json:"packageDigest,omitempty"`
	ExpandedBytes int64        `json:"expandedBytes,omitempty"`
}

func (s RunSkillSnapshot) Validate() error {
	logical := ArtifactRef{Namespace: AgentSkillNamespace, Name: s.Name}
	if err := logical.ValidateAgentSkillRef(); err != nil {
		return err
	}
	if s.Source == nil {
		if s.SourceDigest != "" || s.SourceSize != 0 || s.Artifact != nil || s.PackageDigest != "" || s.ExpandedBytes != 0 {
			return invalidf("missing Run Skill source marker has unexpected resolved fields")
		}
		return nil
	}
	if err := validateExactRunSkillRef(*s.Source, s.Name, "source"); err != nil {
		return err
	}
	if err := validateDigest("Run Skill sourceDigest", s.SourceDigest); err != nil {
		return err
	}
	if s.SourceSize < 1 || s.SourceSize > 16<<20 {
		return invalidf("Run Skill sourceSize must be between 1 and 16 MiB")
	}
	if s.Artifact == nil {
		if s.PackageDigest != "" || s.ExpandedBytes != 0 {
			return invalidf("uninitialized Run Skill has package result fields")
		}
		return nil
	}
	if err := validateExactRunSkillRef(*s.Artifact, s.Name, "artifact"); err != nil {
		return err
	}
	if err := validateDigest("Run Skill packageDigest", s.PackageDigest); err != nil {
		return err
	}
	if s.PackageDigest != s.SourceDigest {
		return invalidf("Run Skill source and package digests differ")
	}
	if s.ExpandedBytes < 1 || s.ExpandedBytes > 32<<20 {
		return invalidf("Run Skill expandedBytes must be between 1 and 32 MiB")
	}
	return nil
}

func (s RunSkillSnapshot) Initialized() bool { return s.Artifact != nil }

func validateExactRunSkillRef(ref ArtifactRef, name, field string) error {
	if ref.Namespace != AgentSkillNamespace || ref.Name != name {
		return invalidf("Run Skill %s must identify skills/%s", field, name)
	}
	if err := ref.ValidateExact(); err != nil {
		return err
	}
	return nil
}
