package contracts

// ResolvedSkill is the allocation-local projection of an immutable Run Skill
// snapshot (agentskills.RunSkillSnapshot). It deliberately carries no
// UserScope source authority or package bytes: Runtime can read only this
// exact RunScope artifact through the allocation-bound private Artifact API.
type ResolvedSkill struct {
	Name          string      `json:"name" strictjson:"required"`
	Artifact      ArtifactRef `json:"artifact" strictjson:"required"`
	PackageDigest string      `json:"packageDigest" strictjson:"required"`
}

func (s ResolvedSkill) Validate() error {
	logical := ArtifactRef{Namespace: AgentSkillNamespace, Name: s.Name}
	if err := logical.ValidateAgentSkillRef(); err != nil {
		return err
	}
	if err := ValidateExactRunSkillRef(s.Artifact, s.Name, "artifact"); err != nil {
		return err
	}
	return ValidateDigest("resolved Skill packageDigest", s.PackageDigest)
}

// ValidateResolvedSkills enforces the exact, canonical projection selected for
// one AgentTemplate. A non-nil empty slice is mandatory even when the template
// has no Skills so private-wire upgrades fail closed.
func ValidateResolvedSkills(template ResolvedAgentTemplate, skills []ResolvedSkill) error {
	if skills == nil {
		return Invalidf("resolvedSkills is required")
	}
	if len(skills) > MaxAgentTemplateSkills {
		return Invalidf("resolvedSkills exceeds %d entries", MaxAgentTemplateSkills)
	}
	if len(skills) != len(template.Skills) {
		return Invalidf("resolvedSkills must exactly match AgentTemplate skills")
	}
	previous := ""
	for index, skill := range skills {
		if err := skill.Validate(); err != nil {
			return err
		}
		if skill.Name <= previous {
			return Invalidf("resolvedSkills must be sorted and unique")
		}
		if template.Skills[index].Name != skill.Name {
			return Invalidf("resolvedSkills must exactly match AgentTemplate skills")
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

// ValidateExactRunSkillRef requires ref to be the exact skills/<name> revision
// of a Run Skill, as a ResolvedSkill artifact and the source and artifact of
// a Run Skill snapshot are; field names the member in the error.
func ValidateExactRunSkillRef(ref ArtifactRef, name, field string) error {
	if ref.Namespace != AgentSkillNamespace || ref.Name != name {
		return Invalidf("Run Skill %s must identify skills/%s", field, name)
	}
	return ref.ValidateExact()
}
