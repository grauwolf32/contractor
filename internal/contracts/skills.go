package contracts

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
