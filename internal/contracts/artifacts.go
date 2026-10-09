package contracts

import "regexp"

const (
	AgentSkillNamespace    = "skills"
	MaxAgentTemplateSkills = 32
	MaxWorkflowRunSkills   = 128
)

// ArtifactNamePattern is shared by namespaces and logical binding names.
const ArtifactNamePattern = `^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$`

var artifactNamePattern = regexp.MustCompile(ArtifactNamePattern)

func ValidateArtifactName(value string) error {
	if !artifactNamePattern.MatchString(value) {
		return Invalidf("artifact name must be 1 through 128 ASCII letters, digits, dots, underscores or hyphens, starting with a letter or digit")
	}
	return nil
}

// AcceptsMediaType reports whether an artifact slot accepting the given media
// types admits actual; a slot declaring */* admits every media type.
func AcceptsMediaType(accepted []string, actual string) bool {
	for _, mediaType := range accepted {
		if mediaType == "*/*" || mediaType == actual {
			return true
		}
	}
	return false
}

type ArtifactRef struct {
	Namespace string  `json:"namespace"`
	Name      string  `json:"name"`
	Revision  *string `json:"revision,omitempty"`
}

func (r ArtifactRef) Validate() error {
	if err := ValidateArtifactName(r.Namespace); err != nil {
		return err
	}
	if err := ValidateArtifactName(r.Name); err != nil {
		return err
	}
	if r.Revision != nil {
		return ValidateOpaqueID("artifact revision", *r.Revision)
	}
	return nil
}

func (r ArtifactRef) ValidateExact() error {
	if err := r.Validate(); err != nil {
		return err
	}
	if r.Revision == nil {
		return Invalidf("artifact revision is required")
	}
	return nil
}

// Clone returns a copy that does not share the Revision pointer.
func (r ArtifactRef) Clone() ArtifactRef {
	if r.Revision != nil {
		revision := *r.Revision
		r.Revision = &revision
	}
	return r
}

// SameExact reports whether both refs name the same pinned revision. A ref
// without a revision never matches.
func (r ArtifactRef) SameExact(other ArtifactRef) bool {
	return r.Namespace == other.Namespace && r.Name == other.Name &&
		r.Revision != nil && other.Revision != nil && *r.Revision == *other.Revision
}

// Equal reports whether both refs have the same namespace, name and optional
// revision value.
func (r ArtifactRef) Equal(other ArtifactRef) bool {
	if r.Namespace != other.Namespace || r.Name != other.Name || (r.Revision == nil) != (other.Revision == nil) {
		return false
	}
	return r.Revision == nil || *r.Revision == *other.Revision
}

// CloneArtifactRefs returns a never-nil copy of refs whose values share no
// revision pointers with the source.
func CloneArtifactRefs(refs map[string]ArtifactRef) map[string]ArtifactRef {
	result := make(map[string]ArtifactRef, len(refs))
	for name, ref := range refs {
		result[name] = ref.Clone()
	}
	return result
}

// Key returns a map key unique per namespace, name and revision. An absent
// revision keys like an empty one.
func (r ArtifactRef) Key() string {
	revision := ""
	if r.Revision != nil {
		revision = *r.Revision
	}
	return r.Namespace + "\x00" + r.Name + "\x00" + revision
}

// ValidateAgentSkillRef validates the logical, versionless ref accepted in an
// AgentTemplate. Exact revisions belong to the Run snapshot instead.
func (r ArtifactRef) ValidateAgentSkillRef() error {
	if r.Namespace != AgentSkillNamespace || r.Revision != nil || !validAgentSkillName(r.Name) {
		return Invalidf("AgentTemplate skill ref must be versionless skills/<portable-name>")
	}
	return nil
}

func validAgentSkillName(value string) bool {
	if len(value) < 1 || len(value) > 64 || value[0] == '-' || value[len(value)-1] == '-' {
		return false
	}
	previousHyphen := false
	for _, character := range []byte(value) {
		if character == '-' {
			if previousHyphen {
				return false
			}
			previousHyphen = true
			continue
		}
		previousHyphen = false
		if (character < 'a' || character > 'z') && (character < '0' || character > '9') {
			return false
		}
	}
	return true
}

type ArtifactReadResult struct {
	APIVersion string      `json:"apiVersion"`
	Artifact   ArtifactRef `json:"artifact"`
	MediaType  string      `json:"mediaType"`
	Size       int64       `json:"size"`
}

func (r ArtifactReadResult) Validate() error {
	if err := ValidateAPIVersion(r.APIVersion); err != nil {
		return err
	}
	if err := r.Artifact.ValidateExact(); err != nil {
		return err
	}
	return validateArtifactMetadata(r.MediaType, r.Size)
}

type ArtifactWriteResult struct {
	APIVersion string      `json:"apiVersion"`
	Artifact   ArtifactRef `json:"artifact"`
	MediaType  string      `json:"mediaType"`
	Size       int64       `json:"size"`
}

type ArtifactListResult struct {
	APIVersion string        `json:"apiVersion"`
	Artifacts  []ArtifactRef `json:"artifacts"`
}

func (r ArtifactListResult) Validate() error {
	if err := ValidateAPIVersion(r.APIVersion); err != nil {
		return err
	}
	for _, artifact := range r.Artifacts {
		if err := artifact.Validate(); err != nil {
			return err
		}
		if artifact.Revision != nil {
			return Invalidf("listed artifact refs must be versionless")
		}
	}
	return nil
}

func (r ArtifactWriteResult) Validate() error {
	if err := ValidateAPIVersion(r.APIVersion); err != nil {
		return err
	}
	if err := r.Artifact.ValidateExact(); err != nil {
		return err
	}
	return validateArtifactMetadata(r.MediaType, r.Size)
}

func validateArtifactMetadata(mediaType string, size int64) error {
	if !ValidMediaType(mediaType) {
		return Invalidf("mediaType must be lowercase type/subtype without parameters")
	}
	if size < 0 {
		return Invalidf("artifact size must be non-negative")
	}
	return nil
}
