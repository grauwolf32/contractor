package contracts

import (
	"regexp"
	"strings"
)

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
		return invalidf("artifact name must be 1 through 128 ASCII letters, digits, dots, underscores or hyphens, starting with a letter or digit")
	}
	return nil
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
		return validateOpaqueID("artifact revision", *r.Revision)
	}
	return nil
}

func (r ArtifactRef) ValidateExact() error {
	if err := r.Validate(); err != nil {
		return err
	}
	if r.Revision == nil {
		return invalidf("artifact revision is required")
	}
	return nil
}

// ValidateAgentSkillRef validates the logical, versionless ref accepted in an
// AgentTemplate. Exact revisions belong to the Run snapshot instead.
func (r ArtifactRef) ValidateAgentSkillRef() error {
	if r.Namespace != AgentSkillNamespace || r.Revision != nil || !validAgentSkillName(r.Name) {
		return invalidf("AgentTemplate skill ref must be versionless skills/<portable-name>")
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
	if err := validateAPIVersion(r.APIVersion); err != nil {
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
	if err := validateAPIVersion(r.APIVersion); err != nil {
		return err
	}
	for _, artifact := range r.Artifacts {
		if err := artifact.Validate(); err != nil {
			return err
		}
		if artifact.Revision != nil {
			return invalidf("listed artifact refs must be versionless")
		}
	}
	return nil
}

func (r ArtifactWriteResult) Validate() error {
	if err := validateAPIVersion(r.APIVersion); err != nil {
		return err
	}
	if err := r.Artifact.ValidateExact(); err != nil {
		return err
	}
	return validateArtifactMetadata(r.MediaType, r.Size)
}

func validateArtifactMetadata(mediaType string, size int64) error {
	typePart, subtypePart, found := strings.Cut(mediaType, "/")
	if !found || typePart == "" || subtypePart == "" ||
		mediaType != strings.ToLower(mediaType) ||
		strings.ContainsAny(mediaType, "; ") || strings.Contains(subtypePart, "/") {
		return invalidf("mediaType must be lowercase type/subtype without parameters")
	}
	if size < 0 {
		return invalidf("artifact size must be non-negative")
	}
	return nil
}
