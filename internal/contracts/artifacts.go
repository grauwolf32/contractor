package contracts

import "strings"

type ArtifactRef struct {
	Namespace string  `json:"namespace"`
	Name      string  `json:"name"`
	Revision  *string `json:"revision,omitempty"`
}

func (r ArtifactRef) Validate() error {
	if strings.TrimSpace(r.Namespace) == "" || strings.Contains(r.Namespace, "/") {
		return invalidf("artifact namespace must be non-empty and contain no slash")
	}
	if strings.TrimSpace(r.Name) == "" || strings.Contains(r.Name, "/") {
		return invalidf("artifact name must be non-empty and contain no slash")
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
