package control

import (
	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/pagelimit"
)

// SharedConfig is the immutable startup policy fetched before registration.
// Only cross-component policy belongs here; allocation settings remain separate.
type SharedConfig struct {
	APIVersion string       `json:"apiVersion" strictjson:"required"`
	Version    int          `json:"version" strictjson:"required"`
	Limits     SharedLimits `json:"limits" strictjson:"required"`
	Digest     string       `json:"digest" strictjson:"required"`
}

type SharedLimits struct {
	ArtifactPayloadMaxBytes int64 `json:"artifactPayloadMaxBytes" strictjson:"required"`
	PublicPageMaxItems      int   `json:"publicPageMaxItems" strictjson:"required"`
	ArtifactListMaxItems    int   `json:"artifactListMaxItems" strictjson:"required"`
}

// DefaultSharedConfig derives policy from the Server-owned definitions.
func DefaultSharedConfig() (SharedConfig, error) {
	c := SharedConfig{APIVersion: contracts.APIVersion, Version: 1,
		Limits: SharedLimits{
			ArtifactPayloadMaxBytes: contracts.MaxArtifactPayloadBytes,
			PublicPageMaxItems:      pagelimit.MaxRows,
			ArtifactListMaxItems:    contracts.MaxArtifactListItems,
		}}
	var err error
	c.Digest, err = c.digest()
	if err != nil {
		return SharedConfig{}, err
	}
	return c, c.Validate()
}

func (c SharedConfig) digest() (string, error) {
	// Hash every field of the typed limits projection in canonical key order.
	return contentdigest.CanonicalJSON(struct {
		Limits  SharedLimits `json:"limits"`
		Version int          `json:"version"`
	}{c.Limits, c.Version})
}

func (c SharedConfig) Validate() error {
	if err := contracts.ValidateAPIVersion(c.APIVersion); err != nil {
		return err
	}
	if c.Version != 1 {
		return contracts.Invalidf("unsupported shared config version")
	}
	for _, limit := range []int64{c.Limits.ArtifactPayloadMaxBytes, int64(c.Limits.PublicPageMaxItems), int64(c.Limits.ArtifactListMaxItems)} {
		if limit <= 0 || limit > contracts.MaxExactInteger {
			return contracts.Invalidf("invalid shared config limit")
		}
	}
	digest, err := c.digest()
	if err != nil {
		return err
	}
	if c.Digest != digest {
		return contracts.Invalidf("invalid shared config digest")
	}
	return nil
}
