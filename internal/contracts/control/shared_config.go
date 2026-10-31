package control

import (
	"fmt"
	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/contracts"
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
}

// DefaultSharedConfig derives policy from the Server-owned definitions.
func DefaultSharedConfig() SharedConfig {
	c := SharedConfig{APIVersion: contracts.APIVersion, Version: 1,
		Limits: SharedLimits{ArtifactPayloadMaxBytes: contracts.MaxArtifactPayloadBytes}}
	c.Digest = c.digest()
	return c
}

func (c SharedConfig) digest() string {
	// The projection has only exact integers; this fixed canonical encoding avoids
	// a second serializer/digest convention for this small wire document.
	return contentdigest.Bytes([]byte(fmt.Sprintf(`{"limits":{"artifactPayloadMaxBytes":%d},"version":%d}`, c.Limits.ArtifactPayloadMaxBytes, c.Version)))
}

func (c SharedConfig) Validate() error {
	if err := contracts.ValidateAPIVersion(c.APIVersion); err != nil {
		return err
	}
	if c.Version != 1 {
		return contracts.Invalidf("unsupported shared config version")
	}
	if c.Limits.ArtifactPayloadMaxBytes <= 0 || c.Limits.ArtifactPayloadMaxBytes > contracts.MaxExactInteger {
		return contracts.Invalidf("invalid shared Artifact payload limit")
	}
	if c.Digest != c.digest() {
		return contracts.Invalidf("invalid shared config digest")
	}
	return nil
}
