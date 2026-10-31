package control

import "testing"

func TestSharedConfigValidatesVersionBoundsAndDigest(t *testing.T) {
	good := DefaultSharedConfig()
	if err := good.Validate(); err != nil {
		t.Fatal(err)
	}
	for _, mutate := range []func(*SharedConfig){
		func(c *SharedConfig) { c.Version++ },
		func(c *SharedConfig) { c.Limits.ArtifactPayloadMaxBytes = 0 },
		func(c *SharedConfig) { c.Limits.ArtifactPayloadMaxBytes++ },
		func(c *SharedConfig) { c.Digest = "sha256:" },
	} {
		invalid := good
		mutate(&invalid)
		if invalid.Validate() == nil {
			t.Fatal("accepted invalid config")
		}
	}
}
