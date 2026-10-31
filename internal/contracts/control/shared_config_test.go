package control

import (
	"github.com/grauwolf32/contractor/internal/contracts"
	"reflect"
	"testing"
)

func TestSharedConfigValidatesVersionBoundsAndDigest(t *testing.T) {
	good, err := DefaultSharedConfig()
	if err != nil {
		t.Fatal(err)
	}
	if err := good.Validate(); err != nil {
		t.Fatal(err)
	}
	for _, mutate := range []func(*SharedConfig){
		func(c *SharedConfig) { c.Version++ },
		func(c *SharedConfig) { c.Limits.ArtifactPayloadMaxBytes = 0 },
		func(c *SharedConfig) { c.Limits.ArtifactPayloadMaxBytes++ },
		func(c *SharedConfig) { c.Limits.PublicPageMaxItems = 0 },
		func(c *SharedConfig) { c.Limits.ArtifactListMaxItems = 0 },
		func(c *SharedConfig) { c.Limits.PublicPageMaxItems++ },
		func(c *SharedConfig) { c.Limits.ArtifactListMaxItems++ },
		func(c *SharedConfig) { c.Digest = "sha256:" },
	} {
		invalid := good
		mutate(&invalid)
		if invalid.Validate() == nil {
			t.Fatal("accepted invalid config")
		}
	}
}

func TestEverySharedLimitHasIndependentBoundsAndParticipatesInDigest(t *testing.T) {
	original, err := DefaultSharedConfig()
	if err != nil {
		t.Fatal(err)
	}
	fields := reflect.TypeOf(original.Limits)
	for i := 0; i < fields.NumField(); i++ {
		t.Run(fields.Field(i).Name, func(t *testing.T) {
			for _, invalid := range []int64{0, contracts.MaxExactInteger + 1} {
				value := original
				reflect.ValueOf(&value.Limits).Elem().Field(i).SetInt(invalid)
				value.Digest, err = value.digest()
				if err != nil {
					t.Fatal(err)
				}
				if value.Validate() == nil {
					t.Fatalf("accepted limit %d", invalid)
				}
			}
			value := original
			field := reflect.ValueOf(&value.Limits).Elem().Field(i)
			field.SetInt(field.Int() + 1)
			if value.Validate() == nil {
				t.Fatal("accepted stale digest")
			}
			value.Digest, err = value.digest()
			if err != nil {
				t.Fatal(err)
			}
			if err := value.Validate(); err != nil {
				t.Fatal(err)
			}
		})
	}
}
