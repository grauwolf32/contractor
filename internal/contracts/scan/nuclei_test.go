package scan

import (
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/strictjson"
)

func TestSharedNucleiInputs(t *testing.T) {
	data := contractstest.ReadFile(t, "api", "testdata", "nuclei-input-cases.json")
	var fixture struct {
		Cases []struct {
			Name      string
			Arguments map[string]any
			Valid     bool
		}
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	for _, c := range fixture.Cases {
		t.Run(c.Name, func(t *testing.T) {
			if err := ValidateNucleiArguments(c.Arguments); (err == nil) != c.Valid {
				t.Fatalf("valid=%v err=%v", c.Valid, err)
			}
		})
	}
}

func TestSharedNucleiPackageCanonicalManifest(t *testing.T) {
	data := contractstest.ReadFile(t, "api", "testdata", "nuclei-package-manifest.json")
	var fixture struct {
		Manifest any
		Digest   string
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	canonical, err := strictjson.Canonical(fixture.Manifest)
	if err != nil {
		t.Fatal(err)
	}
	if contentdigest.Bytes(canonical) != fixture.Digest {
		t.Fatal("canonical package identity drift")
	}
}
