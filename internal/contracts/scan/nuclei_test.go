package scan

import (
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

func TestSharedNucleiInputs(t *testing.T) {
	data := contractstest.ReadFile(t, "api", "testdata", "nuclei-input-cases.json")
	var fixture struct {
		RulepackLimits struct {
			ZIPBytes  int
			FileBytes int
			MediaType string
		}
		Cases []struct {
			Name      string
			Arguments map[string]any
			Valid     bool
		}
		RulepackMetadataCases []struct {
			Name      string
			MediaType string
			Size      int64
			Valid     bool
		}
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.RulepackLimits.ZIPBytes != MaxNucleiRulepackZIPBytes || fixture.RulepackLimits.FileBytes != MaxNucleiRulepackFileBytes || fixture.RulepackLimits.MediaType != NucleiRulepackMediaType {
		t.Fatal("rulepack admission limits drift")
	}
	for _, c := range fixture.Cases {
		t.Run(c.Name, func(t *testing.T) {
			if err := ValidateNucleiArguments(c.Arguments); (err == nil) != c.Valid {
				t.Fatalf("valid=%v err=%v", c.Valid, err)
			}
		})
	}
	for _, c := range fixture.RulepackMetadataCases {
		t.Run(c.Name, func(t *testing.T) {
			if valid := ValidNucleiRulepackMetadata(c.MediaType, c.Size); valid != c.Valid {
				t.Fatalf("valid=%v, want %v", valid, c.Valid)
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
