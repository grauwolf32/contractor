package contracts

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type configIdentityCases struct {
	ID      configIdentityCaseGroup `json:"id"`
	Version configIdentityCaseGroup `json:"version"`
}

type configIdentityCaseGroup struct {
	MaxLength int      `json:"maxLength"`
	Valid     []string `json:"valid"`
	Invalid   []string `json:"invalid"`
}

func loadConfigIdentityCases(t *testing.T) configIdentityCases {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("..", "..", "api", "testdata", "v1alpha1", "config-identity-cases.json"))
	if err != nil {
		t.Fatal(err)
	}
	var cases configIdentityCases
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	return cases
}

func TestConfigIdentityCases(t *testing.T) {
	t.Parallel()
	cases := loadConfigIdentityCases(t)
	if cases.ID.MaxLength != MaxConfigIDLength || cases.Version.MaxLength != MaxConfigVersionLength {
		t.Fatalf("shared bounds = %d/%d, want %d/%d",
			cases.ID.MaxLength, cases.Version.MaxLength, MaxConfigIDLength, MaxConfigVersionLength)
	}
	for _, group := range []struct {
		values []string
		valid  bool
	}{{cases.ID.Valid, true}, {cases.ID.Invalid, false}} {
		for _, value := range group.values {
			if ValidIdentifier(value) != group.valid || ValidConfigID(value) != group.valid {
				t.Errorf("identifier %q: want valid=%v", value, group.valid)
			}
			// Private wire refs carry the grammar without the configuration bound.
			if err := validateSelector("ref", value+"@1"); (err == nil) != group.valid {
				t.Errorf("selector id %q: valid=%v, error=%v", value, group.valid, err)
			}
		}
	}
	for _, group := range []struct {
		values []string
		valid  bool
	}{{cases.Version.Valid, true}, {cases.Version.Invalid, false}} {
		for _, value := range group.values {
			if ValidVersion(value) != group.valid || ValidConfigVersion(value) != group.valid {
				t.Errorf("version %q: want valid=%v", value, group.valid)
			}
			if err := validateSelector("ref", "a@"+value); (err == nil) != group.valid {
				t.Errorf("selector version %q: valid=%v, error=%v", value, group.valid, err)
			}
		}
	}
	for _, unit := range []string{"a", "z"} {
		if !ValidConfigID(strings.Repeat(unit, MaxConfigIDLength)) || ValidConfigID(strings.Repeat(unit, MaxConfigIDLength+1)) {
			t.Errorf("configuration id bound is not %d characters", MaxConfigIDLength)
		}
	}
	for _, unit := range []string{"1", "A"} {
		if !ValidConfigVersion(strings.Repeat(unit, MaxConfigVersionLength)) || ValidConfigVersion(strings.Repeat(unit, MaxConfigVersionLength+1)) {
			t.Errorf("configuration version bound is not %d characters", MaxConfigVersionLength)
		}
	}
}

func TestValidRuntimeAgentID(t *testing.T) {
	t.Parallel()
	if !ValidRuntimeAgentID(strings.Repeat("0123456789abcdef", 4)) {
		t.Fatal("lowercase SHA-256 fingerprint rejected")
	}
	for _, invalid := range []string{
		"", strings.Repeat("a", 63), strings.Repeat("a", 65), strings.Repeat("A", 64),
		strings.Repeat("g", 64), strings.Repeat("a", 63) + "\n", "sha256:" + strings.Repeat("a", 64),
	} {
		if ValidRuntimeAgentID(invalid) {
			t.Errorf("Runtime Agent ID %q accepted", invalid)
		}
	}
}
