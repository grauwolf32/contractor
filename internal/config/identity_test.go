package config

import (
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/contracts"
	"go.yaml.in/yaml/v4"
)

type configIdentityCaseGroup struct {
	MaxLength int      `json:"maxLength"`
	Valid     []string `json:"valid"`
	Invalid   []string `json:"invalid"`
}

func loadConfigIdentityCases(t *testing.T) (ids, versions configIdentityCaseGroup) {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("..", "..", "api", "testdata", "v1alpha1", "config-identity-cases.json"))
	if err != nil {
		t.Fatal(err)
	}
	var cases struct {
		ID      configIdentityCaseGroup `json:"id"`
		Version configIdentityCaseGroup `json:"version"`
	}
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	return cases.ID, cases.Version
}

// identityAcceptance reports, in order, whether ParseSelector, file metadata
// validation and publication accept one configuration identity.
func identityAcceptance(id, version string) [3]bool {
	_, selectorErr := ParseSelector(id + "@" + version)
	_, metadataErr := validateMetadata(&metadataSource{Name: id, Version: version})
	request := validPolicyPublication("identity-probe")
	request.Name, request.Version = id, version
	_, publicationErr := preparePublication(request)
	return [3]bool{selectorErr == nil, metadataErr == nil, publicationErr == nil}
}

func TestConfigIdentityMatchesSharedCases(t *testing.T) {
	t.Parallel()
	ids, versions := loadConfigIdentityCases(t)
	check := func(kind, id, version string, valid bool) {
		t.Helper()
		if got := identityAcceptance(id, version); got != [3]bool{valid, valid, valid} {
			t.Errorf("%s %q@%q: selector/metadata/publication acceptance = %v, want %v", kind, id, version, got, valid)
		}
	}
	for _, value := range ids.Valid {
		check("id", value, "1", true)
	}
	for _, value := range ids.Invalid {
		check("id", value, "1", false)
	}
	for _, value := range versions.Valid {
		check("version", "a", value, true)
	}
	for _, value := range versions.Invalid {
		check("version", "a", value, false)
	}
	check("id", strings.Repeat("a", ids.MaxLength), "1", true)
	check("id", strings.Repeat("a", ids.MaxLength+1), "1", false)
	check("version", "a", strings.Repeat("1", versions.MaxLength), true)
	check("version", "a", strings.Repeat("1", versions.MaxLength+1), false)
}

type publicStringSchema struct {
	Pattern   string `yaml:"pattern"`
	MinLength *int   `yaml:"minLength"`
	MaxLength *int   `yaml:"maxLength"`
}

func (s publicStringSchema) accepts(t *testing.T, value string) bool {
	t.Helper()
	length := utf8.RuneCountInString(value)
	if s.MinLength != nil && length < *s.MinLength || s.MaxLength != nil && length > *s.MaxLength {
		return false
	}
	pattern, err := regexp.Compile(s.Pattern)
	if err != nil {
		t.Fatalf("OpenAPI pattern %q: %v", s.Pattern, err)
	}
	return pattern.MatchString(value)
}

// identityProbes covers every printable ASCII character and a few control and
// non-ASCII characters in leading, inner and trailing positions, plus the
// length bounds.
func identityProbes() []string {
	characters := []string{"\n", "\t", "\x00", "é", "я"}
	for character := byte(32); character < 127; character++ {
		characters = append(characters, string(character))
	}
	seen := map[string]struct{}{}
	var probes []string
	add := func(value string) {
		if _, duplicate := seen[value]; !duplicate {
			seen[value] = struct{}{}
			probes = append(probes, value)
		}
	}
	add("")
	for _, character := range characters {
		for _, value := range []string{
			character, "a" + character, "a" + character + "a", "1" + character, "1" + character + "1", character + "a",
		} {
			add(value)
		}
	}
	for _, length := range []int{63, 64, 65, 127, 128, 129} {
		for _, unit := range []string{"a", "z", "1", "A"} {
			add(strings.Repeat(unit, length))
		}
	}
	return probes
}

func TestConfigIdentityMatchesPublicOpenAPI(t *testing.T) {
	t.Parallel()
	data, err := os.ReadFile(filepath.Join("..", "..", "api", "openapi", "contractor-public-v1.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	var document struct {
		Components struct {
			Schemas map[string]publicStringSchema `yaml:"schemas"`
		} `yaml:"components"`
	}
	if err := yaml.Unmarshal(data, &document); err != nil {
		t.Fatal(err)
	}
	name := document.Components.Schemas["ConfigurationName"]
	version := document.Components.Schemas["ConfigVersion"]
	selector := document.Components.Schemas["Selector"]
	if name.Pattern == "" || version.Pattern == "" || selector.Pattern == "" {
		t.Fatal("OpenAPI lacks the ConfigurationName, ConfigVersion or Selector pattern")
	}
	if *name.MaxLength != contracts.MaxConfigIDLength || *version.MaxLength != contracts.MaxConfigVersionLength {
		t.Fatalf("OpenAPI bounds = %d/%d", *name.MaxLength, *version.MaxLength)
	}
	for _, probe := range identityProbes() {
		wantID := name.accepts(t, probe)
		if got := identityAcceptance(probe, "1"); got != [3]bool{wantID, wantID, wantID} {
			t.Errorf("id %q: selector/metadata/publication acceptance = %v, OpenAPI ConfigurationName = %v", probe, got, wantID)
		}
		if got := selector.accepts(t, probe+"@1"); got != wantID {
			t.Errorf("Selector %q@1 = %v, want %v", probe, got, wantID)
		}
		wantVersion := version.accepts(t, probe)
		if got := identityAcceptance("a", probe); got != [3]bool{wantVersion, wantVersion, wantVersion} {
			t.Errorf("version %q: selector/metadata/publication acceptance = %v, OpenAPI ConfigVersion = %v", probe, got, wantVersion)
		}
		if got := selector.accepts(t, "a@"+probe); got != wantVersion {
			t.Errorf("Selector a@%q = %v, want %v", probe, got, wantVersion)
		}
	}
}
