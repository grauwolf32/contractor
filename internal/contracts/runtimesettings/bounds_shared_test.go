package runtimesettings

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

// TestSharedRuntimeSettingsBoundsCases checks RuntimeSettings decoding against
// api/testdata/v1alpha1/runtime-settings-bounds-cases.json, which the Runtime
// and the private JSON Schema are held to as well
// (runtime/tests/test_shared_runtime_settings_bounds.py), and the defaults
// the Server sends for an omitted export or retry block.
func TestSharedRuntimeSettingsBoundsCases(t *testing.T) {
	t.Parallel()
	var fixture struct {
		Description    string `json:"description"`
		Base           string `json:"base"`
		ServerDefaults struct {
			Export TelemetryExport `json:"export"`
		} `json:"serverDefaults"`
		Cases []struct {
			Name       string                     `json:"name"`
			Set        map[string]json.RawMessage `json:"set"`
			Remove     []string                   `json:"remove"`
			Valid      *bool                      `json:"valid"`
			CrossField bool                       `json:"crossField"`
		} `json:"cases"`
	}
	decoder := json.NewDecoder(bytes.NewReader(contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "runtime-settings-bounds-cases.json")))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&fixture); err != nil {
		t.Fatal(err)
	}
	if len(fixture.Cases) == 0 || !strings.HasPrefix(fixture.Base, "valid/") {
		t.Fatalf("runtime settings bound cases are incomplete: %+v", fixture)
	}
	if want := PinTelemetryExport(nil); fixture.ServerDefaults.Export != want {
		t.Fatalf("shared server defaults = %+v, want the Server's %+v", fixture.ServerDefaults.Export, want)
	}
	base := contractstest.ReadFixture(t, "valid", strings.TrimPrefix(fixture.Base, "valid/"))
	for _, test := range fixture.Cases {
		if test.Name == "" || test.Valid == nil || test.CrossField && *test.Valid {
			t.Fatalf("runtime settings bound case %q needs a name and a verdict; a crossField case is invalid", test.Name)
		}
		var document map[string]any
		if err := json.Unmarshal(base, &document); err != nil {
			t.Fatal(err)
		}
		for pointer, raw := range test.Set {
			parent, key := boundCaseParent(t, document, pointer)
			parent[key] = raw
		}
		for _, pointer := range test.Remove {
			parent, key := boundCaseParent(t, document, pointer)
			delete(parent, key)
		}
		data, err := json.Marshal(document)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := contracts.DecodePrivateStrict[RuntimeSettings](data); (err == nil) != *test.Valid {
			t.Errorf("%s: RuntimeSettings accepted = %t, want %t: %v", test.Name, err == nil, *test.Valid, err)
		}
	}
}

// boundCaseParent returns the object holding the last segment of a JSON
// pointer and that segment; every segment before it must name an object.
func boundCaseParent(t *testing.T, document map[string]any, pointer string) (map[string]any, string) {
	t.Helper()
	segments := strings.Split(strings.TrimPrefix(pointer, "/"), "/")
	if !strings.HasPrefix(pointer, "/") || len(segments) == 0 {
		t.Fatalf("invalid JSON pointer %q", pointer)
	}
	current := document
	for _, segment := range segments[:len(segments)-1] {
		next, ok := current[segment].(map[string]any)
		if !ok {
			t.Fatalf("JSON pointer %q does not name an object at %q", pointer, segment)
		}
		current = next
	}
	return current, segments[len(segments)-1]
}
