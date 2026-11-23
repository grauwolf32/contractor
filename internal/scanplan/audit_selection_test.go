package scanplan_test

import (
	"encoding/json"
	"maps"
	"reflect"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/scanplan"
)

const auditSettings = `{"scanner":"sqlmap","server":"https://target.test/api","operations":{"#/paths/~1pets~1{id}/get":{"parameters":{"path:id":42,"query:q":"abc"}}},"testParameters":["q"]}`

func TestAuditSelectionRejectsMalformedValuesAndParameters(t *testing.T) {
	params := map[string]string{"server": "https://target.test/api", "operations": `["#/paths/~1pets~1{id}/get"]`, "test_parameters": `["q"]`}
	for _, data := range []string{`null`, `[]`, `{"#/paths/~1pets~1{id}/get":null}`, `{"#/paths/~1pets~1{id}/get":{"Parameters":{}}}`, `{"#/paths/~1pets~1{id}/get":{"parameters":null}}`, `{"#/paths/~1pets~1{id}/get":{"parameters":{"path:id":1,"path:id":2}}}`, `{"#/paths/~1foreign/get":{}}`, `{"authentication":{"bearer":"secret"}}`, `{"#/paths/~1pets~1{id}/get":{"body":{"mediaType":"application/json"}}}`, strings.Repeat(" ", scanplan.MaxOptionsBytes) + `{}`} {
		if _, err := scanplan.CompileAuditSelection("sqlmap", params, []byte(data)); err == nil || strings.Contains(err.Error(), "secret") {
			t.Fatalf("accepted malformed values or leaked data: %v", err)
		}
	}
	for name, values := range map[string][]string{"server": {"/relative", "https://user:secret@target.test", "https://{host}", "https://target.test?"}, "operations": {"[]", "null", `[null]`, `["#/paths/~1pets/get","#/paths/~1pets/get"]`, `["#/paths/~2pets/get"]`}, "test_parameters": {"[]", `["q","q"]`, `[null]`}, "credential_id": {"invalid id"}} {
		for _, value := range values {
			changed := maps.Clone(params)
			changed[name] = value
			if _, err := scanplan.CompileAuditSelection("sqlmap", changed, nil); err == nil {
				t.Fatalf("accepted invalid %s", name)
			}
		}
	}
}

func TestAuditSelectionPrepareOnlySelectedConcreteRequest(t *testing.T) {
	data := encoded(t, document(map[string]any{
		"/pets/{id}": map[string]any{"get": map[string]any{"parameters": []any{
			map[string]any{"in": "path", "name": "id", "required": true},
			map[string]any{"in": "query", "name": "q"},
		}}},
		"/unselected": map[string]any{"get": map[string]any{}},
	}))
	settings, err := compileSelectionFixture([]byte(auditSettings))
	if err != nil {
		t.Fatal(err)
	}
	pointer := "#/paths/~1pets~1{id}/get"
	operations, parameters := settings.Operations(), settings.TestParameters()
	operations[0], parameters[0] = "mutated", "mutated"
	if !reflect.DeepEqual(settings.Operations(), []string{pointer}) || !reflect.DeepEqual(settings.TestParameters(), []string{"q"}) {
		t.Fatal("mutable settings accessors")
	}
	prepared, err := settings.PrepareOperation(data, "application/json", sourceRef(), pointer)
	if err != nil || !prepared.Runnable || prepared.Target != nil || prepared.RequestSet == nil ||
		len(prepared.RequestSet.Requests) != 1 || prepared.RequestSet.Requests[0].Request.URL != "https://target.test/api/pets/42?q=abc" ||
		prepared.RequestSet.Source.ContentDigest != digest(data) {
		t.Fatalf("wrong concrete request: %+v, %v", prepared, err)
	}
	prepared.RequestSet.Requests[0].Request.URL = "http://mutated.test"
	again, err := settings.PrepareOperation(data, "application/json", sourceRef(), pointer)
	if err != nil || again.RequestSet.Requests[0].Request.URL != "https://target.test/api/pets/42?q=abc" {
		t.Fatal("preparation mutated settings")
	}
	if _, err := settings.PrepareOperation(data, "application/json", sourceRef(), "#/paths/~1unselected/get"); err == nil {
		t.Fatal("unassigned operation allowed")
	}
	missing, err := compileSelectionFixture([]byte(strings.Replace(auditSettings, `["q"]`, `["excluded"]`, 1)))
	if err != nil {
		t.Fatal(err)
	}
	skipped, err := missing.PrepareOperation(data, "application/json", sourceRef(), pointer)
	if err != nil || skipped.Runnable || !hasAuditGap(skipped, "test_parameter_unavailable") {
		t.Fatalf("excluded parameter was broadened: %+v, %v", skipped, err)
	}
}

func TestAuditNucleiSettingsRejectDroppedDataAndRetainMissingTarget(t *testing.T) {
	var settings map[string]any
	if err := json.Unmarshal([]byte(auditSettings), &settings); err != nil {
		t.Fatal(err)
	}
	settings["scanner"] = "nuclei"
	delete(settings, "testParameters")
	for name, input := range map[string]any{
		"header": map[string]any{"parameters": map[string]any{"header:Authorization": "never-echo-token"}},
		"cookie": map[string]any{"parameters": map[string]any{"cookie:session": "never-echo-token"}},
		"body":   map[string]any{"body": map[string]any{"mediaType": "text/plain", "value": "never-echo-token"}},
	} {
		t.Run(name, func(t *testing.T) {
			settings["operations"] = map[string]any{"#/paths/~1pets~1{id}/get": input}
			if _, err := compileSelectionFixture(encoded(t, settings)); err == nil {
				t.Fatal("dropped explicit data")
			}
		})
	}
	settings["operations"] = map[string]any{"#/paths/~1pets~1{id}/get": map[string]any{}}
	decoded, err := compileSelectionFixture(encoded(t, settings))
	if err != nil {
		t.Fatal(err)
	}
	source := encoded(t, document(map[string]any{"/pets/{id}": map[string]any{"get": map[string]any{"parameters": []any{map[string]any{"name": "id", "in": "path"}}}}}))
	prepared, err := decoded.PrepareOperation(source, "application/json", sourceRef(), "#/paths/~1pets~1{id}/get")
	if err != nil || prepared.Runnable || prepared.RequestSet != nil || prepared.Target == nil || prepared.Target.URL != "" ||
		!hasAuditGap(prepared, "missing_required_parameter") || !hasAuditGap(prepared, "url_template_scan_only") {
		t.Fatalf("missing target became successful: %+v, %v", prepared, err)
	}
}

func hasAuditGap(prepared scanplan.PreparedAuditOperation, code string) bool {
	for _, gap := range prepared.Gaps {
		if gap.Code == code {
			return true
		}
	}
	return false
}
