package scanplan_test

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/scanplan"
)

const auditSettings = `{"schema":"contractor.audit.openapi-scan-settings.v1","scanner":"sqlmap","server":"https://target.test/api","operations":{"#/paths/~1pets~1{id}/get":{"parameters":{"path:id":42,"query:q":"abc"}}},"testParameters":["q"]}`

func TestAuditScanSettingsRejectAmbiguityAndImplicitAuthority(t *testing.T) {
	cases := map[string]string{
		"duplicate":                 strings.Replace(auditSettings, `"scanner":"sqlmap"`, `"scanner":"nuclei","scanner":"sqlmap"`, 1),
		"case alias":                strings.Replace(auditSettings, `"scanner"`, `"Scanner"`, 1),
		"unknown":                   strings.Replace(auditSettings, `"scanner"`, `"extra":true,"scanner"`, 1),
		"nested alias":              strings.Replace(auditSettings, `"parameters"`, `"Parameters"`, 1),
		"nested unknown":            strings.Replace(auditSettings, `"parameters"`, `"extra":0,"parameters"`, 1),
		"nested duplicate":          strings.Replace(auditSettings, `"path:id":42`, `"path:id":1,"path:id":42`, 1),
		"null operation":            strings.Replace(auditSettings, `{"parameters":{"path:id":42,"query:q":"abc"}}`, `null`, 1),
		"null parameter map":        strings.Replace(auditSettings, `{"path:id":42,"query:q":"abc"}`, `null`, 1),
		"relative server":           strings.Replace(auditSettings, `https://target.test/api`, `/api`, 1),
		"templated server":          strings.Replace(auditSettings, `https://target.test/api`, `https://{host}/api`, 1),
		"credentials":               strings.Replace(auditSettings, `https://target.test/api`, `https://user:secret@target.test/api`, 1),
		"server query":              strings.Replace(auditSettings, `https://target.test/api`, `https://target.test/api?id=2`, 1),
		"empty query":               strings.Replace(auditSettings, `https://target.test/api`, `https://target.test/api?`, 1),
		"server fragment":           strings.Replace(auditSettings, `https://target.test/api`, `https://target.test/api#secret`, 1),
		"server port":               strings.Replace(auditSettings, `https://target.test/api`, `https://target.test:0/api`, 1),
		"unknown scanner":           strings.Replace(auditSettings, `"sqlmap"`, `"automatic"`, 1),
		"version":                   strings.Replace(auditSettings, `settings.v1`, `settings.v2`, 1),
		"bad pointer":               strings.Replace(auditSettings, `~1pets`, `~2pets`, 1),
		"uppercase method":          strings.Replace(auditSettings, `/get`, `/GET`, 1),
		"invalid parameter":         strings.Replace(auditSettings, `"path:id"`, `"id"`, 1),
		"empty test parameters":     strings.Replace(auditSettings, `["q"]`, `[]`, 1),
		"duplicate test parameters": strings.Replace(auditSettings, `["q"]`, `["q","q"]`, 1),
		"invalid test parameter":    strings.Replace(auditSettings, `["q"]`, `["q,secret"]`, 1),
		"null test parameter":       strings.Replace(auditSettings, `["q"]`, `[null]`, 1),
		"null credentials":          strings.Replace(auditSettings, `"scanner"`, `"authentication":{"bearer":null},"scanner"`, 1),
		"header alias":              strings.Replace(auditSettings, `"path:id":42`, `"header:X-Key":"a","header:x-key":"b","path:id":42`, 1),
		"unsafe number":             strings.Replace(auditSettings, `42`, `9007199254740993`, 1),
		"trailing":                  auditSettings + `{}`,
		"size":                      strings.Repeat(" ", scanplan.MaxOptionsBytes) + auditSettings,
	}
	for name, data := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := scanplan.DecodeAuditScanSettings([]byte(data))
			if err == nil || err.Error() != "request preparation: invalid_audit_scan_settings" {
				t.Fatalf("invalid settings accepted or data leaked: %v", err)
			}
		})
	}
}

func TestAuditScanSettingsPrepareOnlySelectedConcreteRequest(t *testing.T) {
	data := encoded(t, document(map[string]any{
		"/pets/{id}": map[string]any{"get": map[string]any{"parameters": []any{
			map[string]any{"in": "path", "name": "id", "required": true},
			map[string]any{"in": "query", "name": "q"},
		}}},
		"/unselected": map[string]any{"get": map[string]any{}},
	}))
	settings, err := scanplan.DecodeAuditScanSettings([]byte(auditSettings))
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
	missing, err := scanplan.DecodeAuditScanSettings([]byte(strings.Replace(auditSettings, `["q"]`, `["excluded"]`, 1)))
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
	base := encoded(t, settings)
	for name, inserted := range map[string]string{
		"auth":            `"authentication":{"bearer":"never-echo-token"},`,
		"test parameters": `"testParameters":["q"],`,
	} {
		t.Run(name, func(t *testing.T) {
			data := "{" + inserted + string(base[1:])
			if _, err := scanplan.DecodeAuditScanSettings([]byte(data)); err == nil {
				t.Fatal("dropped explicit data")
			}
		})
	}
	for name, input := range map[string]any{
		"header": map[string]any{"parameters": map[string]any{"header:Authorization": "never-echo-token"}},
		"cookie": map[string]any{"parameters": map[string]any{"cookie:session": "never-echo-token"}},
		"body":   map[string]any{"body": map[string]any{"mediaType": "text/plain", "value": "never-echo-token"}},
	} {
		t.Run(name, func(t *testing.T) {
			settings["operations"] = map[string]any{"#/paths/~1pets~1{id}/get": input}
			if _, err := scanplan.DecodeAuditScanSettings(encoded(t, settings)); err == nil {
				t.Fatal("dropped explicit data")
			}
		})
	}
	settings["operations"] = map[string]any{"#/paths/~1pets~1{id}/get": map[string]any{}}
	decoded, err := scanplan.DecodeAuditScanSettings(encoded(t, settings))
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
