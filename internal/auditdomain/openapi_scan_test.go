package auditdomain

import (
	"bytes"
	"encoding/json"
	"os"
	"reflect"
	"strings"
	"testing"
)

func TestOpenAPIScanDocumentedInputs(t *testing.T) {
	root := "../../configs/scan/examples/audit-openapi-scan/"
	source, err := os.ReadFile(root + "openapi.json")
	if err != nil {
		t.Fatal(err)
	}
	for _, scanner := range []string{"sqlmap", "nuclei"} {
		t.Run(scanner, func(t *testing.T) {
			settings, err := os.ReadFile(root + scanner + "-settings.json")
			if err != nil {
				t.Fatal(err)
			}
			options := testInventoryOptions("scan")
			options.ApprovalRequirement = ApprovalActiveCheck
			input := ExactInput{Name: "settings", Ref: copyArtifactRef(options.SourceRef), Digest: DigestBytes(settings)}
			input.Ref.Name = scanner + "-settings"
			inventory, err := BuildOpenAPIScanInventory(source, "application/json", settings, input, options)
			if err != nil || len(inventory.Tasks) != 1 {
				t.Fatalf("documented inventory: %v", err)
			}
			prepared, err := PrepareOpenAPIScanTask(inventory.Tasks[0].Document, source, settings)
			if err != nil || !prepared.Runnable {
				t.Fatalf("documented preparation: %v", err)
			}
			wantURL := "http://127.0.0.1:8080/api/pets/7?search=Milo"
			if scanner == "nuclei" {
				if prepared.Target == nil || prepared.Target.URL != wantURL || prepared.RequestSet != nil {
					t.Fatal("documented URL changed")
				}
			} else {
				request := prepared.RequestSet.Requests[0].Request
				if request.URL != wantURL || request.Method != "POST" || request.Body != `{"name":"Milo"}` {
					t.Fatal("documented request changed")
				}
			}
		})
	}
}

const scanSource = `{
 "openapi":"3.1.0","servers":[{"url":"https://unselected.test"}],
 "paths":{
  "/pets/{id}":{"post":{
   "parameters":[{"in":"path","name":"id","required":true}],
   "security":[{"bearer":[]}],
   "requestBody":{"required":true,"content":{"application/json":{"schema":{"type":"object"}}}}
  }},
  "/missing/{id}":{"get":{"parameters":[{"in":"path","name":"id","required":true}]}},
  "/unselected":{"get":{}}
 },
 "components":{"securitySchemes":{"bearer":{"type":"http","scheme":"bearer"}}}
}`

const sqlmapSettings = `{
 "schema":"contractor.audit.openapi-scan-settings.v1","scanner":"sqlmap","server":"https://target.test/api",
 "authentication":{"bearer":"never-copy-credential"},"testParameters":["q"],
 "operations":{"#/paths/~1pets~1{id}/post":{
  "parameters":{"path:id":7},"body":{"mediaType":"application/json","value":{"q":"abc"}}
 }}
}`

const nucleiSettings = `{
 "schema":"contractor.audit.openapi-scan-settings.v1","scanner":"nuclei","server":"https://target.test/api",
 "operations":{
  "#/paths/~1pets~1{id}/post":{"parameters":{"path:id":7}},
  "#/paths/~1missing~1{id}/get":{}
 }
}`

func scanInventoryFixture(t *testing.T, settings string) (Inventory, InventoryOptions, ExactInput) {
	t.Helper()
	options := testInventoryOptions("scan")
	options.ApprovalRequirement = ApprovalActiveCheck
	input := ExactInput{Name: "settings", Ref: copyArtifactRef(options.SourceRef), Digest: DigestBytes([]byte(settings))}
	input.Ref.Name = "scan-settings"
	inventory, err := BuildOpenAPIScanInventory([]byte(scanSource), "application/json", []byte(settings), input, options)
	if err != nil {
		t.Fatal(err)
	}
	return inventory, options, input
}

func TestOpenAPIScanInventoryPinsRequestSettingsAndApproval(t *testing.T) {
	inventory, options, input := scanInventoryFixture(t, sqlmapSettings)
	if len(inventory.Tasks) != 1 {
		t.Fatalf("broadened inventory: %d", len(inventory.Tasks))
	}
	task := inventory.Tasks[0].Document
	if task.Kind != "openapi-scan" || task.Operation != nil || task.Checklist != nil || task.Scan == nil ||
		!task.Scan.Runnable || task.Scan.Scanner != "sqlmap" || task.Scan.TargetURL != "" ||
		!validDigest(task.Scan.RequestDigest) || !sameCanonicalValue(task.Scan.Settings, input) ||
		!equalStringSlices(inventory.Coverage.Rows[0].Requested, []string{"sqlmap-request-scan"}) ||
		inventory.Worklist.Items[0].ApprovalRequirement != ApprovalActiveCheck {
		t.Fatalf("wrong scan task: %+v", task)
	}
	if bytes.Contains(inventory.CanonicalInventory, []byte("never-copy-credential")) {
		t.Fatal("credential copied into inventory")
	}
	encoded, err := EncodeItemTask(task)
	if err != nil || bytes.Contains(encoded, []byte("never-copy-credential")) {
		t.Fatal("credential copied into task")
	}
	decoded, err := DecodeItemTask(encoded)
	if err != nil || !reflect.DeepEqual(task, decoded) {
		t.Fatalf("task roundtrip: %v", err)
	}
	inputs := inventory.ExecutionManifest.Items[0].Inputs
	if len(inputs) != 2 || inputs[0].Name != "settings" || inputs[1].Name != "source" {
		t.Fatalf("inputs not pinned: %+v", inputs)
	}
	prepared, err := PrepareOpenAPIScanTask(decoded, []byte(scanSource), []byte(sqlmapSettings))
	if err != nil || !prepared.Runnable || prepared.RequestSet == nil {
		t.Fatalf("prepare accepted task: %v", err)
	}
	request := prepared.RequestSet.Requests[0].Request
	if request.URL != "https://target.test/api/pets/7" || request.Method != "POST" || request.Body != `{"q":"abc"}` {
		t.Fatalf("lost request data: %+v", request)
	}
	auth := false
	for _, header := range request.Headers {
		auth = auth || header.Name == "authorization" && header.Value == "Bearer never-copy-credential"
	}
	if !auth {
		t.Fatal("request authentication lost")
	}
	*input.Ref.Revision, *options.SourceRef.Revision = "changed-settings", "changed-source"
	if *task.Scan.Settings.Ref.Revision != "source-revision-1" || *task.SourceRef.Revision != "source-revision-1" || ValidateInventory(inventory) != nil {
		t.Fatal("retained refs alias caller data")
	}
	again, _, _ := scanInventoryFixture(t, sqlmapSettings)
	if !bytes.Equal(inventory.Tasks[0].Package, again.Tasks[0].Package) || !bytes.Equal(inventory.CanonicalInventory, again.CanonicalInventory) {
		t.Fatal("scan inventory is not deterministic")
	}
}

func TestOpenAPIScanInventoryKeepsNucleiLimitationsAndUnpreparedItem(t *testing.T) {
	inventory, _, _ := scanInventoryFixture(t, nucleiSettings)
	if len(inventory.Tasks) != 2 {
		t.Fatal("lost unprepared operation")
	}
	missing, target := inventory.Tasks[0].Document, inventory.Tasks[1].Document
	if missing.Scan.Runnable || missing.Scan.TargetURL != "" || !containsString(missing.Scan.Gaps, "missing_required_parameter") {
		t.Fatalf("missing value invented: %+v", missing.Scan)
	}
	if !target.Scan.Runnable || target.Scan.TargetURL != "https://target.test/api/pets/7" || target.Scan.RequestDigest != "" {
		t.Fatalf("lost fixed URL: %+v", target.Scan)
	}
	for _, code := range []string{"url_template_scan_only", "http_method_not_replayed", "request_body_not_replayed", "authentication_not_applied"} {
		if !containsString(target.Scan.Gaps, code) {
			t.Fatalf("missing %s: %+v", code, target.Scan.Gaps)
		}
	}
	for index, task := range inventory.Tasks {
		if !equalStringSlices(inventory.Coverage.Rows[index].Requested, []string{"nuclei-url-template-scan"}) || inventory.Coverage.Rows[index].Status != "not-tested" {
			t.Fatal("URL preparation counted as passing coverage")
		}
		prepared, err := PrepareOpenAPIScanTask(task.Document, []byte(scanSource), []byte(nucleiSettings))
		if err != nil || prepared.RequestSet != nil || prepared.Target == nil || prepared.Runnable != task.Document.Scan.Runnable {
			t.Fatalf("cannot reproduce task: %v", err)
		}
	}
}

func TestOpenAPIScanPreparationRejectsChangedBytesAndAuthority(t *testing.T) {
	inventory, _, _ := scanInventoryFixture(t, sqlmapSettings)
	for name, mutate := range map[string]func(*ItemTask){
		"scanner":            func(task *ItemTask) { task.Scan.Scanner = "nuclei" },
		"operation":          func(task *ItemTask) { task.Scan.Operation = "#/paths/~1unselected/get" },
		"settings digest":    func(task *ItemTask) { task.Scan.Settings.Digest = DigestBytes([]byte("changed")) },
		"source ref":         func(task *ItemTask) { *task.SourceRef.Revision = "changed" },
		"preparation digest": func(task *ItemTask) { task.Scan.PreparationDigest = DigestBytes([]byte("changed")) },
		"request digest":     func(task *ItemTask) { task.Scan.RequestDigest = DigestBytes([]byte("changed")) },
		"parameters":         func(task *ItemTask) { task.Scan.TestParameters = []string{"excluded"} },
		"runnable":           func(task *ItemTask) { task.Scan.Runnable = false; task.Scan.Gaps = []string{"invented"} },
		"kind":               func(task *ItemTask) { task.Kind = "operation-trace" },
	} {
		t.Run(name, func(t *testing.T) {
			data, err := EncodeItemTask(inventory.Tasks[0].Document)
			if err != nil {
				t.Fatal(err)
			}
			task, err := DecodeItemTask(data)
			if err != nil {
				t.Fatal(err)
			}
			mutate(&task)
			if _, err := PrepareOpenAPIScanTask(task, []byte(scanSource), []byte(sqlmapSettings)); err == nil {
				t.Fatal("changed authority accepted")
			}
		})
	}
	for _, pair := range [][2]string{{scanSource + " ", sqlmapSettings}, {scanSource, sqlmapSettings + " "}} {
		if _, err := PrepareOpenAPIScanTask(inventory.Tasks[0].Document, []byte(pair[0]), []byte(pair[1])); err == nil {
			t.Fatal("changed exact input bytes accepted")
		}
	}
}

func TestOpenAPIScanInventoryIdentityIncludesExactSettingsAndSelection(t *testing.T) {
	initial, options, input := scanInventoryFixture(t, sqlmapSettings)
	for _, settings := range []string{
		sqlmapSettings + " ",
		strings.Replace(sqlmapSettings, "never-copy-credential", "changed-credential", 1),
		strings.Replace(sqlmapSettings, `"testParameters":["q"]`, `"testParameters":["excluded"]`, 1),
		strings.Replace(sqlmapSettings, "https://target.test/api", "https://other.test/api", 1),
	} {
		input.Digest = DigestBytes([]byte(settings))
		changed, err := BuildOpenAPIScanInventory([]byte(scanSource), "application/json", []byte(settings), input, options)
		if err != nil {
			t.Fatal(err)
		}
		if changed.CanonicalInventoryDigest == initial.CanonicalInventoryDigest || changed.Tasks[0].PackageDigest == initial.Tasks[0].PackageDigest {
			t.Fatal("settings identity lost")
		}
		if strings.Contains(settings, `["excluded"]`) && (changed.Tasks[0].Document.Scan.Runnable || !containsString(changed.Tasks[0].Document.Scan.Gaps, "test_parameter_unavailable")) {
			t.Fatal("unavailable test parameter did not produce gap")
		}
	}
}

func TestOpenAPIScanInventoryRejectsMalformedAssignmentsAtomically(t *testing.T) {
	_, options, input := scanInventoryFixture(t, nucleiSettings)
	for _, settings := range []string{
		strings.Replace(nucleiSettings, `"#/paths/~1pets~1{id}/post"`, `"#/paths/~1unknown/get"`, 1),
		strings.Replace(nucleiSettings, `"nuclei"`, `"unknown"`, 1),
	} {
		input.Digest = DigestBytes([]byte(settings))
		inventory, err := BuildOpenAPIScanInventory([]byte(scanSource), "application/json", []byte(settings), input, options)
		if err == nil || ErrorCode(err) == "" || len(inventory.Tasks) != 0 {
			t.Fatalf("accepted partial/malformed inventory: %v", err)
		}
	}
	input.Digest = DigestBytes([]byte(nucleiSettings))
	options.ApprovalRequirement = ApprovalNone
	if _, err := BuildOpenAPIScanInventory([]byte(scanSource), "application/json", []byte(nucleiSettings), input, options); err == nil {
		t.Fatal("scan approval dropped")
	}
}

func TestOpenAPIScanInventoryValidatorChecksSettingsAndCoverage(t *testing.T) {
	for name, mutate := range map[string]func(*Inventory){
		"settings absent":  func(v *Inventory) { v.ExecutionManifest.Items[0].Inputs = v.ExecutionManifest.Items[0].Inputs[1:] },
		"settings changed": func(v *Inventory) { v.ExecutionManifest.Items[0].Inputs[0].Digest = DigestBytes([]byte("changed")) },
		"source absent":    func(v *Inventory) { v.ExecutionManifest.Items[0].Inputs = v.ExecutionManifest.Items[0].Inputs[:1] },
		"trace coverage":   func(v *Inventory) { v.Coverage.Rows[0].Requested = []string{"operation-resolution"} },
		"approval": func(v *Inventory) {
			v.Worklist.Items[0].ApprovalRequirement = ApprovalNone
			v.Tasks[0].Item.ApprovalRequirement = ApprovalNone
		},
	} {
		t.Run(name, func(t *testing.T) {
			v, _, _ := scanInventoryFixture(t, sqlmapSettings)
			mutate(&v)
			if ValidateInventory(v) == nil {
				t.Fatal("invalid scan inventory accepted")
			}
		})
	}
	// Existing trace inventories retain their original shape and semantics.
	trace, err := BuildOpenAPIInventory([]byte(scanSource), "application/json", testInventoryOptions("trace"))
	if err != nil {
		t.Fatal(err)
	}
	for _, task := range trace.Tasks {
		data, err := json.Marshal(task.Document)
		if err != nil || bytes.Contains(data, []byte(`"scan"`)) || task.Document.Operation == nil {
			t.Fatal("legacy task shape changed")
		}
	}
}
