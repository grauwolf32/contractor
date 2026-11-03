package scanplan_test

import (
	"bytes"
	"errors"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/scan"
	"github.com/grauwolf32/contractor/internal/scanplan"
)

// goldenPreparations were recorded from the implementation that parsed and
// hashed the source once per selected operation (base d7b825c4). Inventory
// preparation from one parsed source must reproduce every byte of them: the
// canonical inventory digest, every prepared request set or URL target, and
// the scan plans built from them.
var goldenPreparations = map[string]string{
	"json/nuclei/operations": "sha256:341df51408787c731f3d6b645ea44b8455fb882407a8b26869abb0ed37878c7d",
	"json/nuclei/plan":       "sha256:d435ef098492920360b56850c54bfb3fcdc85b1b5eb37d5a9b5ba87a8e1c151a",
	"json/sqlmap/operations": "sha256:55d565199ea5f6471af20b310c7e67f6da77287f2e99dc5c41062c19597b3e17",
	"json/sqlmap/plan":       "sha256:2ca8d4226a83d413529e15dd22f613875068ea2274b55a43564444362407d9da",
	"yaml/nuclei/operations": "sha256:29eba50c159459d28b6b07e2c0d8daec0bb0a428372129a569107dc287078dd5",
	"yaml/nuclei/plan":       "sha256:d435ef098492920360b56850c54bfb3fcdc85b1b5eb37d5a9b5ba87a8e1c151a",
	"yaml/sqlmap/operations": "sha256:40734e307b706af8dda2829663bff7ba14d62aa55011300fda7d2fd8c89c5451",
	"yaml/sqlmap/plan":       "sha256:8357ac4456c90d4abbae16f2c8478d47282b73d4e98bfd38e1c3b8302e283f07",
}

func TestInventoryPreparationKeepsGoldenResults(t *testing.T) {
	got := map[string]string{}
	for _, format := range []string{"json", "yaml"} {
		source := encodeSource(t, reuseDocument(), format)
		mediaType, ref := sourceFormats[format], inventorySourceRef("source-1")
		for _, scanner := range []string{"sqlmap", "nuclei"} {
			key := format + "/" + scanner
			settingsText := map[string]string{"sqlmap": reuseSQLMapSettings, "nuclei": reuseNucleiSettings}[scanner]
			settings, err := scanplan.DecodeAuditScanSettings([]byte(settingsText))
			if err != nil {
				t.Fatal(err)
			}
			inventory, err := prepareScanBatch(t, source, mediaType, ref, settingsText)
			if err != nil {
				t.Fatalf("%s inventory: %v", key, err)
			}
			batch := map[string]scanplan.PreparedAuditOperation{}
			for _, item := range inventory {
				batch[item.Operation] = item
			}
			prepared := []scanplan.PreparedAuditOperation{}
			for _, pointer := range settings.Operations() {
				single, err := settings.PrepareOperation(source, mediaType, ref, pointer)
				if err != nil {
					t.Fatalf("%s %s: %v", key, pointer, err)
				}
				// The Run-time caller reproduces the accepted task from the
				// same bytes with the same result.
				runTime, err := batch[pointer], error(nil)
				if err != nil || !reflect.DeepEqual(runTime, single) {
					t.Fatalf("%s %s: Run-time preparation differs: %v", key, pointer, err)
				}
				prepared = append(prepared, single)
			}
			requireReuseSemantics(t, key, prepared)
			got[key+"/operations"] = digest(encoded(t, prepared))
			got[key+"/plan"] = digest(goldenPlan(t, scanner, prepared[1]))
		}
	}
	if !reflect.DeepEqual(got, goldenPreparations) {
		t.Fatalf("preparation output changed:\n got %#v\nwant %#v", got, goldenPreparations)
	}
}

// requireReuseSemantics states what the golden fixture covers, so a changed
// fixture cannot silently stop covering it. Operations are in pointer order:
// missing, pets get, pets post, wide a, wide b.
func requireReuseSemantics(t *testing.T, key string, prepared []scanplan.PreparedAuditOperation) {
	t.Helper()
	missing := prepared[0]
	if missing.Runnable || !slices.ContainsFunc(missing.Gaps, func(gap scan.PreparationGap) bool { return gap.Code == "missing_required_parameter" }) {
		t.Fatalf("%s: missing concrete value became runnable: %+v", key, missing)
	}
	for _, operation := range prepared[1:] {
		if !operation.Runnable {
			t.Fatalf("%s: %s is not runnable: %+v", key, operation.Operation, operation.Gaps)
		}
	}
	if !strings.HasSuffix(key, "/sqlmap") {
		return
	}
	// Each wide operation needs 3000 of the 4096 reference resolutions.
	for _, operation := range prepared[3:] {
		if len(operation.RequestSet.Requests) != 1 || len(operation.RequestSet.Requests[0].Request.Body) < 3000*len(`"p0000":"x",`) {
			t.Fatalf("%s: %s lost its referenced body", key, operation.Operation)
		}
	}
	post := prepared[2].RequestSet.Requests[0].Request
	if post.Body != `{"name":"Milo"}` || !slices.Contains(post.Headers, scan.HTTPRequestHeader{Name: "authorization", Value: "Bearer golden-credential"}) {
		t.Fatalf("%s: authenticated POST changed: %+v", key, post)
	}
}

// goldenPlan builds the existing scan plan for the prepared pets GET.
func goldenPlan(t *testing.T, scanner string, prepared scanplan.PreparedAuditOperation) []byte {
	t.Helper()
	if prepared.Operation != "#/paths/~1pets~1{id}/get" {
		t.Fatalf("unexpected plan operation %s", prepared.Operation)
	}
	if scanner == "nuclei" {
		policy, bindings := planFixture("scan_nuclei")
		bindings["nuclei"].Template.Execution.Arguments["template_ids"] = contracts.ToolArgumentBinding{Source: "literal", Value: "fixture-http"}
		return marshalPlan(t, buildPlan(t, planInput(prepared.Target.URL+"\n"), policy, bindings, nil))
	}
	wire, err := scan.MarshalHTTPRequestSet(*prepared.RequestSet)
	if err != nil {
		t.Fatal(err)
	}
	policy, bindings := planFixture("scan_sqlmap")
	policy.Tools[0].TestParameters = []string{"q"}
	input := planInput("")
	input.MediaType, input.Data = scan.HTTPRequestSetMediaType, wire
	return marshalPlan(t, buildPlan(t, input, policy, bindings, nil))
}

// Source, selection and reference-work failures keep their codes, in the
// inventory (where the first failing operation in pointer order rejects the
// whole inventory) and in single-operation preparation.
func TestInventoryPreparationKeepsErrorClassification(t *testing.T) {
	valid := encodeSource(t, reuseDocument(), "json")
	ref := inventorySourceRef("source-1")
	selectOnly := func(pointers ...string) string {
		operations := []string{}
		for _, pointer := range pointers {
			operations = append(operations, `"`+pointer+`":{}`)
		}
		return `{"schema":"contractor.audit.openapi-scan-settings.v1","scanner":"sqlmap","server":"https://target.test/api",` +
			`"testParameters":["q"],"operations":{` + strings.Join(operations, ",") + `}}`
	}
	for _, test := range []struct {
		name, mediaType, settings, code string
		source                          []byte
		ref                             contracts.ArtifactRef
	}{
		{"source ref", "application/json", reuseSQLMapSettings, "invalid_source_ref", valid, contracts.ArtifactRef{Namespace: "inputs", Name: "openapi"}},
		{"media type", "text/plain", reuseSQLMapSettings, "unsupported_media_type", valid, ref},
		{"source bytes", "application/json", reuseSQLMapSettings, "source_limit_exceeded", append(slices.Clone(valid), bytes.Repeat([]byte(" "), scanplan.MaxSourceBytes)...), ref},
		{"json syntax", "application/json", reuseSQLMapSettings, "invalid_document", valid[:len(valid)-1], ref},
		{"yaml alias", "application/yaml", reuseSQLMapSettings, "invalid_document", []byte("openapi: &v 3.0.3\nversion: *v\npaths: {}\n"), ref},
		{"version", "application/json", reuseSQLMapSettings, "unsupported_openapi_version", []byte(`{"openapi":"2.0","paths":{}}`), ref},
		{"paths", "application/yaml", reuseSQLMapSettings, "invalid_document", []byte("openapi: 3.0.3\npaths: []\n"), ref},
		{"unknown operation", "application/json", selectOnly("#/paths/~1pets~1{id}/get", "#/paths/~1zzz/get"), "unknown_operation_selection", valid, ref},
		{"selected reference work", "application/json", selectOnly("#/paths/~1heavy/post", "#/paths/~1pets~1{id}/get"), "reference_limit_exceeded", valid, ref},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, err := prepareScanBatch(t, test.source, test.mediaType, test.ref, test.settings)
			var batchFailure *scanplan.PreparationError
			if !errors.As(err, &batchFailure) || batchFailure.Code != test.code {
				t.Fatalf("batch error = %v, want %s", err, test.code)
			}

			settings, err := scanplan.DecodeAuditScanSettings([]byte(test.settings))
			if err != nil {
				t.Fatal(err)
			}
			err = nil
			for _, pointer := range settings.Operations() {
				if _, err = settings.PrepareOperation(test.source, test.mediaType, test.ref, pointer); err != nil {
					break
				}
			}
			var failure *scanplan.PreparationError
			if !errors.As(err, &failure) || failure.Code != test.code {
				t.Fatalf("single-operation error = %v, want %s", err, test.code)
			}
		})
	}
}
