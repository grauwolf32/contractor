package public

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"

	"github.com/getkin/kin-openapi/openapi3"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/contracts/runlabels"
	"github.com/grauwolf32/contractor/internal/projectsitemap"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// utf8ByteBoundOwners binds each shared byte-bounded rule to the Go bound
// that enforces it. LIM-AUD-29 is unexported in auditservice, whose own
// TestSharedAuditTextBoundsCountUTF8Bytes binds it.
var utf8ByteBoundOwners = map[string]int{
	"LIM-SMT-6":  projectsitemap.MaxNoteBytes,
	"LIM-SMT-11": projectsitemap.MaxTagBytes,
	"LIM-RUN-10": maxCancellationReasonBytes,
	"LIM-PRJ-3":  projectstore.MaxNameBytes,
	"LIM-PRJ-5":  projectstore.MaxDescriptionBytes,
	"LIM-REV-11": auditservice.MaxReviewRationale,
	"LIM-RUN-5":  runlabels.MaxRunMetadataLabelValue,
	"LIM-AUD-29": -1,
}

// TestSharedUTF8ByteBoundsAreDeclared holds every public string that the
// Server bounds in UTF-8 bytes to the schema that declares it: the schema
// carries x-max-utf8-bytes equal to its maxLength and to the Go bound, and no
// other schema than these and the login password declares a byte bound.
func TestSharedUTF8ByteBoundsAreDeclared(t *testing.T) {
	document := loadPublicOpenAPI(t)
	_, fields := contractstest.ReadUTF8ByteBoundCases(t)
	listed := map[string]bool{"LoginRequest/properties/password": true}
	for _, field := range fields {
		location := field.Schema + "/" + strings.Join(field.Path, "/")
		listed[location] = true
		bound, owned := utf8ByteBoundOwners[field.Rule]
		if !owned {
			t.Errorf("%s: rule %s has no Go owner", location, field.Rule)
		} else if bound >= 0 && bound != field.MaxBytes {
			t.Errorf("%s: shared bound %d, Go bound %d", location, field.MaxBytes, bound)
		}
		schema := schemaAtPath(document, field.Schema, field.Path)
		if schema == nil {
			t.Errorf("%s: no such schema", location)
			continue
		}
		if declared := fmt.Sprint(schema.Extensions["x-max-utf8-bytes"]); declared != strconv.Itoa(field.MaxBytes) {
			t.Errorf("%s: x-max-utf8-bytes = %s, want %d", location, declared, field.MaxBytes)
		}
		if schema.MaxLength == nil || *schema.MaxLength != uint64(field.MaxBytes) {
			t.Errorf("%s: maxLength = %v, want %d next to the byte bound", location, schema.MaxLength, field.MaxBytes)
		}
	}
	found := map[string]bool{}
	for name, reference := range document.Components.Schemas {
		walkInlineSchemas(reference, name, func(location string, schema *openapi3.Schema) {
			if _, declared := schema.Extensions["x-max-utf8-bytes"]; declared {
				found[location] = true
				if !listed[location] {
					t.Errorf("%s declares x-max-utf8-bytes, but the shared UTF-8 byte bound cases do not list it", location)
				}
			}
		})
	}
	if len(found) != len(listed) {
		t.Errorf("the schema walk found %d byte-bounded strings, want the %d listed", len(found), len(listed))
	}
}

// TestSharedCancellationReasonCountsUTF8Bytes sends each shared character's
// reason at and one byte over the bound: the first cancels the Run and is
// stored exactly, the second is 400 invalid_request and changes nothing.
func TestSharedCancellationReasonCountsUTF8Bytes(t *testing.T) {
	characters, fields := contractstest.ReadUTF8ByteBoundCases(t)
	var bound contractstest.UTF8ByteBound
	for _, field := range fields {
		if field.Schema == "CancelRunRequest" {
			bound = field
		}
	}
	if bound.MaxBytes == 0 {
		t.Fatal("no shared CancelRunRequest bound")
	}
	for index, value := range bound.Values(characters) {
		fixture := newHandlerFixture(t)
		runID := "run-reason-" + strconv.Itoa(index)
		fixture.runs.runs[runID] = runstore.WorkflowRun{
			RunID: runID, OwnerID: "user-1", WorkflowName: "artifact-copy", WorkflowVersion: "1", State: runstore.RunRunning,
		}
		body, err := json.Marshal(map[string]string{"reason": value.Value})
		if err != nil {
			t.Fatal(err)
		}
		response := httptest.NewRecorder()
		fixture.handler.ServeHTTP(response, authenticatedRequest(http.MethodPost, "/v1/runs/"+runID+"/cancel", bytes.NewReader(body)))
		stored := fixture.runs.runs[runID].Cancellation
		switch {
		case value.Within && (response.Code != http.StatusAccepted || stored == nil || stored.Reason == nil || *stored.Reason != value.Value):
			t.Errorf("%q reason of %d bytes = %d %s, want it stored", value.Character, len(value.Value), response.Code, response.Body.String())
		case !value.Within && (response.Code != http.StatusBadRequest || stored != nil):
			t.Errorf("%q reason of %d bytes = %d %s, want 400 and no cancellation", value.Character, len(value.Value), response.Code, response.Body.String())
		}
	}
}

// schemaAtPath follows properties and additionalProperties from a component
// schema.
func schemaAtPath(document *openapi3.T, name string, path []string) *openapi3.Schema {
	reference := document.Components.Schemas[name]
	for index := 0; reference != nil && index < len(path); index++ {
		if reference.Value == nil {
			return nil
		}
		switch path[index] {
		case "properties":
			if index+1 == len(path) {
				return nil
			}
			index++
			reference = reference.Value.Properties[path[index]]
		case "additionalProperties":
			reference = reference.Value.AdditionalProperties.Schema
		case "items":
			reference = reference.Value.Items
		default:
			return nil
		}
	}
	if reference == nil {
		return nil
	}
	return reference.Value
}

// walkInlineSchemas visits a component schema and its inline subschemas with
// their locations; a referenced schema is visited as its own component.
func walkInlineSchemas(reference *openapi3.SchemaRef, location string, visit func(string, *openapi3.Schema)) {
	if reference == nil || reference.Value == nil {
		return
	}
	schema := reference.Value
	visit(location, schema)
	inline := func(child *openapi3.SchemaRef, at string) {
		if child != nil && child.Ref == "" {
			walkInlineSchemas(child, at, visit)
		}
	}
	for name, child := range schema.Properties {
		inline(child, location+"/properties/"+name)
	}
	inline(schema.AdditionalProperties.Schema, location+"/additionalProperties")
	inline(schema.Items, location+"/items")
	for keyword, children := range map[string]openapi3.SchemaRefs{"allOf": schema.AllOf, "oneOf": schema.OneOf, "anyOf": schema.AnyOf} {
		for index, child := range children {
			inline(child, location+"/"+keyword+"/"+strconv.Itoa(index))
		}
	}
}
