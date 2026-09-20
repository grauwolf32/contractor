package scanplan_test

import (
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/scanplan"
)

func TestPrepareAssignedOperationDoesNotScanOrSpendBudgetOnNeighbors(t *testing.T) {
	data := encoded(t, document(map[string]any{
		"/a":      map[string]any{"get": map[string]any{}},
		"/broken": map[string]any{"$ref": "https://never-fetch.invalid/path"},
		"/z": map[string]any{
			"get":  map[string]any{"parameters": []any{map[string]any{"in": "query", "name": "id", "example": 7}}},
			"post": map[string]any{"requestBody": map[string]any{"required": true}},
		},
	}))
	ref := sourceRef()
	set, err := scanplan.PrepareOperation(data, "application/json", ref, scanplan.Options{MaxRequests: 1}, "#/paths/~1z/get")
	if err != nil {
		t.Fatal(err)
	}
	if len(set.Requests) != 1 || set.Requests[0].Request.URL != "https://api.example.test/base/z?id=7" ||
		!reflect.DeepEqual(set.Requests[0].Origins, []contracts.RequestOrigin{{Pointer: "#/paths/~1z/get"}}) ||
		set.Coverage.Operations != 1 || set.Coverage.Prepared != 1 || !set.Coverage.Complete || len(set.Gaps) != 0 {
		t.Fatalf("assigned operation was broadened or affected by neighbors: %+v", set)
	}
	if set.Source.ContentDigest != digest(data) || !reflect.DeepEqual(set.Source.Artifact, ref) {
		t.Fatal("preparation lost the full exact source identity")
	}
	*ref.Revision = "changed-by-caller"
	if *set.Source.Artifact.Revision != "source-1" {
		t.Fatal("source revision aliases caller state")
	}
	policy, bindings := planFixture("scan_sqlmap")
	wire, err := contracts.MarshalHTTPRequestSet(set)
	if err != nil {
		t.Fatal(err)
	}
	input := planInput("")
	input.MediaType, input.Data = contracts.HTTPRequestSetMediaType, wire
	plan := buildPlan(t, input, policy, bindings, nil)
	if len(plan.Jobs) != 1 || plan.Jobs[0].Request == nil || plan.Jobs[0].Request.URL != set.Requests[0].Request.URL ||
		!reflect.DeepEqual(plan.Jobs[0].Request.TestParameters, []string{"id"}) {
		t.Fatalf("assigned request did not reach the existing SQLMap planner: %+v", plan)
	}
}

func TestPrepareAssignedOperationRejectsUnknownAndForeignSelections(t *testing.T) {
	data := encoded(t, document(map[string]any{"/pets": map[string]any{"get": map[string]any{}}}))
	for _, pointer := range []string{"", "#/paths/~1pets/GET", "#/paths/~2pets/get", "#/paths//pets/get", "#/paths/~1missing/get", "#/paths/~1pets/post"} {
		t.Run(pointer, func(t *testing.T) {
			_, err := scanplan.PrepareOperation(data, "application/json", sourceRef(), scanplan.Options{}, pointer)
			var failure *scanplan.PreparationError
			if !errors.As(err, &failure) || !strings.Contains(failure.Code, "operation_selection") {
				t.Fatalf("invalid assignment accepted: %v", err)
			}
		})
	}
	_, err := scanplan.PrepareOperation(data, "application/json", sourceRef(), scanplan.Options{
		Operations: map[string]scanplan.OperationInput{"#/paths/~1other/get": {}},
	}, "#/paths/~1pets/get")
	if err == nil || err.Error() != "request preparation: unassigned_operation_binding" {
		t.Fatalf("foreign operation settings accepted: %v", err)
	}
}

func TestAssignedSQLMapRequestKeepsPostBodyAndAuthentication(t *testing.T) {
	pointer := "#/paths/~1pets/post"
	doc := document(map[string]any{"/pets": map[string]any{"post": map[string]any{
		"security": []any{map[string]any{"bearerAuth": []any{}}},
		"requestBody": map[string]any{"required": true, "content": map[string]any{
			"application/json": map[string]any{"example": map[string]any{"id": 7}},
		}},
	}}})
	doc["components"] = map[string]any{"securitySchemes": map[string]any{
		"bearerAuth": map[string]any{"type": "http", "scheme": "bearer"},
	}}
	set, err := scanplan.PrepareOperation(encoded(t, doc), "application/json", sourceRef(), scanplan.Options{
		Server:         "http://target.test/api",
		Authentication: map[string]contracts.SecretString{"bearerAuth": contracts.NewSecretString("fixture-token")},
	}, pointer)
	if err != nil {
		t.Fatal(err)
	}
	wire, err := contracts.MarshalHTTPRequestSet(set)
	if err != nil {
		t.Fatal(err)
	}
	input := planInput("")
	input.MediaType, input.Data = contracts.HTTPRequestSetMediaType, wire
	policy, bindings := planFixture("scan_sqlmap")
	plan := buildPlan(t, input, policy, bindings, nil)
	if len(plan.Jobs) != 1 || plan.Jobs[0].Request == nil {
		t.Fatalf("authenticated request was not selected: %+v", plan)
	}
	request := plan.Jobs[0].Request
	if request.Method != "POST" || request.URL != "http://target.test/api/pets" || request.Body != `{"id":7}` ||
		!reflect.DeepEqual(request.Headers, []contracts.HTTPRequestHeader{
			{Name: "authorization", Value: "Bearer fixture-token"}, {Name: "content-type", Value: "application/json"},
		}) || !set.Coverage.Complete {
		t.Fatal("SQLMap request changed its method, body, headers or target")
	}
}

func TestPrepareAssignedOperationRetainsFailureAsOneGap(t *testing.T) {
	for name, path := range map[string]any{
		"unresolved path": map[string]any{"$ref": "https://never-fetch.invalid/path"},
		"missing value": map[string]any{"get": map[string]any{"parameters": []any{
			map[string]any{"in": "query", "name": "id", "required": true, "schema": map[string]any{"type": "string"}},
		}}},
	} {
		t.Run(name, func(t *testing.T) {
			data := encoded(t, document(map[string]any{"/pets": path}))
			set, err := scanplan.PrepareOperation(data, "application/json", sourceRef(), scanplan.Options{}, "#/paths/~1pets/get")
			if err != nil {
				t.Fatal(err)
			}
			if len(set.Requests) != 0 || set.Coverage.Operations != 1 || set.Coverage.Skipped != 1 || set.Coverage.Complete || len(set.Gaps) == 0 {
				t.Fatalf("missing request was reported as completed: %+v", set)
			}
		})
	}
}

func TestPrepareNucleiTargetPinsURLWithoutClaimingPostOrAuthReplay(t *testing.T) {
	pointer := "#/paths/~1pets~1{id}/post"
	data := encoded(t, document(map[string]any{"/pets/{id}": map[string]any{"post": map[string]any{
		"parameters": []any{
			map[string]any{"in": "path", "name": "id", "required": true},
			map[string]any{"in": "query", "name": "q", "example": "a b"},
			map[string]any{"in": "header", "name": "X-Account", "required": true},
		},
		"security":    []any{map[string]any{"bearerAuth": []any{}}},
		"requestBody": map[string]any{"required": true, "content": map[string]any{}},
	}}}))
	options := scanplan.Options{
		Server:     "http://target.test/v2",
		Operations: map[string]scanplan.OperationInput{pointer: {Parameters: map[string]any{"path:id": "7/8"}}},
	}
	target, err := scanplan.PrepareOperationTarget(data, "application/json", sourceRef(), options, pointer)
	if err != nil {
		t.Fatal(err)
	}
	wantURL := "http://target.test/v2/pets/7%2F8?q=a%20b"
	if target.URL != wantURL || target.Operation != pointer || target.Source.ContentDigest != digest(data) {
		t.Fatalf("target/provenance changed: %+v", target)
	}
	for _, code := range []string{"url_template_scan_only", "http_method_not_replayed", "request_body_not_replayed", "authentication_not_applied", "non_url_parameter_not_applied"} {
		found := false
		for _, gap := range target.Gaps {
			found = found || gap.Code == code && gap.Pointer == pointer
		}
		if !found {
			t.Fatalf("missing URL-interface limitation %s: %+v", code, target.Gaps)
		}
	}
	policy, bindings := planFixture("scan_nuclei")
	bindings["nuclei"].Template.Execution.Arguments["template_ids"] = contracts.ToolArgumentBinding{Source: "literal", Value: "fixture-http"}
	plan := buildPlan(t, planInput(target.URL+"\n"), policy, bindings, nil)
	if len(plan.Jobs) != 1 || plan.Jobs[0].Parameters["target"] != wantURL || plan.Jobs[0].Request != nil {
		t.Fatalf("Nuclei did not receive the fixed URL: %+v", plan)
	}
	if plan.Jobs[0].Execution.Arguments["template_ids"].Value != "fixture-http" {
		t.Fatal("Nuclei template selection was not retained")
	}
	bindings["nuclei"].Template.Execution.Arguments["template_ids"] = contracts.ToolArgumentBinding{Source: "literal", Value: "fixture-other"}
	changed := buildPlan(t, planInput(target.URL+"\n"), policy, bindings, nil)
	if changed.ID == plan.ID || changed.Jobs[0].ID == plan.Jobs[0].ID {
		t.Fatal("changing Nuclei templates did not change pinned plan identity")
	}
	// URL preparation does not weaken the separate full-request/auth contract.
	request, err := scanplan.PrepareOperation(data, "application/json", sourceRef(), options, pointer)
	if err != nil || len(request.Requests) != 0 || request.Coverage.Complete {
		t.Fatalf("unsupported authenticated request was silently converted: %+v, %v", request, err)
	}
}

func TestNucleiTargetDoesNotGuessPathValuesOrDropExplicitCredentials(t *testing.T) {
	pointer := "#/paths/~1pets~1{id}/get"
	data := encoded(t, document(map[string]any{"/pets/{id}": map[string]any{"get": map[string]any{
		"parameters": []any{map[string]any{"in": "path", "name": "id", "schema": map[string]any{"type": "integer"}}},
	}}}))
	target, err := scanplan.PrepareOperationTarget(data, "application/json", sourceRef(), scanplan.Options{}, pointer)
	if err != nil || target.URL != "" || len(target.Gaps) == 0 {
		t.Fatalf("invented a concrete target: %+v, %v", target, err)
	}
	for name, options := range map[string]scanplan.Options{
		"auth":   {Authentication: map[string]contracts.SecretString{"bearer": contracts.NewSecretString("never-echo-token")}},
		"header": {Operations: map[string]scanplan.OperationInput{pointer: {Parameters: map[string]any{"header:Authorization": "never-echo-token"}}}},
		"cookie": {Operations: map[string]scanplan.OperationInput{pointer: {Parameters: map[string]any{"cookie:session": "never-echo-token"}}}},
		"body":   {Operations: map[string]scanplan.OperationInput{pointer: {Body: &scanplan.BodyInput{MediaType: "text/plain", Value: "never-echo-token"}}}},
	} {
		t.Run(name, func(t *testing.T) {
			target, err := scanplan.PrepareOperationTarget(data, "application/json", sourceRef(), options, pointer)
			if err == nil || err.Error() != "request preparation: unsupported_url_target_binding" || target.URL != "" {
				t.Fatalf("unsupported request binding was silently dropped: %v", err)
			}
		})
	}
}

func TestOperationPreparationIdentitySeparatesSourceAssignmentAndMode(t *testing.T) {
	data := encoded(t, document(map[string]any{"/pets": map[string]any{"get": map[string]any{}, "post": map[string]any{}}}))
	ref := sourceRef()
	get, err := scanplan.PrepareOperation(data, "application/json", ref, scanplan.Options{}, "#/paths/~1pets/get")
	if err != nil {
		t.Fatal(err)
	}
	again, err := scanplan.PrepareOperation(data, "application/json", ref, scanplan.Options{}, "#/paths/~1pets/get")
	if err != nil || !reflect.DeepEqual(get, again) {
		t.Fatal("identical preparation was not deterministic")
	}
	post, err := scanplan.PrepareOperation(data, "application/json", ref, scanplan.Options{}, "#/paths/~1pets/post")
	if err != nil {
		t.Fatal(err)
	}
	target, err := scanplan.PrepareOperationTarget(data, "application/json", ref, scanplan.Options{}, "#/paths/~1pets/get")
	if err != nil {
		t.Fatal(err)
	}
	all, err := scanplan.Prepare(data, "application/json", ref, scanplan.Options{})
	if err != nil {
		t.Fatal(err)
	}
	*ref.Revision = "source-2"
	newRevision, err := scanplan.PrepareOperation(data, "application/json", ref, scanplan.Options{}, "#/paths/~1pets/get")
	if err != nil {
		t.Fatal(err)
	}
	seen := map[string]bool{}
	for _, identity := range []string{get.PreparationDigest, post.PreparationDigest, target.PreparationDigest, all.PreparationDigest, newRevision.PreparationDigest} {
		if identity == "" || seen[identity] {
			t.Fatal("preparation identity lost mode, operation or exact revision")
		}
		seen[identity] = true
	}
}
