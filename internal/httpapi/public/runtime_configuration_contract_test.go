package public

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"github.com/getkin/kin-openapi/openapi3"
	"github.com/getkin/kin-openapi/routers/gorillamux"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

func TestPublicRuntimeConfigAuthorAndReadContracts(t *testing.T) {
	contract := loadPublicOpenAPI(t)
	author := contract.Paths.Value("/v1/operations/runtime-configs").Post.RequestBody.Value.Content["application/json"].Schema.Value
	read := contract.Components.Schemas["RuntimeConfigResource"].Value.Properties["document"].Value
	exact := `{"gatewayId":"local-litellm","version":"1","digest":"sha256:` + strings.Repeat("a", 64) + `"}`
	for _, test := range []struct {
		name, spec   string
		author, read bool
	}{
		{"selector", `{"worker":{"llmGateway":{"gateway":"local-litellm@1"}}}`, true, false},
		{"resolved gateway", `{"worker":{"llmGateway":{"gateway":` + exact + `}}}`, false, true},
		{"null gateway", `{"worker":{"llmGateway":{"gateway":null}}}`, false, false},
		{"null gateway block", `{"worker":{"llmGateway":null}}`, false, false},
		{"empty gateway block", `{"worker":{"llmGateway":{}}}`, false, false},
		{"empty worker", `{"worker":{}}`, false, false},
		{"empty planner", `{"planner":{}}`, false, false},
		{"built-in empty spec", `{}`, false, true},
		{"credential clear", `{"worker":{"llmGateway":{"credential":null}}}`, true, true},
		{"credential value", `{"worker":{"llmGateway":{"credential":"worker-local"}}}`, true, true},
		{"atomic clears", `{"worker":{"telemetry":null,"httpProxy":null,"caido":null},"planner":{"telemetry":null}}`, true, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			data := []byte(`{"apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig","metadata":{"name":"contract-test","version":"1"},"spec":` + test.spec + `}`)
			var value any
			if err := json.Unmarshal(data, &value); err != nil {
				t.Fatal(err)
			}
			for _, schema := range []struct {
				name  string
				value *openapi3.Schema
				valid bool
			}{{"author", author, test.author}, {"read", read, test.read}} {
				err := schema.value.VisitJSON(value, openapi3.EnableJSONSchema2020())
				if (err == nil) != schema.valid {
					t.Errorf("%s schema valid=%v want %v: %v", schema.name, err == nil, schema.valid, err)
				}
			}
			_, err := runtimeconfig.PreparePublication(data)
			if (err == nil) != test.author {
				t.Errorf("author parser valid=%v want %v: %v", err == nil, test.author, err)
			}
		})
	}
}

func TestPublicRuntimeConfigPublicationResolvesGatewayAndPreservesClears(t *testing.T) {
	contract := loadPublicOpenAPI(t)
	router, err := gorillamux.NewRouter(contract)
	if err != nil {
		t.Fatal(err)
	}
	fixture := newHandlerFixture(t)
	// Fake storage delegates publication and resolution to the production parser;
	// these are handler/contract fixtures, not a PostgreSQL durability check.
	data := `{"apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig","metadata":{"name":"contract-clear","version":"1"},"spec":{"worker":{"llmGateway":{"gateway":"local-litellm@1","credential":null},"telemetry":null,"httpProxy":null,"caido":null},"planner":{"telemetry":null}}}`
	request := newPublicContractRequest(http.MethodPost, "/v1/operations/runtime-configs", []byte(data))
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Idempotency-Key", "contract-runtime-author")
	response := serveAndValidatePublicContract(t, router, fixture.handler, request, true)
	if response.Code != http.StatusCreated {
		t.Fatalf("publication status=%d body=%s", response.Code, response.Body.String())
	}
	var resource runtimeConfigResourceResponse
	if err := json.Unmarshal(response.Body.Bytes(), &resource); err != nil {
		t.Fatal(err)
	}
	stored, err := runtimeconfig.DecodeStoredDocument(resource.Document)
	if err != nil {
		t.Fatal(err)
	}
	worker := stored.Spec.Worker
	if worker.LLMGateway.Gateway.Value.GatewayID != "local-litellm" || worker.LLMGateway.Gateway.Value.Version != "1" || !worker.LLMGateway.Credential.Clear || !worker.Telemetry.Clear || !worker.HTTPProxy.Clear || !worker.Caido.Clear || !stored.Spec.Planner.Telemetry.Clear {
		t.Fatalf("resolved patch semantics changed: %+v", stored.Spec)
	}
	if _, err := runtimeconfig.PreparePublication(resource.Document); err == nil {
		t.Fatal("normalized read document incorrectly accepted for publication")
	}
	get := newPublicContractRequest(http.MethodGet, "/v1/operations/runtime-configs/contract-clear/versions/1", nil)
	read := serveAndValidatePublicContract(t, router, fixture.handler, get, true)
	if read.Code != http.StatusOK {
		t.Fatalf("read status=%d body=%s", read.Code, read.Body.String())
	}
	exact, err := json.Marshal(worker.LLMGateway.Gateway.Value)
	if err != nil {
		t.Fatal(err)
	}
	for _, spec := range []string{
		`{"worker":{"llmGateway":null}}`,
		`{"worker":{"llmGateway":{"gateway":null}}}`,
		`{"worker":{"llmGateway":{"gateway":` + string(exact) + `}}}`,
	} {
		var invalid map[string]any
		if err := json.Unmarshal([]byte(data), &invalid); err != nil {
			t.Fatal(err)
		}
		var patch any
		if err := json.Unmarshal([]byte(spec), &patch); err != nil {
			t.Fatal(err)
		}
		invalid["spec"] = patch
		encoded, err := json.Marshal(invalid)
		if err != nil {
			t.Fatal(err)
		}
		reject := newPublicContractRequest(http.MethodPost, "/v1/operations/runtime-configs", encoded)
		reject.Header.Set("Content-Type", "application/json")
		reject.Header.Set("Idempotency-Key", "contract-invalid-gateway")
		rejected := serveAndValidatePublicContract(t, router, fixture.handler, reject, false)
		if rejected.Code != http.StatusBadRequest {
			t.Fatalf("invalid gateway status=%d body=%s", rejected.Code, rejected.Body.String())
		}
	}
}
