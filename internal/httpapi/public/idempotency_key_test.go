package public

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"

	"github.com/getkin/kin-openapi/routers/gorillamux"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/credentials"
)

// TestSharedIdempotencyKeyCasesMatchPublicHeaderParameters holds both public
// Idempotency-Key header parameters to the shared cases, so the OpenAPI
// schema, the generated clients and the Server give every key one verdict.
func TestSharedIdempotencyKeyCasesMatchPublicHeaderParameters(t *testing.T) {
	document := loadPublicOpenAPI(t)
	cases := contractstest.ReadIdempotencyKeyCases(t)
	for _, name := range []string{"IdempotencyKey", "OptionalIdempotencyKey"} {
		parameter := document.Components.Parameters[name]
		if parameter == nil || parameter.Value == nil || parameter.Value.Name != idempotencyKeyHeader ||
			parameter.Value.Schema == nil || parameter.Value.Schema.Value == nil {
			t.Fatalf("components.parameters.%s is not an Idempotency-Key header schema", name)
		}
		schema := parameter.Value.Schema.Value
		if schema.MaxLength == nil || *schema.MaxLength != uint64(contracts.MaxIdempotencyKeyLength) {
			t.Fatalf("%s maxLength = %v, want %d", name, schema.MaxLength, contracts.MaxIdempotencyKeyLength)
		}
		for _, key := range cases.Valid {
			if err := schema.VisitJSON(key); err != nil {
				t.Errorf("%s rejects valid key %q: %v", name, key, err)
			}
		}
		for _, key := range cases.Invalid {
			if err := schema.VisitJSON(key); err == nil {
				t.Errorf("%s accepts invalid key %q", name, key)
			}
		}
	}
}

// TestSharedIdempotencyKeyCasesGateProjectCreation sends every shared key to
// POST /v1/projects: a key outside the grammar, a missing key and every
// duplicate header set are 400 invalid_request and create no Project, and
// every valid key, the 128-byte boundary keys included, creates one.
func TestSharedIdempotencyKeyCasesGateProjectCreation(t *testing.T) {
	serial := 0
	fixture := newHandlerFixtureWithAuth(t, "../../config/testdata/valid", newTestAuthentication(t), mustTestOrigins(t), false, nil,
		func(services *testServices) {
			services.NewID = func(prefix string) (string, error) {
				serial++
				return prefix + strconv.Itoa(serial), nil
			}
		})
	cases := contractstest.ReadIdempotencyKeyCases(t)
	create := func(keys ...string) *httptest.ResponseRecorder {
		request := authenticatedRequest(http.MethodPost, "/v1/projects", bytes.NewReader([]byte(`{"kind":"project","name":"Keyed"}`)))
		request.Header.Set("Content-Type", "application/json")
		for _, key := range keys {
			request.Header.Add(idempotencyKeyHeader, key)
		}
		response := httptest.NewRecorder()
		fixture.handler.ServeHTTP(response, request)
		return response
	}
	rejected := append([][]string{nil}, cases.DuplicateHeaders...)
	for _, key := range cases.Invalid {
		rejected = append(rejected, []string{key})
	}
	for _, keys := range rejected {
		response := create(keys...)
		var failure errorResponse
		if err := json.Unmarshal(response.Body.Bytes(), &failure); err != nil || response.Code != http.StatusBadRequest ||
			failure.Code != "invalid_request" {
			t.Errorf("Idempotency-Key %q = %d %s, want 400 invalid_request", keys, response.Code, response.Body.String())
		}
	}
	if count := len(fixture.projects.projects); count != 0 {
		t.Fatalf("rejected Idempotency-Keys created %d Projects", count)
	}
	for _, key := range cases.Valid {
		if response := create(key); response.Code != http.StatusCreated || response.Header().Get("Idempotency-Replayed") != "" {
			t.Errorf("valid Idempotency-Key %q = %d %s, want a new Project", key, response.Code, response.Body.String())
		}
	}
	if count := len(fixture.projects.projects); count != len(cases.Valid) {
		t.Fatalf("valid Idempotency-Keys created %d Projects, want %d", count, len(cases.Valid))
	}
}

// TestSharedIdempotencyKeyCasesGateEvalMutations sends every managed Eval
// mutation of api/testdata/evals/http-mutations.json with each shared key.
// A key outside the grammar, such as a!b or a/b, and every duplicate header
// set are the route's documented 422 eval_invalid before the Eval service is
// called, while every valid key reaches the service exactly as the
// fixture's own key does.
func TestSharedIdempotencyKeyCasesGateEvalMutations(t *testing.T) {
	fixture := newHandlerFixture(t)
	router, err := gorillamux.NewRouter(loadPublicOpenAPI(t))
	if err != nil {
		t.Fatal(err)
	}
	var mutations []struct {
		Name, Kind, Method, Path string
		Headers                  map[string]string
		Body                     json.RawMessage
		RequiresRevision         bool
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "evals", "http-mutations.json"), &mutations); err != nil {
		t.Fatal(err)
	}
	if len(mutations) == 0 {
		t.Fatal("no managed Eval mutations to check")
	}
	cases := contractstest.ReadIdempotencyKeyCases(t)
	for _, mutation := range mutations {
		send := func(keys []string, validate bool) (*httptest.ResponseRecorder, errorResponse) {
			t.Helper()
			request := newPublicContractRequest(mutation.Method, mutation.Path, mutation.Body)
			for name, value := range mutation.Headers {
				if name != idempotencyKeyHeader {
					request.Header.Set(name, value)
				}
			}
			for _, key := range keys {
				request.Header.Add(idempotencyKeyHeader, key)
			}
			var response *httptest.ResponseRecorder
			if validate {
				response = serveAndValidatePublicContract(t, router, fixture.handler, request, false)
			} else {
				response = httptest.NewRecorder()
				fixture.handler.ServeHTTP(response, request)
			}
			var failure errorResponse
			_ = json.Unmarshal(response.Body.Bytes(), &failure)
			return response, failure
		}
		baseline, baselineFailure := send([]string{mutation.Headers[idempotencyKeyHeader]}, false)
		if baselineFailure.Code == "eval_invalid" {
			t.Fatalf("%s: the fixture request itself is eval_invalid: %s", mutation.Name, baseline.Body.String())
		}
		rejected := append([][]string{nil}, cases.DuplicateHeaders...)
		for _, key := range cases.Invalid {
			rejected = append(rejected, []string{key})
		}
		for index, keys := range rejected {
			response, failure := send(keys, index == 0)
			if response.Code != http.StatusUnprocessableEntity || failure.Code != "eval_invalid" {
				t.Errorf("%s with Idempotency-Key %q = %d %s, want 422 eval_invalid", mutation.Name, keys, response.Code, response.Body.String())
			}
		}
		for _, key := range cases.Valid {
			if response, failure := send([]string{key}, false); response.Code != baseline.Code || failure.Code != baselineFailure.Code {
				t.Errorf("%s with valid Idempotency-Key %q = %d %s, want %d %s like the fixture key",
					mutation.Name, key, response.Code, failure.Code, baseline.Code, baselineFailure.Code)
			}
		}
	}
}

// countingCredentialDeletes counts the RuntimeCredential deletions that reach
// the service.
type countingCredentialDeletes struct {
	*fakeRuntimeCredentialManagement
	deletes *int
}

func (s countingCredentialDeletes) Delete(ctx context.Context, id, actor string) (credentials.RuntimeCredentialDeleteResult, error) {
	*s.deletes++
	return s.fakeRuntimeCredentialManagement.Delete(ctx, id, actor)
}

// TestSharedIdempotencyKeyCasesGateOptionalKeys sends every shared key to
// each operation that takes the OptionalIdempotencyKey parameter (API-R28).
// Such an operation binds no key, so an absent key and every valid key reach
// its service, but a key outside the grammar and every duplicate header set
// are 400 invalid_request before the service is called. The requests here
// are exactly the operations that declare the parameter: one that declares
// it without a request here, or a request whose operation does not declare
// it, fails the test.
func TestSharedIdempotencyKeyCasesGateOptionalKeys(t *testing.T) {
	calls := map[string]*int{"deleteRuntimeCredential": new(int)}
	fixture := newHandlerFixtureWithAuth(t, "../../config/testdata/valid", newTestAuthentication(t), mustTestOrigins(t), false, nil,
		func(services *testServices) {
			services.RuntimeCredentials = countingCredentialDeletes{
				fakeRuntimeCredentialManagement: services.RuntimeCredentials.(*fakeRuntimeCredentialManagement),
				deletes:                         calls["deleteRuntimeCredential"],
			}
		})
	requests := map[string]struct{ method, path string }{
		"deleteRuntimeCredential": {http.MethodDelete, "/v1/operations/runtime-credentials/otel-debug"},
	}
	document := loadPublicOpenAPI(t)
	declared := map[string]bool{}
	for path, item := range document.Paths.Map() {
		for _, operation := range item.Operations() {
			for _, parameter := range operation.Parameters {
				if strings.HasSuffix(parameter.Ref, "/OptionalIdempotencyKey") {
					declared[operation.OperationID] = true
					if _, listed := requests[operation.OperationID]; !listed {
						t.Errorf("%s %s takes OptionalIdempotencyKey but has no request here", operation.OperationID, path)
					}
				}
			}
		}
	}
	router, err := gorillamux.NewRouter(document)
	if err != nil {
		t.Fatal(err)
	}
	for operation, target := range requests {
		route, _, err := router.FindRoute(newPublicContractRequest(target.method, target.path, nil))
		switch {
		case err != nil || route.Operation.OperationID != operation:
			t.Errorf("%s %s is not the %s operation: %v", target.method, target.path, operation, err)
		case !declared[operation]:
			t.Errorf("%s binds no key here but does not declare OptionalIdempotencyKey", operation)
		}
	}
	cases := contractstest.ReadIdempotencyKeyCases(t)
	rejected := cases.DuplicateHeaders
	for _, key := range cases.Invalid {
		rejected = append(rejected, []string{key})
	}
	accepted := [][]string{nil}
	for _, key := range cases.Valid {
		accepted = append(accepted, []string{key})
	}
	for operation, target := range requests {
		send := func(keys []string) (*httptest.ResponseRecorder, errorResponse) {
			request := authenticatedRequest(target.method, target.path, bytes.NewReader(nil))
			for _, key := range keys {
				request.Header.Add(idempotencyKeyHeader, key)
			}
			response := httptest.NewRecorder()
			fixture.handler.ServeHTTP(response, request)
			var failure errorResponse
			_ = json.Unmarshal(response.Body.Bytes(), &failure)
			return response, failure
		}
		for _, keys := range rejected {
			if response, failure := send(keys); response.Code != http.StatusBadRequest || failure.Code != "invalid_request" {
				t.Errorf("%s with Idempotency-Key %q = %d %s, want 400 invalid_request", operation, keys, response.Code, response.Body.String())
			}
		}
		if *calls[operation] != 0 {
			t.Errorf("%s: rejected Idempotency-Keys reached the service %d times", operation, *calls[operation])
		}
		for _, keys := range accepted {
			if response, failure := send(keys); failure.Code == "invalid_request" {
				t.Errorf("%s with Idempotency-Key %q = %d %s, want it to reach the service", operation, keys, response.Code, response.Body.String())
			}
		}
		if *calls[operation] != len(accepted) {
			t.Errorf("%s: %d of %d accepted requests reached the service", operation, *calls[operation], len(accepted))
		}
	}
}

// TestSharedIdempotencyKeyCasesGateEveryKeyedOperation sends each keyed
// operation other than the Eval mutations (Runs, Projects, Audits and their
// reviews, configuration, RuntimeConfig, credentials, Runtime Agent
// principals) a key outside the grammar of the shared cases,
// such as a/b or -a, and every duplicate header set. Each is 400
// invalid_request before its dependency is reached: the dependencies of
// keyedOperationRequests answer any request that reaches them with an
// idempotency conflict, which the well formed key still gets.
func TestSharedIdempotencyKeyCasesGateEveryKeyedOperation(t *testing.T) {
	fixture, audits, requests := keyedOperationRequests(t)
	cases := contractstest.ReadIdempotencyKeyCases(t)
	malformed := cases.DuplicateHeaders
	for _, key := range cases.Invalid {
		malformed = append(malformed, []string{key})
	}
	for operation, keyed := range requests {
		t.Run(operation, func(t *testing.T) {
			send := func(keys ...string) *httptest.ResponseRecorder {
				request := authenticatedRequest(keyed.method, keyed.path, bytes.NewReader([]byte(keyed.body)))
				if keyed.body != "" {
					request.Header.Set("Content-Type", "application/json")
				}
				request.Header.Del(idempotencyKeyHeader) // authenticatedRequest keys POST /v1/runs.
				for _, key := range keys {
					request.Header.Add(idempotencyKeyHeader, key)
				}
				for name, value := range keyed.headers {
					request.Header.Set(name, value)
				}
				response := httptest.NewRecorder()
				fixture.handler.ServeHTTP(response, request)
				return response
			}
			for _, keys := range malformed {
				response := send(keys...)
				var failure errorResponse
				if err := json.Unmarshal(response.Body.Bytes(), &failure); err != nil ||
					response.Code != http.StatusBadRequest || failure.Code != "invalid_request" {
					t.Errorf("Idempotency-Key %q = %d %s, want 400 invalid_request", keys, response.Code, response.Body.String())
				}
			}
			assertIdempotencyConflict(t, send("well-formed"))
		})
	}
	if !audits.reached["EditProjectFinding"] || !audits.reached["CreateDraft"] {
		t.Fatalf("Audit mutations were not reached with a well formed key: %v", audits.reached)
	}
}
