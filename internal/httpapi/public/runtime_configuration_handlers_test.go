package public

import (
	"bytes"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

func TestRuntimeOperationsCreateBindAndKeepCredentialMaterialWriteOnly(t *testing.T) {
	fixture := newHandlerFixture(t)
	secret := "RUNTIME_CREDENTIAL_WRITE_ONLY_CANARY"
	createCredential := runtimeMutationRequest(
		t, http.MethodPost, "/v1/operations/runtime-credentials",
		`{"credentialId":"otel-debug","kind":"otlp-headers@1","material":{"headers":{"authorization":"`+secret+`"}}}`,
		"runtime-credential-create", "", "",
	)
	createdCredential := httptest.NewRecorder()
	fixture.handler.ServeHTTP(createdCredential, createCredential)
	if createdCredential.Code != http.StatusCreated ||
		createdCredential.Header().Get("Cache-Control") != "no-store" ||
		strings.Contains(createdCredential.Body.String(), secret) ||
		strings.Contains(strings.ToLower(createdCredential.Body.String()), "authorization") {
		t.Fatalf("create Runtime credential = %d headers=%v body=%s", createdCredential.Code, createdCredential.Header(), createdCredential.Body.String())
	}

	document := `{"apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig","metadata":{"name":"debug","version":"1"},"spec":{"worker":{"telemetry":{"adapter":"otlp-http@1","endpoint":"https://otel.example/v1/traces","credential":"otel-debug"}}}}`
	publish := runtimeMutationRequest(
		t, http.MethodPost, "/v1/operations/runtime-configs", document,
		"runtime-config-publish", "", "",
	)
	published := httptest.NewRecorder()
	fixture.handler.ServeHTTP(published, publish)
	if published.Code != http.StatusCreated || published.Header().Get("ETag") == "" {
		t.Fatalf("publish RuntimeConfig = %d: %s", published.Code, published.Body.String())
	}
	var resource runtimeConfigResourceResponse
	if err := json.Unmarshal(published.Body.Bytes(), &resource); err != nil {
		t.Fatal(err)
	}
	bindBody, err := json.Marshal(runtimeLabelMutationRequest{Config: resource.Ref})
	if err != nil {
		t.Fatal(err)
	}
	bind := runtimeMutationRequest(
		t, http.MethodPut, "/v1/operations/runtime-labels/debug", string(bindBody),
		"runtime-label-create", "", "*",
	)
	bound := httptest.NewRecorder()
	fixture.handler.ServeHTTP(bound, bind)
	if bound.Code != http.StatusCreated || bound.Header().Get("ETag") != `"1"` {
		t.Fatalf("bind Runtime label = %d headers=%v body=%s", bound.Code, bound.Header(), bound.Body.String())
	}

	for _, path := range []string{
		"/v1/operations/runtime-credentials/otel-debug",
		"/v1/operations/runtime-configs/debug/versions/1",
		"/v1/operations/runtime-labels/debug",
	} {
		request := authenticatedRequest(http.MethodGet, path, bytes.NewReader(nil))
		response := httptest.NewRecorder()
		fixture.handler.ServeHTTP(response, request)
		if response.Code != http.StatusOK || response.Header().Get("Cache-Control") != "no-store" ||
			strings.Contains(response.Body.String(), secret) {
			t.Fatalf("safe read %s = %d headers=%v body=%s", path, response.Code, response.Header(), response.Body.String())
		}
	}
}

func TestRuntimeLabelConcurrentCASAndReplay(t *testing.T) {
	fixture := newHandlerFixture(t)
	refs := make([]runtimeconfig.Ref, 3)
	for index, name := range []string{"base", "winner-a", "winner-b"} {
		document := `{"apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig","metadata":{"name":"` + name + `","version":"1"},"spec":{"worker":{"telemetry":{"adapter":"otlp-http@1","endpoint":"https://otel.example/v1/traces"}}}}`
		request := runtimeMutationRequest(
			t, http.MethodPost, "/v1/operations/runtime-configs", document,
			"publish-"+name, "", "",
		)
		response := httptest.NewRecorder()
		fixture.handler.ServeHTTP(response, request)
		if response.Code != http.StatusCreated {
			t.Fatalf("publish %s = %d: %s", name, response.Code, response.Body.String())
		}
		var resource runtimeConfigResourceResponse
		if err := json.Unmarshal(response.Body.Bytes(), &resource); err != nil {
			t.Fatal(err)
		}
		refs[index] = resource.Ref
	}
	createBody, _ := json.Marshal(runtimeLabelMutationRequest{Config: refs[0]})
	created := httptest.NewRecorder()
	fixture.handler.ServeHTTP(created, runtimeMutationRequest(
		t, http.MethodPut, "/v1/operations/runtime-labels/race", string(createBody),
		"create-race", "", "*",
	))
	if created.Code != http.StatusCreated {
		t.Fatalf("create binding = %d: %s", created.Code, created.Body.String())
	}

	type outcome struct {
		key      string
		body     string
		response *httptest.ResponseRecorder
	}
	outcomes := []outcome{{key: "race-a"}, {key: "race-b"}}
	for index := range outcomes {
		encoded, _ := json.Marshal(runtimeLabelMutationRequest{Config: refs[index+1]})
		outcomes[index].body = string(encoded)
		outcomes[index].response = httptest.NewRecorder()
	}
	var wait sync.WaitGroup
	for index := range outcomes {
		wait.Add(1)
		go func(item *outcome) {
			defer wait.Done()
			fixture.handler.ServeHTTP(item.response, runtimeMutationRequest(
				t, http.MethodPut, "/v1/operations/runtime-labels/race", item.body,
				item.key, `"1"`, "",
			))
		}(&outcomes[index])
	}
	wait.Wait()
	winner := -1
	for index, item := range outcomes {
		switch item.response.Code {
		case http.StatusOK:
			winner = index
		case http.StatusPreconditionFailed:
		default:
			t.Fatalf("CAS response %d = %d: %s", index, item.response.Code, item.response.Body.String())
		}
	}
	if winner < 0 || outcomes[1-winner].response.Code != http.StatusPreconditionFailed {
		t.Fatalf("CAS outcomes = %d, %d", outcomes[0].response.Code, outcomes[1].response.Code)
	}
	replay := httptest.NewRecorder()
	fixture.handler.ServeHTTP(replay, runtimeMutationRequest(
		t, http.MethodPut, "/v1/operations/runtime-labels/race", outcomes[winner].body,
		outcomes[winner].key, `"1"`, "",
	))
	if replay.Code != http.StatusOK || replay.Header().Get("Idempotency-Replayed") != "true" ||
		replay.Header().Get("ETag") != `"2"` {
		t.Fatalf("winner replay = %d headers=%v body=%s", replay.Code, replay.Header(), replay.Body.String())
	}
}

func TestRuntimeCredentialUnknownFieldsAndOversizeFailWithoutMutationOrEcho(t *testing.T) {
	fixture := newHandlerFixture(t)
	secret := "UNKNOWN_FIELD_SECRET_CANARY"
	for name, body := range map[string]string{
		"unknown":  `{"credentialId":"bad","kind":"http-proxy-bearer@1","material":{"token":"value","unexpected":"` + secret + `"}}`,
		"oversize": `{"credentialId":"bad","kind":"http-proxy-bearer@1","material":{"token":"` + strings.Repeat("x", maxJSONRequestSize) + `"}}`,
	} {
		request := runtimeMutationRequest(
			t, http.MethodPost, "/v1/operations/runtime-credentials", body,
			"invalid-"+name, "", "",
		)
		response := httptest.NewRecorder()
		fixture.handler.ServeHTTP(response, request)
		if response.Code != http.StatusBadRequest || strings.Contains(response.Body.String(), secret) {
			t.Fatalf("%s response = %d: %s", name, response.Code, response.Body.String())
		}
	}
	if _, err := fixture.runtimeCredentials.Get(t.Context(), "bad"); !errors.Is(err, credentials.ErrRuntimeCredentialNotFound) {
		t.Fatalf("invalid request mutated credential store: %v", err)
	}
}

func TestCaidoBearerCredentialAPIStoresOnlySafeMetadata(t *testing.T) {
	fixture := newHandlerFixture(t)
	secret := "caido-api-secret-canary"
	request := runtimeMutationRequest(
		t, http.MethodPost, "/v1/operations/runtime-credentials",
		`{"credentialId":"caido-lab","kind":"caido-bearer@1","material":{"token":"`+secret+`"}}`,
		"create-caido-lab", "", "",
	)
	response := httptest.NewRecorder()
	fixture.handler.ServeHTTP(response, request)
	if response.Code != http.StatusCreated || strings.Contains(response.Body.String(), secret) ||
		!strings.Contains(response.Body.String(), `"kind":"caido-bearer@1"`) {
		t.Fatalf("Caido credential response = %d: %s", response.Code, response.Body.String())
	}
	metadata, err := fixture.runtimeCredentials.Get(t.Context(), "caido-lab")
	if err != nil || metadata.Kind != credentials.RuntimeCredentialCaidoBearer {
		t.Fatalf("Caido credential metadata = (%+v, %v)", metadata, err)
	}
}

func TestRuntimeOperationsReturnBoundedInUseDetails(t *testing.T) {
	fixture := newHandlerFixture(t)
	fixture.runtimeConfigs.deleteErr = &runtimeconfig.LabelInUseError{
		RuntimeAgentIDs: []string{strings.Repeat("a", 64)},
	}
	labelDelete := runtimeMutationRequest(
		t, http.MethodDelete, "/v1/operations/runtime-labels/debug", "",
		"delete-in-use-label", `"1"`, "",
	)
	labelResponse := httptest.NewRecorder()
	fixture.handler.ServeHTTP(labelResponse, labelDelete)
	if labelResponse.Code != http.StatusConflict ||
		!strings.Contains(labelResponse.Body.String(), `"code":"runtime_label_in_use"`) ||
		strings.Contains(labelResponse.Body.String(), "RuntimeConfig") {
		t.Fatalf("label in-use response = %d: %s", labelResponse.Code, labelResponse.Body.String())
	}

	fixture.runtimeCredentials.deleteErr = &credentials.RuntimeCredentialInUseError{
		Usage: credentials.RuntimeCredentialUsage{
			BindingLabels: []string{"debug"}, RunIDs: []string{"run-safe"}, AllocationIDs: []string{"allocation-safe"},
		},
	}
	credentialDelete := runtimeMutationRequest(
		t, http.MethodDelete, "/v1/operations/runtime-credentials/otel-debug", "",
		"delete-in-use-credential", "", "",
	)
	credentialResponse := httptest.NewRecorder()
	fixture.handler.ServeHTTP(credentialResponse, credentialDelete)
	if credentialResponse.Code != http.StatusConflict ||
		!strings.Contains(credentialResponse.Body.String(), `"code":"runtime_credential_in_use"`) ||
		!strings.Contains(credentialResponse.Body.String(), `"bindingLabels":["debug"]`) {
		t.Fatalf("credential in-use response = %d: %s", credentialResponse.Code, credentialResponse.Body.String())
	}
}

func runtimeMutationRequest(
	t *testing.T,
	method string,
	path string,
	body string,
	idempotencyKey string,
	ifMatch string,
	ifNoneMatch string,
) *http.Request {
	t.Helper()
	request := authenticatedRequest(method, path, bytes.NewReader([]byte(body)))
	if body != "" {
		request.Header.Set("Content-Type", "application/json")
	}
	if idempotencyKey != "" {
		request.Header.Set("Idempotency-Key", idempotencyKey)
	}
	if ifMatch != "" {
		request.Header.Set("If-Match", ifMatch)
	}
	if ifNoneMatch != "" {
		request.Header.Set("If-None-Match", ifNoneMatch)
	}
	return request
}
