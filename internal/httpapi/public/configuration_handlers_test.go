package public

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/config"
)

func TestConfigurationPublicationListDetailAndReplay(t *testing.T) {
	fixture := newHandlerFixture(t)
	body := []byte(`{
  "name":"ui-worker",
  "version":"2",
  "modelPolicy":{
    "model":"qwen/qwen3.8-27b",
    "maxOutputTokens":4096,
    "maxModelCalls":8,
    "maxToolCalls":16,
    "maxTotalTokens":32768,
    "temperature":0.2
  }
}`)
	publish := func(key string) *httptest.ResponseRecorder {
		request := authenticatedRequest(
			http.MethodPost, "/v1/configurations/model-policies", bytes.NewReader(body),
		)
		request.Header.Set("Content-Type", "application/json")
		request.Header.Set(idempotencyKeyHeader, key)
		response := httptest.NewRecorder()
		fixture.handler.ServeHTTP(response, request)
		return response
	}
	created := publish("publish-ui-worker")
	if created.Code != http.StatusCreated || created.Header().Get("ETag") == "" ||
		created.Header().Get("Idempotency-Replayed") != "" {
		t.Fatalf("publish = %d headers=%v body=%s", created.Code, created.Header(), created.Body.String())
	}
	if revision := fixture.operations.SnapshotOperations().Cursor.Revision; revision != 1 {
		t.Fatalf("configuration publication Operations revision = %d", revision)
	}
	var resource config.ConfigurationResource
	if err := json.Unmarshal(created.Body.Bytes(), &resource); err != nil {
		t.Fatal(err)
	}
	if resource.Ref.Name != "ui-worker" || resource.Ref.Version != "2" ||
		resource.Ref.Kind != config.ConfigurationModelPolicies || resource.Source != config.ConfigurationSourceManaged {
		t.Fatalf("published resource = %+v", resource)
	}

	replayed := publish("publish-ui-worker")
	if replayed.Code != http.StatusCreated || replayed.Header().Get("Idempotency-Replayed") != "true" ||
		replayed.Header().Get("ETag") != created.Header().Get("ETag") {
		t.Fatalf("replay = %d headers=%v body=%s", replayed.Code, replayed.Header(), replayed.Body.String())
	}
	if revision := fixture.operations.SnapshotOperations().Cursor.Revision; revision != 1 {
		t.Fatalf("configuration replay advanced Operations revision to %d", revision)
	}

	detailRequest := authenticatedRequest(
		http.MethodGet, "/v1/configurations/model-policies/ui-worker/versions/2", bytes.NewReader(nil),
	)
	detail := httptest.NewRecorder()
	fixture.handler.ServeHTTP(detail, detailRequest)
	if detail.Code != http.StatusOK || detail.Header().Get("ETag") != created.Header().Get("ETag") ||
		strings.Contains(strings.ToLower(detail.Body.String()), "bearer") {
		t.Fatalf("detail = %d headers=%v body=%s", detail.Code, detail.Header(), detail.Body.String())
	}

	firstPageRequest := authenticatedRequest(
		http.MethodGet, "/v1/configurations/model-policies?limit=1", bytes.NewReader(nil),
	)
	firstPage := httptest.NewRecorder()
	fixture.handler.ServeHTTP(firstPage, firstPageRequest)
	if firstPage.Code != http.StatusOK {
		t.Fatalf("first page = %d: %s", firstPage.Code, firstPage.Body.String())
	}
	var page configurationPageResponse
	if err := json.Unmarshal(firstPage.Body.Bytes(), &page); err != nil {
		t.Fatal(err)
	}
	if len(page.Items) != 1 || !page.Page.HasMore || page.Page.NextCursor == nil {
		t.Fatalf("first page = %+v", page)
	}
	secondPageRequest := authenticatedRequest(
		http.MethodGet,
		"/v1/configurations/model-policies?limit=1&cursor="+url.QueryEscape(*page.Page.NextCursor),
		bytes.NewReader(nil),
	)
	secondPage := httptest.NewRecorder()
	fixture.handler.ServeHTTP(secondPage, secondPageRequest)
	if secondPage.Code != http.StatusOK {
		t.Fatalf("second page = %d: %s", secondPage.Code, secondPage.Body.String())
	}
	if err := json.Unmarshal(secondPage.Body.Bytes(), &page); err != nil || len(page.Items) != 1 || page.Page.HasMore {
		t.Fatalf("second page = %+v, error %v", page, err)
	}

	crossKindRequest := authenticatedRequest(
		http.MethodGet,
		"/v1/configurations/llm-gateways?cursor="+url.QueryEscape(*firstPageResponseCursor(t, firstPage)),
		bytes.NewReader(nil),
	)
	crossKind := httptest.NewRecorder()
	fixture.handler.ServeHTTP(crossKind, crossKindRequest)
	if crossKind.Code != http.StatusBadRequest {
		t.Fatalf("cross-kind cursor = %d: %s", crossKind.Code, crossKind.Body.String())
	}
}

func TestConfigurationPublicationRejectsUnsafeOrMismatchedRequests(t *testing.T) {
	fixture := newHandlerFixture(t)
	tests := []struct {
		name string
		path string
		key  string
		body string
		want int
	}{
		{name: "missing idempotency", path: "/v1/configurations/model-policies", body: `{"name":"p","version":"1","modelPolicy":{"model":"m"}}`, want: http.StatusBadRequest},
		{name: "read only kind", path: "/v1/configurations/agent-templates", key: "readonly", body: `{"name":"p","version":"1","modelPolicy":{"model":"m"}}`, want: http.StatusBadRequest},
		{name: "mismatched body", path: "/v1/configurations/llm-gateways", key: "mismatch", body: `{"name":"p","version":"1","modelPolicy":{"model":"m"}}`, want: http.StatusBadRequest},
		{name: "path escape", path: "/v1/configurations/model-policies", key: "escape", body: `{"name":"../p","version":"1","modelPolicy":{"model":"m"}}`, want: http.StatusBadRequest},
		{name: "explicit zero", path: "/v1/configurations/model-policies", key: "zero", body: `{"name":"p","version":"1","modelPolicy":{"model":"m","maxModelCalls":0}}`, want: http.StatusBadRequest},
		{name: "two bodies", path: "/v1/configurations/model-policies", key: "two", body: `{"name":"p","version":"1","modelPolicy":{"model":"m"},"llmGateway":{"protocol":"openai-compatible@1","url":"https://example.test/v1"}}`, want: http.StatusBadRequest},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			request := authenticatedRequest(http.MethodPost, test.path, bytes.NewReader([]byte(test.body)))
			request.Header.Set("Content-Type", "application/json")
			if test.key != "" {
				request.Header.Set(idempotencyKeyHeader, test.key)
			}
			response := httptest.NewRecorder()
			fixture.handler.ServeHTTP(response, request)
			if response.Code != test.want {
				t.Fatalf("response = %d: %s", response.Code, response.Body.String())
			}
		})
	}
}

func TestConfigurationQueriesRejectHeadAndInvalidKind(t *testing.T) {
	fixture := newHandlerFixture(t)
	for _, target := range []string{
		"/v1/configurations/model-policies",
		"/v1/configurations/model-policies/worker/versions/1",
	} {
		request := authenticatedRequest(http.MethodHead, target, bytes.NewReader(nil))
		response := httptest.NewRecorder()
		fixture.handler.ServeHTTP(response, request)
		if response.Code != http.StatusMethodNotAllowed {
			t.Errorf("HEAD %s = %d", target, response.Code)
		}
	}
	request := authenticatedRequest(http.MethodGet, "/v1/configurations/instructions", bytes.NewReader(nil))
	response := httptest.NewRecorder()
	fixture.handler.ServeHTTP(response, request)
	if response.Code != http.StatusBadRequest {
		t.Fatalf("invalid kind = %d: %s", response.Code, response.Body.String())
	}
}

func firstPageResponseCursor(t *testing.T, response *httptest.ResponseRecorder) *string {
	t.Helper()
	var page configurationPageResponse
	if err := json.Unmarshal(response.Body.Bytes(), &page); err != nil {
		t.Fatal(err)
	}
	return page.Page.NextCursor
}
