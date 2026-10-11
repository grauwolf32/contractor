package privateartifacts

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestPrivateArtifactPagedListHidesBindingsBeforeTheLimit(t *testing.T) {
	repository := newMemoryRepository()
	for i := 0; i < 40; i++ {
		repository.seed("run-a", "builder", fmt.Sprintf("a_%02d", i), "r", nil)
		repository.seed("run-a", "builder", fmt.Sprintf("http.body.%02d", i), "r", nil)
		repository.seed("run-a", "builder", fmt.Sprintf("memory.%02d", i), "r", nil)
		repository.seed("run-a", "builder", fmt.Sprintf("z_%02d", i), "r", nil)
	}
	for _, ref := range []artifacts.ArtifactRef{
		{Namespace: "inputs", Name: "memory.allowed"}, {Namespace: "outputs", Name: "http.body.result"},
		{Namespace: "skills", Name: "likec4"}, {Namespace: "contractor-system", Name: "repeat-request"},
	} {
		repository.seed("run-a", ref.Namespace, ref.Name, "r", nil)
	}
	repository.seed("run-b", "builder", "foreign", "r", nil)
	handler := newTestHandler(t, &fakeRegistry{grant: testGrant("run-a")}, repository)

	want := map[string]bool{"inputs/memory.allowed": true, "outputs/http.body.result": true}
	for i := 0; i < 40; i++ {
		want[fmt.Sprintf("builder/a_%02d", i)] = true
		want[fmt.Sprintf("builder/z_%02d", i)] = true
	}
	assertPagedListing(t, handler, "", 7, want)
	builder := make(map[string]bool)
	for key := range want {
		if key[:len("builder/")] == "builder/" {
			builder[key] = true
		}
	}
	assertPagedListing(t, handler, "builder", 9, builder)
	assertPagedListing(t, handler, "skills", 5, map[string]bool{})
	assertPagedListing(t, handler, "contractor-system", 5, map[string]bool{})
}

// assertPagedListing follows nextCursor and requires every page but the last
// to be full and every wanted binding to appear exactly once.
func assertPagedListing(t *testing.T, handler http.Handler, namespace string, limit int, want map[string]bool) {
	t.Helper()
	seen := make(map[string]bool)
	cursor := ""
	for pages := 0; ; pages++ {
		if pages > 1000 {
			t.Fatal("paged listing did not terminate")
		}
		query := url.Values{"limit": {fmt.Sprint(limit)}}
		if namespace != "" {
			query.Set("namespace", namespace)
		}
		if cursor != "" {
			query.Set("cursor", cursor)
		}
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, trustedRequest(http.MethodGet, "/private/v1/allocations/allocation-1/artifacts?"+query.Encode(), nil))
		var page contracts.ArtifactListResult
		if err := json.Unmarshal(response.Body.Bytes(), &page); err != nil || response.Code != http.StatusOK || page.Validate() != nil {
			t.Fatalf("page %d of %q = %d %s", pages, namespace, response.Code, response.Body.String())
		}
		if len(page.Artifacts) > limit || page.NextCursor != nil && len(page.Artifacts) != limit {
			t.Fatalf("page %d of %q holds %d refs for limit %d", pages, namespace, len(page.Artifacts), limit)
		}
		for _, ref := range page.Artifacts {
			key := ref.Namespace + "/" + ref.Name
			if seen[key] || !want[key] || ref.Revision != nil {
				t.Fatalf("page %d of %q listed %+v unexpectedly or twice", pages, namespace, ref)
			}
			seen[key] = true
		}
		if page.NextCursor == nil {
			break
		}
		cursor = *page.NextCursor
	}
	if len(seen) != len(want) {
		t.Fatalf("listing of %q returned %d of %d bindings", namespace, len(seen), len(want))
	}
}

func TestPrivateArtifactPagedListRejectsInvalidQueriesBeforeStorage(t *testing.T) {
	repository := newMemoryRepository()
	repository.seed("run-a", "builder", "report", "r", nil)
	handler := newTestHandler(t, &fakeRegistry{grant: testGrant("run-a")}, repository)
	encode := func(value string) string { return base64.RawURLEncoding.EncodeToString([]byte(value)) }
	foreignNamespace := encode(`{"v":1,"namespace":"inputs","name":"source"}`)
	for _, query := range []string{
		"", "namespace=builder", "limit=", "limit=bad", "limit=1.5", "limit=1&limit=2",
		"limit=0", "limit=257", "limit=-1", "namespace=&limit=1", "namespace=bad%2Fname&limit=1",
		"limit=1&cursor=", "limit=1&cursor=a%3D", "limit=1&cursor=" + encode("not json"),
		"limit=1&cursor=" + encode(`{"v":2,"namespace":"builder","name":"report"}`),
		"limit=1&cursor=" + encode(`{"v":1,"namespace":"builder"}`),
		"limit=1&cursor=" + encode(`{"v":1,"namespace":"builder","name":"report","extra":1}`),
		"limit=1&cursor=" + encode(`{"v":1,"namespace":"builder","name":"report","name":"other"}`),
		"limit=1&cursor=" + encode(`{"v":1,"namespace":"builder","name":"a b"}`),
		"namespace=builder&limit=1&cursor=" + foreignNamespace,
		"limit=1&cursor=" + foreignNamespace + "&cursor=" + foreignNamespace,
		"limit=1&runId=run-b",
	} {
		t.Run(query, func(t *testing.T) {
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, trustedRequest(http.MethodGet, "/private/v1/allocations/allocation-1/artifacts?"+query, nil))
			var failure errorResponse
			if response.Code != http.StatusBadRequest || json.Unmarshal(response.Body.Bytes(), &failure) != nil ||
				failure.Code != "invalid_request" || failure.Retryable {
				t.Fatalf("invalid list query = %d %s", response.Code, response.Body.String())
			}
		})
	}
	if repository.listCalls != 0 {
		t.Fatalf("invalid list queries reached storage %d times", repository.listCalls)
	}
	// An unfiltered cursor from another Namespace is an ordinary position.
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, trustedRequest(http.MethodGet, "/private/v1/allocations/allocation-1/artifacts?limit=1&cursor="+foreignNamespace, nil))
	if response.Code != http.StatusOK || repository.listCalls != 1 {
		t.Fatalf("unfiltered cursor = %d %s", response.Code, response.Body.String())
	}
}
