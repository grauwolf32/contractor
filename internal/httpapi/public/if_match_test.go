package public

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

// malformedIfMatch lists If-Match values that are not one strong entity tag
// carrying the one-character revision verbatim: its escaped spellings, other
// quoting, an empty or weak tag and lists (API-R70).
func malformedIfMatch(revision string) []string {
	if len(revision) != 1 {
		panic("malformedIfMatch needs a one-character revision")
	}
	code := revision[0]
	return []string{
		fmt.Sprintf(`"\x%02x"`, code), fmt.Sprintf(`"\%03o"`, code),
		fmt.Sprintf(`"\u%04x"`, code), fmt.Sprintf(`"\U%08x"`, code),
		`'` + revision + `'`, "`" + revision + "`", revision,
		` "` + revision + `"`, `"` + revision + `" `, `"` + revision + `"x`, `"` + revision + `\"`,
		`""`, `"`, `W/"` + revision + `"`, `"` + revision + `", "` + revision + `"`, `*`,
	}
}

// The shared decimal revision reader of Projects, Audits, Audit reviews,
// Scheduler settings, Runtime labels and Runtime Agent labels accepts only
// the plain quoted revision.
func TestRuntimeRevisionETagAcceptsOnlyOneStrongEntityTag(t *testing.T) {
	t.Parallel()
	if revision, err := parseRuntimeRevisionETag(`"1"`); err != nil || revision != 1 {
		t.Fatalf(`parseRuntimeRevisionETag("1") = %d, %v`, revision, err)
	}
	for _, raw := range append(malformedIfMatch("1"), `"0"`, `"01"`, `"+1"`) {
		if revision, err := parseRuntimeRevisionETag(raw); !errors.Is(err, errInvalidRequest) {
			t.Errorf("parseRuntimeRevisionETag(%q) = %d, %v; want invalid request", raw, revision, err)
		}
	}
}

// A Runtime label rebind or delete reads its binding revision from one
// strong entity tag; any other spelling of the current revision is a bad
// request, and the plain quoted revision still works.
func TestRuntimeLabelBindingIfMatchAcceptsOnlyOneStrongEntityTag(t *testing.T) {
	fixture := newHandlerFixture(t)
	ref := fixture.runtimeConfigs.bindings[runtimeconfig.DefaultLabel].Ref
	fixture.runtimeConfigs.bindings["present"] = runtimeconfig.Binding{Label: "present", Ref: ref, Revision: 1}
	body, err := json.Marshal(runtimeLabelMutationRequest{Config: ref})
	if err != nil {
		t.Fatal(err)
	}
	mutate := func(method, body, key, ifMatch string) *httptest.ResponseRecorder {
		t.Helper()
		response := httptest.NewRecorder()
		fixture.handler.ServeHTTP(response, runtimeMutationRequest(
			t, method, "/v1/operations/runtime-labels/present", body, key, ifMatch, "",
		))
		return response
	}
	for index, ifMatch := range malformedIfMatch("1") {
		t.Run(ifMatch, func(t *testing.T) {
			rebind := mutate(http.MethodPut, string(body), fmt.Sprintf("malformed-rebind-%d", index), ifMatch)
			assertErrorEnvelope(t, rebind, http.StatusBadRequest, "invalid_request", false)
			remove := mutate(http.MethodDelete, "", fmt.Sprintf("malformed-delete-%d", index), ifMatch)
			assertErrorEnvelope(t, remove, http.StatusBadRequest, "invalid_request", false)
		})
	}
	if binding, ok := fixture.runtimeConfigs.bindings["present"]; !ok || binding.Revision != 1 {
		t.Fatalf("binding after rejected preconditions = %+v, %t", binding, ok)
	}
	if rebind := mutate(http.MethodPut, string(body), "plain-rebind", `"1"`); rebind.Code != http.StatusOK ||
		rebind.Header().Get("ETag") != `"1"` {
		t.Fatalf("plain quoted rebind = %d headers=%v %s", rebind.Code, rebind.Header(), rebind.Body.String())
	}
	if remove := mutate(http.MethodDelete, "", "plain-delete", `"1"`); remove.Code/100 != 2 {
		t.Fatalf("plain quoted delete = %d %s", remove.Code, remove.Body.String())
	}
	if _, ok := fixture.runtimeConfigs.bindings["present"]; ok {
		t.Fatal("plain quoted delete left the binding")
	}
}

// The owner Queue control reads its revision from one strong entity tag.
func TestOwnerQueueControlIfMatchAcceptsOnlyOneStrongEntityTag(t *testing.T) {
	fixture := newHandlerFixture(t)
	put := func(ifMatch string) *httptest.ResponseRecorder {
		t.Helper()
		request := authenticatedRequest(http.MethodPut, "/v1/queue/control", bytes.NewReader([]byte(`{"paused":true}`)))
		request.Header.Set("Content-Type", "application/json")
		request.Header.Set("If-Match", ifMatch)
		response := httptest.NewRecorder()
		fixture.handler.ServeHTTP(response, request)
		return response
	}
	for _, ifMatch := range malformedIfMatch("0") {
		t.Run(ifMatch, func(t *testing.T) {
			assertErrorEnvelope(t, put(ifMatch), http.StatusBadRequest, "invalid_request", false)
		})
	}
	if _, exists := fixture.runs.queueControls["user-1"]; exists {
		t.Fatal("a malformed If-Match changed the Queue control")
	}
	if response := put(`"0"`); response.Code != http.StatusOK || response.Header().Get("ETag") != `"1"` {
		t.Fatalf("plain quoted Queue control update = %d headers=%v %s",
			response.Code, response.Header(), response.Body.String())
	}
}

// The model selection precondition reads its revision and digest from one
// strong entity tag.
func TestModelSelectionIfMatchAcceptsOnlyOneStrongEntityTag(t *testing.T) {
	t.Parallel()
	digest := strings.Repeat("a", 64)
	precondition := func(ifMatch string) (uint64, string, error) {
		request := httptest.NewRequest(http.MethodPut, "/v1/operations/settings/models", nil)
		request.Header.Set("If-Match", ifMatch)
		return modelSelectionPrecondition(request)
	}
	if revision, catalog, err := precondition(`"2:` + digest + `"`); err != nil || revision != 2 ||
		catalog != "sha256:"+digest {
		t.Fatalf("plain quoted model selection precondition = %d, %q, %v", revision, catalog, err)
	}
	for _, ifMatch := range []string{
		`"\x32:` + digest + `"`, `'2:` + digest + `'`, "`2:" + digest + "`", `2:` + digest,
		` "2:` + digest + `"`, `"2:` + digest + `" `, `""`, `W/"2:` + digest + `"`,
		`"2:` + digest + `", "2:` + digest + `"`, `"2:` + digest + `\"`,
	} {
		if _, _, err := precondition(ifMatch); !errors.Is(err, errInvalidRequest) {
			t.Errorf("model selection precondition %q error = %v; want invalid request", ifMatch, err)
		}
	}
}
